import copy
import enum
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import build_source_metadata, get_filename_for_bytesio
from dynamiq.connections import AWSTextract as AWSTextractConnection
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import generate_uuid

# https://docs.aws.amazon.com/textract/latest/dg/limits-document.html
SYNC_MAX_BYTES = 10 * 1024 * 1024
SYNC_MAX_PAGES = 1
ASYNC_MAX_PAGES = 3000
ASYNC_MAX_IMAGE_BYTES = 10 * 1024 * 1024
ASYNC_MAX_PDF_TIFF_BYTES = 500 * 1024 * 1024

# Staging is infrastructure owned by Dynamiq rather than part of the converter's public contract.
# The bucket name still includes account and region because S3 bucket names are globally unique and
# Textract requires its input bucket to be in the same region as the API call.
_STAGING_BUCKET_PREFIX = "dynamiq-textract-staging"
_STAGING_S3_PREFIX = "dynamiq/textract"
_STAGED_OBJECT_EXPIRATION_DAYS = 1

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# WORD/CELL/KEY_VALUE_SET text is reached through relationships, never rendered standalone.
_TEXT_BLOCK_TYPES = {"LINE"}


class TextractFeatureType(str, enum.Enum):
    """Lowercase public feature values mapped to AWS's uppercase request values."""

    TABLES = "tables"
    FORMS = "forms"
    QUERIES = "queries"
    SIGNATURES = "signatures"
    LAYOUT = "layout"

    def to_api(self) -> str:
        """Return the value expected by Textract's FeatureTypes parameter."""
        return self.value.upper()


class TextractProcessingMode(str, enum.Enum):
    """How a document is submitted to Textract.

    - ``auto``: use the synchronous API for single-page documents and images, and switch to the
      asynchronous API (via S3) for multi-page documents.
    - ``sync``: always call the synchronous API. Multi-page PDFs are split locally and each page
      is sent as its own single-page request.
    - ``async``: always call the asynchronous API. Direct S3 inputs are used in place; local files
      are uploaded temporarily to Dynamiq's internal staging bucket.
    """

    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


class TextractQuery(BaseModel):
    """A natural-language question answered against the document (QUERIES feature).

    Attributes:
        text (str): The question, in English. Textract allows up to 15 queries per page
            synchronously and 30 asynchronously.
        alias (str | None): Stable key for the answer in document metadata. Defaults to the
            question text when omitted.
        pages (list[str] | None): Pages the query applies to, e.g. ``["1", "2-4", "*"]``.
    """

    text: str
    alias: str | None = None
    pages: list[str] | None = None

    def to_api(self) -> dict[str, Any]:
        query: dict[str, Any] = {"Text": self.text}
        if self.alias:
            query["Alias"] = self.alias
        if self.pages:
            query["Pages"] = self.pages
        return query


class TextractAdapter(BaseModel):
    """A trained Textract adapter applied to the request.

    Attributes:
        adapter_id (str): The adapter identifier.
        version (str): The adapter version.
        pages (list[str] | None): Pages the adapter applies to, e.g. ``["1", "*"]``.
    """

    adapter_id: str
    version: str
    pages: list[str] | None = None

    def to_api(self) -> dict[str, Any]:
        adapter: dict[str, Any] = {"AdapterId": self.adapter_id, "Version": self.version}
        if self.pages:
            adapter["Pages"] = self.pages
        return adapter


class TextractS3Object(BaseModel):
    """An S3-resident document to analyze.

    Attributes:
        bucket (str): The S3 bucket name.
        name (str): The object key.
        version (str | None): Optional object version id.
    """

    bucket: str
    name: str
    version: str | None = None

    def to_api(self) -> dict[str, Any]:
        s3_object: dict[str, Any] = {"Bucket": self.bucket, "Name": self.name}
        if self.version:
            s3_object["Version"] = self.version
        return s3_object

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.name}"


class TextractNotificationChannel(BaseModel):
    """SNS topic notified when an asynchronous Textract job completes.

    Attributes:
        sns_topic_arn (str): The topic ARN.
        role_arn (str): IAM role ARN Textract assumes to publish to the topic.
    """

    sns_topic_arn: str
    role_arn: str

    def to_api(self) -> dict[str, Any]:
        return {"SNSTopicArn": self.sns_topic_arn, "RoleArn": self.role_arn}


class TextractError(Exception):
    """Raised when a Textract request or job fails."""


class TextractFileConverter(BaseConverter):
    """Converts documents to Dynamiq Documents using Amazon Textract OCR.

    Textract returns a graph of typed blocks (pages, lines, words, tables, cells, key-value
    pairs, query answers, signatures and layout regions) rather than text. This converter walks
    that graph and renders Markdown in reading order, so the output slots into the same
    Document/metadata contract as every other Dynamiq converter.

    Feature-driven output:
        - ``LAYOUT``: headings, lists, figures and reading order come from Textract's layout
          regions, which is the highest-fidelity mode for reports and forms.
        - ``TABLES``: tables are rendered as Markdown tables (merged cells honored) and are also
          exposed structurally under ``metadata["tables"]``.
        - ``FORMS``: key-value pairs are appended as a "Form fields" section and exposed under
          ``metadata["forms"]``.
        - ``QUERIES``: answers to ``queries`` are appended as a "Query answers" section and
          exposed under ``metadata["queries"]``.
        - ``SIGNATURES``: detected signatures are exposed under ``metadata["signatures"]``.
        With no features requested, the cheaper ``DetectDocumentText`` API is used and plain
        lines are returned.

    Usage example:
        ```python
        from dynamiq.components.converters.textract import TextractFeatureType, TextractFileConverter

        converter = TextractFileConverter(feature_types=[TextractFeatureType.LAYOUT, TextractFeatureType.TABLES])
        documents = converter.run(file_paths=["invoice.pdf"])["documents"]
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: AWSTextractConnection | None = None
    client: Any | None = Field(default=None, exclude=True)
    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE, DocumentCreationMode.ONE_DOC_PER_PAGE] = (
        DocumentCreationMode.ONE_DOC_PER_FILE
    )
    feature_types: list[TextractFeatureType] = Field(default_factory=list)
    queries: list[TextractQuery] = Field(default_factory=list)
    adapters: list[TextractAdapter] = Field(default_factory=list)
    processing_mode: TextractProcessingMode = TextractProcessingMode.ASYNC
    separator: str = "\n\n"

    render_tables_as_markdown: bool = True
    include_form_section: bool = True
    include_query_section: bool = True
    skip_headers_and_footers: bool = False
    min_confidence: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Drop text blocks whose Textract confidence (0-100) is below this threshold.",
    )

    notification_channel: TextractNotificationChannel | None = None
    kms_key_id: str | None = None
    job_tag: str | None = None
    poll_interval_seconds: float = Field(default=2.0, gt=0)
    max_poll_seconds: float = Field(default=900.0, gt=0)

    # Private so a derived bucket name never leaks into `to_dict()` as if it were configured.
    _resolved_staging_bucket: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_configuration(self):
        """Reject query configurations Textract would reject server-side."""
        if self.queries and TextractFeatureType.QUERIES not in self.feature_types:
            raise ValueError("`queries` requires TextractFeatureType.QUERIES in `feature_types`.")
        if TextractFeatureType.QUERIES in self.feature_types and not self.queries:
            raise ValueError("TextractFeatureType.QUERIES requires at least one entry in `queries`.")
        return self

    def __init__(self, *args, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSTextractConnection()
        super().__init__(**kwargs)

    def _get_client(self):
        """Return the Textract client, building it from the connection on first use."""
        if self.client is None:
            if self.connection is None:
                raise TextractError("TextractFileConverter requires either a `connection` or a `client`.")
            self.client = self.connection.connect()
        return self.client

    def _get_s3_client(self):
        """Return an S3 client for staging local files ahead of asynchronous analysis."""
        if self.connection is None:
            raise TextractError("Staging a document in S3 requires a `connection` with AWS credentials.")
        return self.connection.get_client("s3")

    def run(
        self,
        file_paths: list[str] | list[Path] | None = None,
        files: list[BytesIO] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        s3_objects: list[TextractS3Object] | list[dict] | None = None,
    ) -> dict[str, list[Any]]:
        """Convert local files and/or S3-resident documents to Documents.

        Extends :meth:`BaseConverter.run` with ``s3_objects``, which Textract can analyze without
        the file ever being read locally (and which is the only supported path for documents
        above the 10 MB synchronous limit).
        """
        if not file_paths and not files and not s3_objects:
            raise ValueError("No input provided. Please provide either file_paths, files or s3_objects.")

        documents: list[Document] = []

        if file_paths or files:
            documents.extend(super().run(file_paths=file_paths, files=files, metadata=metadata)["documents"])

        if s3_objects:
            parsed_objects = [
                s3_object if isinstance(s3_object, TextractS3Object) else TextractS3Object(**s3_object)
                for s3_object in s3_objects
            ]
            if isinstance(metadata, list) and (file_paths or files):
                raise ValueError(
                    "When `s3_objects` is combined with `file_paths`/`files`, `metadata` can only be a "
                    "dictionary (applied to every input). To give each input its own metadata, run the "
                    "local and S3 inputs as separate calls."
                )
            meta_list = self._normalize_metadata(metadata, len(parsed_objects))
            for s3_object, meta in zip(parsed_objects, meta_list):
                documents.extend(self._process_s3_object(s3_object, meta))

        if not documents:
            raise ValueError(
                "No documents were created from the provided inputs. Please check your files and try again."
            )

        return {"documents": documents}

    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """Analyze a single local file and create Documents from the returned blocks."""
        if isinstance(file, (Path, str)):
            file_path = str(file)
            with open(file, "rb") as opened_file:
                content = opened_file.read()
        elif isinstance(file, BytesIO):
            file_path = get_filename_for_bytesio(file)
            file.seek(0)
            content = file.read()
        else:
            raise TypeError("Expected a Path object or a BytesIO object.")

        if not content:
            raise ValueError(f"Empty file cannot be processed: {file_path}")

        extension = Path(file_path).suffix.lower()
        if extension and extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Amazon Textract does not support '{extension}' files. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        blocks, response_metadata = self._analyze_bytes(content, file_path)

        return self._create_documents(
            filepath=file_path,
            elements=blocks,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
            response_metadata=response_metadata,
        )

    def _process_s3_object(self, s3_object: TextractS3Object, metadata: dict[str, Any]) -> list[Document]:
        """Analyze an S3-resident document and create Documents from the returned blocks."""
        if self.processing_mode == TextractProcessingMode.SYNC:
            blocks, response_metadata = self._analyze_sync({"S3Object": s3_object.to_api()})
        else:
            blocks, response_metadata = self._analyze_async(s3_object)

        response_metadata["textract_source_uri"] = s3_object.uri

        return self._create_documents(
            filepath=s3_object.name,
            elements=blocks,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
            response_metadata=response_metadata,
        )

    def _analyze_bytes(self, content: bytes, file_path: str) -> tuple[list[dict], dict[str, Any]]:
        """Route raw document bytes to the synchronous or asynchronous Textract API.

        The synchronous API takes a single page held in memory, so longer or larger documents go
        through the asynchronous API, which reads from S3 and therefore needs the document staged
        there. When staging is impossible, a multi-page PDF is split locally instead and each page
        sent as its own synchronous request; an oversized one has no fallback and raises.
        """
        extension = Path(file_path).suffix.lower()
        page_count = self._count_pdf_pages(content) if extension == ".pdf" else 1
        oversized = len(content) > SYNC_MAX_BYTES
        # Counting TIFF frames would add a heavyweight image dependency. Conservatively send TIFF
        # through async in AUTO mode, where both single-page and multi-page files are supported.
        exceeds_sync_limits = page_count > SYNC_MAX_PAGES or oversized or extension in {".tif", ".tiff"}

        if self.processing_mode == TextractProcessingMode.ASYNC or (
            self.processing_mode == TextractProcessingMode.AUTO and exceeds_sync_limits
        ):
            if page_count > ASYNC_MAX_PAGES:
                raise ValueError(
                    f"'{file_path}' has {page_count} pages, above the Textract limit of {ASYNC_MAX_PAGES}."
                )
            self._validate_async_file_size(content, file_path, extension)
            return self._analyze_async_with_local_fallback(content, file_path, page_count, oversized)

        if oversized:
            raise ValueError(
                f"'{file_path}' is {len(content)} bytes, above the {SYNC_MAX_BYTES}-byte limit of the "
                "synchronous Textract API. Use async processing or pass the document via `s3_objects`."
            )

        if page_count > SYNC_MAX_PAGES:
            return self._analyze_pdf_page_by_page(content, file_path)

        return self._analyze_sync({"Bytes": content})

    def _analyze_async_with_local_fallback(
        self, content: bytes, file_path: str, page_count: int, oversized: bool
    ) -> tuple[list[dict], dict[str, Any]]:
        """Analyze via a staged S3 object, splitting the PDF locally if staging is not possible."""
        try:
            return self._analyze_via_staged_s3_object(content, file_path)
        except TextractError:
            raise
        except Exception as error:  # noqa: BLE001
            if self.processing_mode == TextractProcessingMode.ASYNC or oversized:
                raise TextractError(
                    f"'{file_path}' needs the asynchronous Textract API ({page_count} pages, "
                    f"{len(content)} bytes), which reads the document from S3, but staging it failed: "
                    f"{error}. Ensure the AWS credentials can use Dynamiq's internal staging bucket, "
                    "or pass the document via `s3_objects`."
                ) from error
            logger.warning(
                f"Could not stage '{file_path}' in S3 ({error}). Falling back to splitting the "
                f"{page_count}-page PDF locally into single-page synchronous requests."
            )
            return self._analyze_pdf_page_by_page(content, file_path)

    def _resolve_staging_bucket(self, s3_client) -> str:
        """Return the staging bucket to upload to, provisioning it on first use.

        Textract's asynchronous APIs only read from S3, so a bucket has to exist. Rather than making
        every caller provision one, the default name is derived from the caller's own account and
        region: stable across runs, private to the account, and co-located with the API call, which
        Textract requires.
        """
        if self._resolved_staging_bucket:
            return self._resolved_staging_bucket

        bucket = self._internal_staging_bucket_name(s3_client)
        self._provision_staging_bucket(s3_client, bucket)

        self._resolved_staging_bucket = bucket
        return bucket

    def _internal_staging_bucket_name(self, s3_client) -> str:
        """Derive the per-account, per-region staging bucket name from the caller's identity."""
        if self.connection is None:
            raise TextractError("Deriving a staging bucket name requires a `connection` with AWS credentials.")
        account_id = self.connection.get_client("sts").get_caller_identity()["Account"]
        return f"{_STAGING_BUCKET_PREFIX}-{account_id}-{s3_client.meta.region_name}"

    def _provision_staging_bucket(self, s3_client, bucket: str) -> None:
        """Create the staging bucket when it is missing, then lock it down."""
        try:
            s3_client.head_bucket(Bucket=bucket)
            return
        except Exception as error:  # noqa: BLE001
            logger.debug(f"Staging bucket '{bucket}' is not reachable ({error}); creating it.")

        region = s3_client.meta.region_name
        params: dict[str, Any] = {"Bucket": bucket}
        # us-east-1 is the API's implicit default and rejects an explicit LocationConstraint.
        if region and region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": region}

        try:
            s3_client.create_bucket(**params)
            logger.info(f"Created Textract staging bucket '{bucket}' in {region}.")
        except Exception as error:  # noqa: BLE001
            # A concurrent run may have won the race, leaving the bucket usable as it stands.
            if type(error).__name__ not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                logger.debug(f"Could not create staging bucket '{bucket}': {error}")
                return

        self._harden_staging_bucket(s3_client, bucket)

    def _harden_staging_bucket(self, s3_client, bucket: str) -> None:
        """Block public access and expire the staging prefix on a newly created bucket.

        Staged documents are deleted as soon as their job finishes, but a killed process cannot run
        its cleanup, so the lifecycle rule is the backstop against anything lingering. Both calls are
        best-effort: the bucket works without them, and the credentials may lack PutBucket*.
        """
        try:
            s3_client.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Could not block public access on staging bucket '{bucket}': {error}")

        try:
            s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "dynamiq-textract-staging-expiration",
                            "Status": "Enabled",
                            "Filter": {"Prefix": f"{_STAGING_S3_PREFIX}/"},
                            "Expiration": {"Days": _STAGED_OBJECT_EXPIRATION_DAYS},
                            "NoncurrentVersionExpiration": {
                                "NoncurrentDays": _STAGED_OBJECT_EXPIRATION_DAYS,
                            },
                            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                        }
                    ]
                },
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Could not set the expiration lifecycle on staging bucket '{bucket}': {error}")

    def _analyze_via_staged_s3_object(self, content: bytes, file_path: str) -> tuple[list[dict], dict[str, Any]]:
        """Upload the document to the staging bucket, analyze it asynchronously, then clean up."""
        s3_client = self._get_s3_client()
        bucket = self._resolve_staging_bucket(s3_client)
        key = f"{_STAGING_S3_PREFIX}/{generate_uuid()}-{Path(file_path).name}"
        upload = s3_client.put_object(Bucket=bucket, Key=key, Body=content)
        version = upload.get("VersionId") if isinstance(upload, dict) else None
        staged_object = TextractS3Object(bucket=bucket, name=key, version=version)
        logger.debug(f"Staged '{file_path}' at {staged_object.uri} for asynchronous Textract analysis.")

        try:
            return self._analyze_async(staged_object)
        finally:
            self._delete_staged_object(s3_client, staged_object)

    def _delete_staged_object(self, s3_client, staged_object: TextractS3Object) -> None:
        """Remove a staged document once its job has finished, successfully or not."""
        try:
            params = {"Bucket": staged_object.bucket, "Key": staged_object.name}
            if staged_object.version:
                params["VersionId"] = staged_object.version
            s3_client.delete_object(**params)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Failed to delete staged object {staged_object.uri}: {error}")

    @staticmethod
    def _validate_async_file_size(content: bytes, file_path: str, extension: str) -> None:
        """Reject files that exceed Textract's format-specific asynchronous limits before upload."""
        max_bytes = ASYNC_MAX_PDF_TIFF_BYTES if extension in {".pdf", ".tif", ".tiff"} else ASYNC_MAX_IMAGE_BYTES
        if len(content) > max_bytes:
            raise ValueError(
                f"'{file_path}' is {len(content)} bytes, above the {max_bytes}-byte asynchronous "
                f"Textract limit for {extension or 'this file type'}."
            )

    def _analyze_pdf_page_by_page(self, content: bytes, file_path: str) -> tuple[list[dict], dict[str, Any]]:
        """Split a multi-page PDF locally and analyze each page with the synchronous API.

        Page numbers are rewritten to the original document's numbering, since each single-page
        request reports itself as page 1.
        """
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(BytesIO(content))
        all_blocks: list[dict] = []
        model_versions: list[str] = []

        for page_index, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            buffer = BytesIO()
            writer.write(buffer)

            page_blocks, page_metadata = self._analyze_sync({"Bytes": buffer.getvalue()})
            for block in page_blocks:
                block["Page"] = page_index
            all_blocks.extend(page_blocks)
            if page_metadata.get("textract_model_version"):
                model_versions.append(page_metadata["textract_model_version"])

        logger.debug(f"Analyzed '{file_path}' as {len(reader.pages)} single-page synchronous Textract requests.")

        return all_blocks, {
            "textract_pages": len(reader.pages),
            "textract_processing_mode": "sync-per-page",
            "textract_model_version": model_versions[0] if model_versions else None,
        }

    def _feature_params(self) -> dict[str, Any]:
        """Build the FeatureTypes/QueriesConfig/AdaptersConfig shared by both analysis APIs."""
        params: dict[str, Any] = {"FeatureTypes": [feature.to_api() for feature in self.feature_types]}
        if self.queries:
            params["QueriesConfig"] = {"Queries": [query.to_api() for query in self.queries]}
        if self.adapters:
            params["AdaptersConfig"] = {"Adapters": [adapter.to_api() for adapter in self.adapters]}
        return params

    def _analyze_sync(self, document: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
        """Call ``AnalyzeDocument`` (or ``DetectDocumentText`` when no features are requested)."""
        client = self._get_client()

        try:
            if self.feature_types:
                response = client.analyze_document(Document=document, **self._feature_params())
                model_version = response.get("AnalyzeDocumentModelVersion")
            else:
                response = client.detect_document_text(Document=document)
                model_version = response.get("DetectDocumentTextModelVersion")
        except Exception as error:
            raise TextractError(f"Textract synchronous analysis failed: {error}") from error

        return response.get("Blocks", []), {
            "textract_pages": response.get("DocumentMetadata", {}).get("Pages", 1),
            "textract_processing_mode": TextractProcessingMode.SYNC.value,
            "textract_model_version": model_version,
        }

    def _analyze_async(self, s3_object: TextractS3Object) -> tuple[list[dict], dict[str, Any]]:
        """Start an asynchronous Textract job, poll it to completion and collect all blocks."""
        client = self._get_client()

        request: dict[str, Any] = {"DocumentLocation": {"S3Object": s3_object.to_api()}}
        if self.notification_channel:
            request["NotificationChannel"] = self.notification_channel.to_api()
        if self.kms_key_id:
            request["KMSKeyId"] = self.kms_key_id
        if self.job_tag:
            request["JobTag"] = self.job_tag

        try:
            if self.feature_types:
                request.update(self._feature_params())
                job_id = client.start_document_analysis(**request)["JobId"]
                fetch = client.get_document_analysis
                model_version_key = "AnalyzeDocumentModelVersion"
            else:
                job_id = client.start_document_text_detection(**request)["JobId"]
                fetch = client.get_document_text_detection
                model_version_key = "DetectDocumentTextModelVersion"
        except Exception as error:
            raise TextractError(f"Failed to start Textract job for {s3_object.uri}: {error}") from error

        logger.debug(f"Started Textract job {job_id} for {s3_object.uri}.")

        blocks, response_metadata = self._collect_job_results(fetch, job_id, model_version_key)
        response_metadata["textract_job_id"] = job_id
        response_metadata["textract_processing_mode"] = TextractProcessingMode.ASYNC.value

        return blocks, response_metadata

    def _collect_job_results(self, fetch, job_id: str, model_version_key: str) -> tuple[list[dict], dict[str, Any]]:
        """Poll a Textract job until it terminates, then page through every block it produced."""
        deadline = time.monotonic() + self.max_poll_seconds

        while True:
            try:
                response = fetch(JobId=job_id)
            except Exception as error:
                raise TextractError(f"Failed to poll Textract job {job_id}: {error}") from error

            status = response.get("JobStatus")
            if status in ("SUCCEEDED", "PARTIAL_SUCCESS", "FAILED"):
                break

            if time.monotonic() >= deadline:
                raise TextractError(
                    f"Textract job {job_id} did not finish within {self.max_poll_seconds} seconds "
                    f"(last status: {status})."
                )
            time.sleep(self.poll_interval_seconds)

        if status == "FAILED":
            raise TextractError(
                f"Textract job {job_id} failed: {response.get('StatusMessage') or 'no status message returned'}"
            )
        if status == "PARTIAL_SUCCESS":
            logger.warning(
                f"Textract job {job_id} completed with PARTIAL_SUCCESS. "
                f"Warnings: {response.get('Warnings')}. Message: {response.get('StatusMessage')}"
            )

        blocks = list(response.get("Blocks", []))
        next_token = response.get("NextToken")
        while next_token:
            try:
                response = fetch(JobId=job_id, NextToken=next_token)
            except Exception as error:
                raise TextractError(f"Failed to page through Textract job {job_id} results: {error}") from error
            blocks.extend(response.get("Blocks", []))
            next_token = response.get("NextToken")

        return blocks, {
            "textract_pages": response.get("DocumentMetadata", {}).get("Pages", 1),
            "textract_model_version": response.get(model_version_key),
            "textract_job_status": status,
        }

    @staticmethod
    def _count_pdf_pages(content: bytes) -> int:
        """Count PDF pages locally to pick between the synchronous and asynchronous API."""
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            return len(PdfReader(BytesIO(content)).pages)
        except (PdfReadError, ValueError, OSError) as error:
            logger.warning(f"Could not count PDF pages locally ({error}); assuming a multi-page document.")
            return SYNC_MAX_PAGES + 1

    def _create_documents(
        self,
        filepath: str,
        elements: list[dict],
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        response_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Document]:
        """Render Textract blocks into Documents, one per file or one per page."""
        response_metadata = response_metadata or {}
        blocks_by_id = {block["Id"]: block for block in elements if block.get("Id")}
        pages = self._group_blocks_by_page(elements)

        base_metadata = build_source_metadata(metadata, filepath)
        base_metadata["textract_feature_types"] = [feature.value for feature in self.feature_types]
        for key, value in response_metadata.items():
            if value is not None:
                base_metadata[key] = value

        rendered_pages = [
            (page_number, self._render_page(page_blocks, blocks_by_id, page_number))
            for page_number, page_blocks in sorted(pages.items())
        ]

        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_PAGE:
            documents = []
            for page_number, page in rendered_pages:
                page_metadata = copy.deepcopy(base_metadata)
                page_metadata["page_number"] = page_number
                self._attach_structured_metadata(page_metadata, [page])
                documents.append(Document(content=page["content"], metadata=page_metadata))
            return documents

        content = self.separator.join(page["content"] for _, page in rendered_pages if page["content"].strip())
        document_metadata = copy.deepcopy(base_metadata)
        self._attach_structured_metadata(document_metadata, [page for _, page in rendered_pages])

        return [Document(content=content, metadata=document_metadata)]

    @staticmethod
    def _group_blocks_by_page(blocks: list[dict]) -> dict[int, list[dict]]:
        """Group blocks by page, defaulting to page 1 for single-page responses.

        Textract omits ``Page`` on synchronous single-page responses, so the default keeps
        one-doc-per-page mode working for images and single-page PDFs.
        """
        pages: dict[int, list[dict]] = {}
        for block in blocks:
            pages.setdefault(int(block.get("Page", 1)), []).append(block)
        return pages

    @staticmethod
    def _attach_structured_metadata(metadata: dict[str, Any], pages: list[dict[str, Any]]) -> None:
        """Merge per-page tables/forms/queries/signatures into the document metadata."""
        for key in ("tables", "forms", "queries", "signatures"):
            collected = [entry for page in pages for entry in page[key]]
            if collected:
                metadata[key] = collected

    def _render_page(self, page_blocks: list[dict], blocks_by_id: dict[str, dict], page_number: int) -> dict[str, Any]:
        """Render one page to Markdown plus the structured data extracted from it."""
        tables = self._extract_tables(page_blocks, blocks_by_id, page_number)
        forms = self._extract_forms(page_blocks, blocks_by_id, page_number)
        queries = self._extract_queries(page_blocks, blocks_by_id, page_number)
        signatures = self._extract_signatures(page_blocks, page_number)

        layout_blocks = [block for block in page_blocks if str(block.get("BlockType", "")).startswith("LAYOUT_")]
        if layout_blocks:
            body = self._render_layout(layout_blocks, blocks_by_id, tables)
        else:
            body = self._render_lines(page_blocks, blocks_by_id)
            if self.render_tables_as_markdown:
                body.extend(table["markdown"] for table in tables)

        if self.include_form_section and forms:
            body.append(self._render_form_section(forms))
        if self.include_query_section and queries:
            body.append(self._render_query_section(queries))

        return {
            "content": self.separator.join(section for section in body if section.strip()),
            "tables": tables,
            "forms": forms,
            "queries": queries,
            "signatures": signatures,
        }

    def _render_lines(self, page_blocks: list[dict], blocks_by_id: dict[str, dict]) -> list[str]:
        """Render the page as plain LINE blocks, in the reading order Textract returned."""
        lines = [
            self._block_text(block, blocks_by_id)
            for block in page_blocks
            if block.get("BlockType") in _TEXT_BLOCK_TYPES and self._meets_confidence(block)
        ]
        text = "\n".join(line for line in lines if line.strip())
        return [text] if text.strip() else []

    def _render_layout(
        self, layout_blocks: list[dict], blocks_by_id: dict[str, dict], tables: list[dict[str, Any]]
    ) -> list[str]:
        """Render the page from Textract LAYOUT regions, which carry semantics and reading order."""
        skipped_types = (
            {"LAYOUT_HEADER", "LAYOUT_FOOTER", "LAYOUT_PAGE_NUMBER"} if self.skip_headers_and_footers else set()
        )
        sections: list[str] = []
        consumed_table_indexes: set[int] = set()
        rendered_boxes: list[dict[str, float]] = []

        for block in layout_blocks:
            block_type = block.get("BlockType")
            if block_type in skipped_types:
                continue

            if block_type == "LAYOUT_TABLE":
                # Children are LINE blocks, not TABLE blocks, so match by bounding-box overlap.
                table = self._match_table_by_geometry(block, tables, consumed_table_indexes)
                if table is not None and self.render_tables_as_markdown:
                    consumed_table_indexes.add(table["index"])
                    sections.append(table["markdown"])
                    continue

            if block_type == "LAYOUT_FIGURE":
                caption = self._block_text(block, blocks_by_id).strip()
                sections.append(f"<!-- figure: {caption} -->" if caption else "<!-- figure -->")
                continue

            text = self._block_text(block, blocks_by_id).strip()
            if not text:
                continue

            box = block.get("Geometry", {}).get("BoundingBox")
            if box:
                rendered_boxes.append(box)

            if block_type == "LAYOUT_TITLE":
                sections.append(f"# {text}")
            elif block_type == "LAYOUT_SECTION_HEADER":
                sections.append(f"## {text}")
            elif block_type == "LAYOUT_LIST":
                sections.append("\n".join(f"- {item}" for item in text.splitlines() if item.strip()))
            else:
                sections.append(text)

        if self.render_tables_as_markdown:
            # A table the covering region already rendered as text would otherwise be duplicated.
            sections.extend(
                table["markdown"]
                for table in tables
                if table["index"] not in consumed_table_indexes
                and not self._is_covered_by(table.get("bounding_box"), rendered_boxes)
            )

        return sections

    @staticmethod
    def _is_covered_by(box: dict[str, float] | None, other_boxes: list[dict[str, float]]) -> bool:
        """Whether most of `box` falls inside any of `other_boxes`."""
        if not box:
            return False

        area = box["Width"] * box["Height"]
        if area <= 0:
            return False

        for other in other_boxes:
            left = max(box["Left"], other["Left"])
            top = max(box["Top"], other["Top"])
            right = min(box["Left"] + box["Width"], other["Left"] + other["Width"])
            bottom = min(box["Top"] + box["Height"], other["Top"] + other["Height"])
            if right > left and bottom > top and (right - left) * (bottom - top) / area >= 0.7:
                return True

        return False

    @staticmethod
    def _match_table_by_geometry(
        layout_block: dict, tables: list[dict[str, Any]], consumed_indexes: set[int]
    ) -> dict[str, Any] | None:
        """Find the TABLE block that occupies the same region as a LAYOUT_TABLE block."""
        layout_box = layout_block.get("Geometry", {}).get("BoundingBox")
        if not layout_box:
            return None

        best_table, best_overlap = None, 0.0
        for table in tables:
            if table["index"] in consumed_indexes or not table.get("bounding_box"):
                continue
            overlap = TextractFileConverter._intersection_over_union(layout_box, table["bounding_box"])
            if overlap > best_overlap:
                best_table, best_overlap = table, overlap

        return best_table if best_overlap >= 0.3 else None

    @staticmethod
    def _intersection_over_union(first: dict[str, float], second: dict[str, float]) -> float:
        """Intersection-over-union of two Textract bounding boxes (normalized page coordinates)."""
        left = max(first["Left"], second["Left"])
        top = max(first["Top"], second["Top"])
        right = min(first["Left"] + first["Width"], second["Left"] + second["Width"])
        bottom = min(first["Top"] + first["Height"], second["Top"] + second["Height"])

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        union = first["Width"] * first["Height"] + second["Width"] * second["Height"] - intersection

        return intersection / union if union > 0 else 0.0

    def _extract_tables(
        self, page_blocks: list[dict], blocks_by_id: dict[str, dict], page_number: int
    ) -> list[dict[str, Any]]:
        """Extract every TABLE block on the page as both a grid and a Markdown table."""
        tables: list[dict[str, Any]] = []

        for index, block in enumerate(
            [candidate for candidate in page_blocks if candidate.get("BlockType") == "TABLE"]
        ):
            rows = self._build_table_grid(block, blocks_by_id)
            if not rows:
                continue
            tables.append(
                {
                    "index": index,
                    "page": page_number,
                    "rows": rows,
                    "markdown": self._to_markdown_table(rows),
                    "confidence": block.get("Confidence"),
                    "bounding_box": block.get("Geometry", {}).get("BoundingBox"),
                    "title": self._related_text(block, blocks_by_id, "TABLE_TITLE"),
                    "footer": self._related_text(block, blocks_by_id, "TABLE_FOOTER"),
                }
            )

        return tables

    def _build_table_grid(self, table_block: dict, blocks_by_id: dict[str, dict]) -> list[list[str]]:
        """Turn a TABLE block's CELL children into a dense row/column grid.

        Merged cells are reported both as MERGED_CELL blocks and as the underlying CELLs; the
        merged text is written into the span's top-left position and the covered positions are
        left empty so column alignment survives.
        """
        cells = [
            blocks_by_id[cell_id]
            for cell_id in self._related_ids(table_block, "CHILD")
            if cell_id in blocks_by_id and blocks_by_id[cell_id].get("BlockType") == "CELL"
        ]
        if not cells:
            return []

        row_count = max(int(cell.get("RowIndex", 1)) + int(cell.get("RowSpan", 1)) - 1 for cell in cells)
        column_count = max(int(cell.get("ColumnIndex", 1)) + int(cell.get("ColumnSpan", 1)) - 1 for cell in cells)
        grid = [["" for _ in range(column_count)] for _ in range(row_count)]

        for cell in cells:
            row = int(cell.get("RowIndex", 1)) - 1
            column = int(cell.get("ColumnIndex", 1)) - 1
            if 0 <= row < row_count and 0 <= column < column_count:
                grid[row][column] = self._block_text(cell, blocks_by_id).strip()

        for merged_cell in [
            blocks_by_id[cell_id]
            for cell_id in self._related_ids(table_block, "MERGED_CELL")
            if cell_id in blocks_by_id
        ]:
            row = int(merged_cell.get("RowIndex", 1)) - 1
            column = int(merged_cell.get("ColumnIndex", 1)) - 1
            row_span = int(merged_cell.get("RowSpan", 1))
            column_span = int(merged_cell.get("ColumnSpan", 1))
            text = " ".join(
                self._block_text(blocks_by_id[cell_id], blocks_by_id).strip()
                for cell_id in self._related_ids(merged_cell, "CHILD")
                if cell_id in blocks_by_id
            ).strip()

            for row_offset in range(row_span):
                for column_offset in range(column_span):
                    target_row, target_column = row + row_offset, column + column_offset
                    if 0 <= target_row < row_count and 0 <= target_column < column_count:
                        grid[target_row][target_column] = text if row_offset == 0 and column_offset == 0 else ""

        return grid

    def _extract_forms(
        self, page_blocks: list[dict], blocks_by_id: dict[str, dict], page_number: int
    ) -> list[dict[str, Any]]:
        """Extract FORMS key-value pairs from KEY_VALUE_SET blocks on the page."""
        forms: list[dict[str, Any]] = []

        for block in page_blocks:
            if block.get("BlockType") != "KEY_VALUE_SET" or "KEY" not in block.get("EntityTypes", []):
                continue
            if not self._meets_confidence(block):
                continue

            key = self._block_text(block, blocks_by_id).strip()
            value_blocks = [
                blocks_by_id[value_id] for value_id in self._related_ids(block, "VALUE") if value_id in blocks_by_id
            ]
            value = " ".join(self._block_text(value, blocks_by_id).strip() for value in value_blocks).strip()

            if not key and not value:
                continue

            forms.append(
                {
                    "key": key,
                    "value": value,
                    "page": page_number,
                    "confidence": block.get("Confidence"),
                }
            )

        return forms

    def _extract_queries(
        self, page_blocks: list[dict], blocks_by_id: dict[str, dict], page_number: int
    ) -> list[dict[str, Any]]:
        """Extract QUERIES answers, pairing each QUERY block with its QUERY_RESULT blocks."""
        queries: list[dict[str, Any]] = []

        for block in page_blocks:
            if block.get("BlockType") != "QUERY":
                continue

            query = block.get("Query", {})
            answers = [
                blocks_by_id[answer_id] for answer_id in self._related_ids(block, "ANSWER") if answer_id in blocks_by_id
            ]
            best_answer = max(answers, key=lambda answer: answer.get("Confidence", 0), default=None)

            queries.append(
                {
                    "query": query.get("Text", ""),
                    "alias": query.get("Alias") or query.get("Text", ""),
                    "answer": (best_answer or {}).get("Text", ""),
                    "confidence": (best_answer or {}).get("Confidence"),
                    "page": page_number,
                }
            )

        return queries

    @staticmethod
    def _extract_signatures(page_blocks: list[dict], page_number: int) -> list[dict[str, Any]]:
        """Extract SIGNATURES detections, keeping confidence and position for downstream review."""
        return [
            {
                "page": page_number,
                "confidence": block.get("Confidence"),
                "bounding_box": block.get("Geometry", {}).get("BoundingBox"),
            }
            for block in page_blocks
            if block.get("BlockType") == "SIGNATURE"
        ]

    @staticmethod
    def _render_form_section(forms: list[dict[str, Any]]) -> str:
        """Render extracted key-value pairs as a Markdown definition list."""
        lines = ["## Form fields"]
        lines.extend(f"- **{form['key'] or '(unlabeled)'}**: {form['value']}" for form in forms)
        return "\n".join(lines)

    @staticmethod
    def _render_query_section(queries: list[dict[str, Any]]) -> str:
        """Render query answers as a Markdown definition list."""
        lines = ["## Query answers"]
        lines.extend(f"- **{query['alias']}**: {query['answer']}" for query in queries)
        return "\n".join(lines)

    @staticmethod
    def _to_markdown_table(rows: list[list[str]]) -> str:
        """Render a grid as a Markdown table, treating the first row as the header."""
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return ""

        width = max(len(row) for row in rows)

        def render_row(row: list[str]) -> str:
            padded = row + [""] * (width - len(row))
            cells = [cell.replace("|", "\\|").replace("\n", " ") for cell in padded]
            return "| " + " | ".join(cells) + " |"

        lines = [render_row(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
        lines.extend(render_row(row) for row in rows[1:])
        return "\n".join(lines)

    @staticmethod
    def _related_ids(block: dict, relationship_type: str) -> list[str]:
        """Return the ids a block points to through the given relationship type."""
        return [
            related_id
            for relationship in block.get("Relationships") or []
            if relationship.get("Type") == relationship_type
            for related_id in relationship.get("Ids", [])
        ]

    def _related_text(self, block: dict, blocks_by_id: dict[str, dict], relationship_type: str) -> str:
        """Return the concatenated text of blocks reached through a relationship type."""
        return " ".join(
            self._block_text(blocks_by_id[related_id], blocks_by_id).strip()
            for related_id in self._related_ids(block, relationship_type)
            if related_id in blocks_by_id
        ).strip()

    def _block_text(self, block: dict, blocks_by_id: dict[str, dict]) -> str:
        """Resolve a block's text.

        LINE and QUERY_RESULT blocks carry ``Text`` directly. Container blocks (cells, key-value
        sets, layout regions) carry it only through their CHILD relationships, where WORD blocks
        hold words and SELECTION_ELEMENT blocks hold checkbox state.
        """
        if block.get("BlockType") in ("WORD", "LINE", "QUERY_RESULT"):
            return block.get("Text", "")

        if block.get("BlockType") == "SELECTION_ELEMENT":
            return "[X]" if block.get("SelectionStatus") == "SELECTED" else "[ ]"

        parts: list[str] = []
        for child_id in self._related_ids(block, "CHILD"):
            child = blocks_by_id.get(child_id)
            if child is None or not self._meets_confidence(child):
                continue

            child_type = child.get("BlockType")
            if child_type == "LINE":
                # Keep the line breaks between LINEs so list regions stay splittable.
                parts.append(("\n" if parts else "") + child.get("Text", ""))
            elif child_type in ("WORD", "SELECTION_ELEMENT"):
                parts.append((" " if parts else "") + self._block_text(child, blocks_by_id))
            else:
                nested = self._block_text(child, blocks_by_id)
                if nested:
                    parts.append((" " if parts else "") + nested)

        return "".join(parts).strip()

    def _meets_confidence(self, block: dict) -> bool:
        """Whether a block clears the configured confidence floor."""
        if self.min_confidence is None:
            return True
        confidence = block.get("Confidence")
        return confidence is None or confidence >= self.min_confidence
