import base64
import concurrent.futures
import copy
import enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.components.converters.utils import get_filename_for_bytesio

if TYPE_CHECKING:
    from PIL import Image

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.prompts import (
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils.logger import logger

DEFAULT_EXTRACTION_INSTRUCTION = """
Please extract the text from the provided images and present it in Markdown format.
Maintain the Markdown syntax for all formatting elements such as headings, images, links,
bold text, tables, etc. Do not enclose the text with ```markdown...```. Do not extract the
header and footer. Do not write your own text. Only extract the text from the images.
"""


class DocumentCreationMode(str, enum.Enum):
    ONE_DOC_PER_FILE = "one-doc-per-file"
    ONE_DOC_PER_PAGE = "one-doc-per-page"


def create_vision_prompt_template() -> Prompt:
    """
    Creates a vision prompt template.

    Returns:
        Prompt: The vision prompt template.
    """
    text_message = VisionMessageTextContent(text="{{extraction_instruction}}")
    image_message = VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{img_url}}"))
    vision_message = VisionMessage(content=[text_message, image_message], role="user")
    vision_prompt = Prompt(messages=[vision_message])
    return vision_prompt


class LLMImageConverterInputSchema(BaseModel):
    file_paths: list[str] = Field(default=None, description="Parameter to provide paths to files.")
    files: list[BytesIO | bytes] = Field(default=None, description="Parameter to provide files.")
    metadata: dict | list = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that either `file_paths` or `files` is specified"""
        if not self.file_paths and not self.files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class LLMImageConverter(Node):
    """
    A Node class for extracting text from images using a Large Language Model (LLM).

    This class extracts text from the images using an LLM and saves the text as documents with metadata.

    Attributes:
        group (Literal[NodeGroup.CONVERTERS]): The group the node belongs to. Default is NodeGroup.CONVERTERS.
        name (str): The name of the node. Default is "LLMImageConverter".
        extraction_instruction (str): The instruction for text extraction.
            Default is DEFAULT_EXTRACTION_INSTRUCTION.
        document_creation_mode (DocumentCreationMode): The mode for document creation.
            Default is DocumentCreationMode.ONE_DOC_PER_FILE.
        llm (BaseLLM): The LLM instance used for text extraction. Default is None.

    Example:

        from dynamiq.nodes.extractors import ImageLLMExtractor
        from io import BytesIO

        # Initialize the extractor
        extractor = ImageLLMExtractor(llm=my_llm_instance)

        # Example input data
        input_data = {
            "file_paths": ["path/to/image1.jpeg", "path/to/image2.png"],
            "files": [BytesIO(b"image1 content"), BytesIO(b"image2 content")],
            "metadata": {"source": "example source"}
        }

        # Execute the extractor
        output = extractor.execute(input_data)

        # Output will be a dictionary with extracted documents
        print(output)
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "LLMImageConverter"
    extraction_instruction: str = DEFAULT_EXTRACTION_INSTRUCTION
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE
    llm: Node
    vision_prompt: Prompt = Field(default_factory=create_vision_prompt_template)
    input_schema: ClassVar[type[LLMImageConverterInputSchema]] = LLMImageConverterInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the LLMImageConverter with the given parameters and creates a default LLM node.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the document extractor component.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def execute(
        self, input_data: LLMImageConverterInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the image text extraction process.

        Args:
            input_data (LLMImageConverterInputSchema): An instance containing the images to be processed.
            config (RunnableConfig, optional): Configuration for the execution. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the extracted documents.

        Example:

            input_data = {
                "file_paths": ["path/to/image1.jpeg", "path/to/image2.png"],
                "files": [BytesIO(b"image1 content"), BytesIO(b"image2 content")],
                "metadata": {"source": "example source"}
            }

            output = extractor.execute(input_data)

            # output will be a dictionary with extracted documents
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = self.extract_text_from_images(
            file_paths=input_data.file_paths,
            files=input_data.files,
            metadata=input_data.metadata,
            config=config,
            **kwargs,
        )

        return {"documents": documents}

    def extract_text_from_images(
        self,
        file_paths: list[str] | None = None,
        files: list[BytesIO] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        config: RunnableConfig = None,
        **kwargs,
    ) -> list[Document]:
        """
        Extracts text from images using an LLM.

        Args:
            file_paths (list[str], optional): List of paths to image files. Default is None.
            files (list[BytesIO], optional): List of image files as BytesIO objects. Default is None.
            metadata (dict[str, Any] | list[dict[str, Any]], optional): Metadata for the documents. Default is None.
            config (RunnableConfig, optional): Configuration for the execution. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of extracted documents.
        """

        documents = []

        if file_paths is not None:
            paths_obj = [Path(path) for path in file_paths]
            filepaths = [path for path in paths_obj if path.is_file()]
            filepaths_in_directories = [
                filepath
                for path in paths_obj
                if path.is_dir()
                for filepath in path.glob("*.*")
                if filepath.is_file()
            ]
            if filepaths_in_directories and isinstance(metadata, list):
                raise ValueError(
                    "If providing directories in the `file_paths` parameter, "
                    "`metadata` can only be a dictionary (metadata applied to every file), "
                    "and not a list. To specify different metadata for each file, "
                    "provide an explicit list of direct paths instead."
                )

            all_filepaths = filepaths + filepaths_in_directories
            meta_list = self._normalize_metadata(metadata, len(all_filepaths))

            for file_path, meta in zip(all_filepaths, meta_list):
                with open(file_path, "rb") as upload_file:
                    file = BytesIO(upload_file.read())
                    file.name = upload_file.name

                image = self._load_image(file)
                meta["filename"] = str(file_path)
                documents.extend(self._process_images([image], meta, config, **kwargs))

        if files is not None:
            meta_list = self._normalize_metadata(metadata, len(files))

            for file, meta in zip(files, meta_list):
                if not isinstance(file, BytesIO):
                    raise ValueError("All files must be of type BytesIO.")
                image = self._load_image(file)
                meta["filename"] = get_filename_for_bytesio(file)
                documents.extend(self._process_images([image], meta, config, **kwargs))

        return documents

    def _load_image(self, file: BytesIO) -> "Image":
        """
        Loads an image from a BytesIO object.

        Args:
            file (BytesIO): The BytesIO object containing the image data.

        Returns:
            Image: The loaded image.
        """
        from PIL import Image

        return Image.open(file)

    def _process_images(
        self,
        images: list["Image"],
        metadata: dict[str, Any],
        config: RunnableConfig,
        **kwargs,
    ) -> list[Document]:
        """
        Extracts text from images using a vision prompt.

        Args:
            images (list[Image]): List of images.
            metadata (dict[str, Any]): Metadata for the documents.
            config (RunnableConfig): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of extracted documents.
        """
        urls = self._convert_images_to_urls(images)

        outputs = self.perform_llm_extraction(urls, config, **kwargs)

        if self.document_creation_mode == DocumentCreationMode.ONE_DOC_PER_FILE:
            document_content = "".join(output["content"] for output in outputs)
            return [Document(content=document_content, metadata=metadata)]
        else:
            documents = [
                Document(content=output["content"], metadata=metadata)
                for output in outputs
            ]
            return documents

    def perform_llm_extraction(
        self, urls: list[str], config: RunnableConfig, **kwargs
    ) -> list[dict]:
        """
        Performs the actual extraction of text from images using the LLM.

        Args:
            urls (list[str]): The list of image URLs to extract text from.
            config (RunnableConfig): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            list[dict]: A list of extracted text results from the LLM.

        Example:

            urls = ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."]

            extracted_texts = extractor.perform_llm_extraction(urls, config)

            # extracted_texts will be a list of dictionaries with extracted text
        """
        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        inputs = [
            {"extraction_instruction": self.extraction_instruction, "img_url": url}
            for url in urls
        ]

        prompt = self.vision_prompt

        with concurrent.futures.ThreadPoolExecutor() as executor:
            llm_results = list(
                executor.map(
                    lambda input_data: self.call_llm(
                        input_data, prompt, config, **run_kwargs
                    ),
                    inputs,
                )
            )

        logger.debug(
            f"Node {self.name} - {self.id}: LLM processed {len(llm_results)} images"
        )

        return llm_results

    def call_llm(self, input_data, prompt, config, **run_kwargs):
        """
        Calls the LLM with the given input data and prompt.

        Args:
            input_data (dict): The input data for the LLM.
            prompt (Prompt): The prompt to be used with the LLM.
            config (RunnableConfig): Configuration for the execution.
            **run_kwargs: Additional keyword arguments.

        Returns:
            dict: The result from the LLM.
        """
        llm_result = self.llm.run(
            input_data=input_data,
            prompt=prompt,
            config=config,
            run_depends=self._run_depends,
            **run_kwargs,
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]

        if llm_result.status != RunnableStatus.SUCCESS:
            logger.error(f"Node {self.name} - {self.id}: LLM execution failed")
            raise ValueError("ImageLLMExtractor LLM execution failed")
        return llm_result.output

    @staticmethod
    def _convert_image_to_url(image: "Image") -> str:
        """
        Converts a PIL Image to a base64-encoded URL.

        Args:
            image (Image): The image to convert.

        Returns:
            str: The base64-encoded URL of the image.
        """
        # Ensure the image is in RGB mode (required for JPEG)
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)  # Ensure the buffer is at the beginning
        decoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = f"data:image/jpeg;base64,{decoded_image}"
        return url

    @staticmethod
    def _convert_images_to_urls(images: list["Image"]) -> list[str]:
        """
        Converts a list of PIL Images to a list of base64-encoded URLs.

        Args:
            images (List[Image]): The list of images to convert.

        Returns:
            List[str]: The list of base64-encoded URLs.
        """
        return [LLMImageConverter._convert_image_to_url(image) for image in images]

    @staticmethod
    def _normalize_metadata(
        metadata: dict[str, Any] | list[dict[str, Any]] | None, sources_count: int
    ) -> list[dict[str, Any]]:
        """Normalizes metadata input for a converter.

        Given all possible values of the metadata input for a converter (None, dictionary, or list of
        dicts), ensures to return a list of dictionaries of the correct length for the converter to use.

        Args:
            metadata: The meta input of the converter, as-is. Can be None, a dictionary, or a list of
                dictionaries.
            sources_count: The number of sources the converter received.

        Returns:
            A list of dictionaries of the same length as the sources list.

        Raises:
            ValueError: If metadata is not None, a dictionary, or a list of dictionaries, or if the length
                of the metadata list doesn't match the number of sources.
        """
        if metadata is None:
            return [{} for _ in range(sources_count)]
        if isinstance(metadata, dict):
            return [copy.deepcopy(metadata) for _ in range(sources_count)]
        if isinstance(metadata, list):
            if sources_count != len(metadata):
                raise ValueError(
                    "The length of the metadata list must match the number of sources."
                )
            return metadata
        raise ValueError(
            "metadata must be either None, a dictionary or a list of dictionaries."
        )


class LLMPDFConverterInputSchema(BaseModel):
    file_paths: list[str] = Field(default=None, description="Parameter to provide path to files.")
    files: list[BytesIO | bytes] = Field(default=None, description="Parameter to provide files.")
    metadata: dict | list = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that either `file_paths` or `files` is specified"""
        if not self.file_paths and not self.files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class LLMPDFConverter(LLMImageConverter):
    """
    A Node class for extracting text from PDFs using a Large Language Model (LLM).

    This class converts PDFs to images, extracts text from the images using an LLM,
    and saves the text as documents with metadata.

    Attributes:
        group (Literal[NodeGroup.CONVERTERS]): The group the node belongs to. Default is NodeGroup.CONVERTERS.
        name (str): The name of the node. Default is "LLMPDFConverter".
        extraction_instruction (str): The instruction for text extraction.
            Default is DEFAULT_EXTRACTION_INSTRUCTION.
        document_creation_mode (DocumentCreationMode): The mode for document creation.
            Default is DocumentCreationMode.ONE_DOC_PER_FILE.
        llm (BaseLLM): The LLM instance used for text extraction. Default is None.

    Example:

        from dynamiq.nodes.converters import LLMPDFConverter
        from io import BytesIO

        # Initialize the extractor
        converter = LLMPDFConverter(llm=my_llm_instance)

        # Example input data
        input_data = {
            "file_paths": ["path/to/pdf1.pdf", "path/to/pdf2.pdf"],
            "files": [BytesIO(b"pdf1 content"), BytesIO(b"pdf2 content")],
            "metadata": {"source": "example source"}
        }

        # Execute the converter
        output = converter.execute(input_data)

        # Output will be a dictionary with extracted documents
        print(output)
    """

    _convert_from_bytes: Any = PrivateAttr()
    _convert_from_path: Any = PrivateAttr()
    input_schema: ClassVar[type[LLMPDFConverterInputSchema]] = LLMPDFConverterInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the LLMPDFConverter with the given parameters and creates a default LLM node.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        from pdf2image import convert_from_bytes, convert_from_path

        super().__init__(**kwargs)
        self._convert_from_bytes = convert_from_bytes
        self._convert_from_path = convert_from_path

    def execute(
        self, input_data: LLMPDFConverterInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the PDF text extraction process.

        Args:
            input_data (LLMPDFConverterInputSchema): An instance containing the file paths or
              files of PDFs to be processed.
            config (RunnableConfig, optional): Configuration for the execution. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the extracted documents.

        Example:

            input_data = {
                "file_paths": ["path/to/pdf1.pdf", "path/to/pdf2.pdf"],
                "metadata": {"source": "example source"}
            }

            output = extractor.execute(input_data)

            # output will be a dictionary with extracted documents
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = self.extract_text_from_pdfs(
            file_paths=input_data.file_paths,
            files=input_data.files,
            metadata=input_data.metadata,
            config=config,
            **kwargs,
        )

        return {"documents": documents}

    def extract_text_from_pdfs(
        self,
        file_paths: list[str] | None = None,
        files: list[BytesIO] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        config: RunnableConfig = None,
        **kwargs,
    ) -> list[Document]:
        """
        Extracts text from PDFs by converting them to images and using an LLM.

        Args:
            file_paths (list[str], optional): List of paths to PDF files. Default is None.
            files (list[BytesIO], optional): List of PDF files as BytesIO objects. Default is None.
            metadata (dict[str, Any] | list[dict[str, Any]], optional): Metadata for the documents. Default is None.
            config (RunnableConfig, optional): Configuration for the execution. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of extracted documents.
        """
        documents = []

        if file_paths is not None:
            paths_obj = [Path(path) for path in file_paths]
            filepaths = [path for path in paths_obj if path.is_file()]
            filepaths_in_directories = [
                filepath
                for path in paths_obj
                if path.is_dir()
                for filepath in path.glob("*.*")
                if filepath.is_file()
            ]
            if filepaths_in_directories and isinstance(metadata, list):
                raise ValueError(
                    "If providing directories in the `file_paths` parameter, "
                    "`metadata` can only be a dictionary (metadata applied to every file), "
                    "and not a list. To specify different metadata for each file, "
                    "provide an explicit list of direct paths instead."
                )

            all_filepaths = filepaths + filepaths_in_directories
            meta_list = self._normalize_metadata(metadata, len(all_filepaths))

            for file_path, meta in zip(all_filepaths, meta_list):
                images = self._convert_from_path(file_path)
                meta["filename"] = str(file_path)
                documents.extend(self._process_images(images, meta, config, **kwargs))

        if files is not None:
            meta_list = self._normalize_metadata(metadata, len(files))

            for file, meta in zip(files, meta_list):
                if not isinstance(file, BytesIO):
                    raise ValueError("All files must be of type BytesIO.")
                images = self._convert_from_bytes(file.read())
                meta["filename"] = get_filename_for_bytesio(file)
                documents.extend(self._process_images(images, meta, config, **kwargs))

        return documents
