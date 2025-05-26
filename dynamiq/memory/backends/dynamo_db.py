import json
import time
import uuid
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import AWS
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class BillingMode(str, Enum):
    PAY_PER_REQUEST = "PAY_PER_REQUEST"
    PROVISIONED = "PROVISIONED"


class DynamoDBMemoryError(Exception):
    """Base exception class for DynamoDB Memory Backend errors."""

    pass


def _convert_floats_to_decimals(obj: Any) -> Any:
    """
    Recursively walk through obj and convert any finite float to a Decimal.
    Raises ValueError on NaN or infinity, or if conversion fails.
    """
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            raise ValueError(f"Cannot convert non-finite float {obj} to Decimal for DynamoDB")
        try:
            return Decimal(str(obj))
        except InvalidOperation:
            raise ValueError(f"Could not convert float {obj} to Decimal")
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimals(elem) for elem in obj]
    else:
        return obj


class DynamoDB(MemoryBackend):
    """
    AWS DynamoDB implementation of memory storage using ONLY table scans.

    Relies exclusively on DynamoDB Scan operations. Scans read the
    entire table and can be slow and costly for large datasets.

    Assumed Table Schema:
    - PK: `message_id` (String)
    - SK: `timestamp` (Number - Unix float stored as Decimal)
    - Attributes: `role` (String), `content` (String), `metadata` (Map)
    """

    _MAX_SCAN_PAGE_LIMIT: int = 1000
    _DEFAULT_SCAN_SIZE: int = 5000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "DynamoDB"
    connection: AWS = Field(default_factory=AWS)
    table_name: str = Field("conversations", description="Name of the DynamoDB table.")
    create_if_not_exist: bool = Field(default=False)
    billing_mode: BillingMode = Field(
        default=BillingMode.PAY_PER_REQUEST,
        description="DynamoDB billing mode",
    )
    read_capacity_units: int = Field(default=1, gt=0)
    write_capacity_units: int = Field(default=1, gt=0)

    partition_key_name: str = Field(default="message_id")
    sort_key_name: str = Field(default="timestamp")
    role_attribute_name: str = Field(default="role")
    content_attribute_name: str = Field(default="content")
    metadata_attribute_name: str = Field(default="metadata")
    scan_fetch_target_multiplier: int = Field(
        default=10,
        gt=0,
        description=(
            "When a limit is provided, multiply the limit by this factor "
            "to determine the initial target number of items to fetch via scan, "
            "before client-side filtering/sorting/limiting."
        ),
    )
    default_scan_fetch_target: int = Field(
        default=_DEFAULT_SCAN_SIZE,
        gt=0,
        description=(
            "Default target number of items to fetch during a scan operation "
            "when no specific limit is provided but client-side filtering might occur. "
            "Helps balance fetching enough data vs. excessive scanning."
        ),
    )

    _dynamodb_resource: Any = PrivateAttr(default=None)
    _dynamodb_table: Any = PrivateAttr(default=None)
    _dynamodb_client: Any = PrivateAttr(default=None)

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)
        data["connection"] = self.connection.to_dict(include_secure_params=include_secure_params)
        if "type" not in data:
            data["type"] = self.type
        return data

    def model_post_init(self, __context: Any) -> None:
        try:
            session = self.connection.get_boto3_session()
            self._dynamodb_resource = session.resource("dynamodb")
            self._dynamodb_client = session.client("dynamodb")
            self._dynamodb_table = self._get_or_create_table()
            region = session.region_name or "default"
            logger.debug(f"DynamoDB backend (Scan Only) connected to table '{self.table_name}' in region '{region}'.")
        except ClientError as e:
            logger.error(f"Failed to initialize DynamoDB connection or table: {e}")
            raise DynamoDBMemoryError(f"Failed to initialize DynamoDB connection or table: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing DynamoDB backend: {e}")
            raise DynamoDBMemoryError(f"Unexpected error initializing DynamoDB backend: {e}") from e

    def _get_or_create_table(self):
        if self._dynamodb_resource is None:
            raise DynamoDBMemoryError("DynamoDB resource not initialized.")
        table = self._dynamodb_resource.Table(self.table_name)
        try:
            table.load()
            logger.debug(f"DynamoDB table '{self.table_name}' found.")
            return table
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                if self.create_if_not_exist:
                    logger.info(f"DynamoDB table '{self.table_name}' not found. Attempting creation (Scan Only)...")
                    return self._create_table()
                else:
                    logger.error(f"DynamoDB table '{self.table_name}' not found and create_if_not_exist is False.")
                    raise DynamoDBMemoryError(f"DynamoDB table '{self.table_name}' not found.") from e
            else:
                raise

    def _create_table(self):
        """Creates the DynamoDB table WITHOUT any GSIs."""
        if self._dynamodb_resource is None:
            raise DynamoDBMemoryError("DynamoDB resource not initialized.")
        attribute_definitions = [
            {"AttributeName": self.partition_key_name, "AttributeType": "S"},
            {"AttributeName": self.sort_key_name, "AttributeType": "N"},
        ]
        key_schema = [
            {"AttributeName": self.partition_key_name, "KeyType": "HASH"},
            {"AttributeName": self.sort_key_name, "KeyType": "RANGE"},
        ]
        create_params = {
            "TableName": self.table_name,
            "AttributeDefinitions": attribute_definitions,
            "KeySchema": key_schema,
            "BillingMode": self.billing_mode,
        }
        if self.billing_mode == BillingMode.PROVISIONED:
            create_params["ProvisionedThroughput"] = {
                "ReadCapacityUnits": self.read_capacity_units,
                "WriteCapacityUnits": self.write_capacity_units,
            }
        try:
            table = self._dynamodb_resource.create_table(**create_params)
            logger.info(f"Waiting for table '{self.table_name}' to become active...")
            table.wait_until_exists()
            logger.info(f"DynamoDB table '{self.table_name}' created successfully (Scan Only - No GSIs).")
            return table
        except ClientError as e:
            logger.error(f"Failed to create DynamoDB table '{self.table_name}': {e}")
            raise DynamoDBMemoryError(f"Failed to create DynamoDB table '{self.table_name}': {e}") from e

    def _serialize_timestamp(self, ts: float | int | Decimal) -> Decimal:
        if isinstance(ts, (float, int)):
            if ts != ts or ts == float("inf") or ts == float("-inf"):
                raise ValueError(f"Cannot serialize non-finite float {ts} as timestamp")
            try:
                return Decimal(str(ts))
            except InvalidOperation:
                raise ValueError(f"Could not convert numeric value {ts} to Decimal")
        elif isinstance(ts, Decimal):
            if ts.is_infinite():
                raise ValueError(f"Cannot serialize infinite Decimal {ts} as timestamp")
            elif ts.is_nan():
                raise ValueError("Cannot serialize NaN Decimal as timestamp")
            return ts
        else:
            raise TypeError(f"Timestamp must be float, int, or Decimal, not {type(ts)}")

    def _deserialize_item(self, item: dict[str, Any]) -> Message | None:
        try:
            metadata_raw = item.get(self.metadata_attribute_name, {})
            metadata = metadata_raw
            timestamp_val = item.get(self.sort_key_name)
            timestamp = float(timestamp_val) if isinstance(timestamp_val, Decimal) else timestamp_val
            metadata[self.sort_key_name] = timestamp
            metadata[self.partition_key_name] = item.get(self.partition_key_name)
            return Message(
                role=MessageRole(item.get(self.role_attribute_name, MessageRole.USER.value)),
                content=item.get(self.content_attribute_name, ""),
                metadata=metadata,
            )
        except (TypeError, ValueError, KeyError, InvalidOperation) as e:
            logger.error(f"Error deserializing DynamoDB item {item.get(self.partition_key_name)}: {e}. Item: {item}")
            return None

    def add(self, message: Message) -> None:
        if self._dynamodb_table is None:
            raise DynamoDBMemoryError("DynamoDB table not initialized.")
        try:
            message_id = message.metadata.get(self.partition_key_name, str(uuid.uuid4()))
            timestamp_input = message.metadata.get(self.sort_key_name, time.time())
            timestamp_decimal = self._serialize_timestamp(timestamp_input)
            metadata_to_store = message.metadata.copy() if message.metadata else {}
            metadata_to_store[self.sort_key_name] = timestamp_decimal
            metadata_to_store[self.partition_key_name] = message_id
            processed_metadata = _convert_floats_to_decimals(metadata_to_store)
            logger.debug(
                f"Saving item with PK: {message_id}, SK: {timestamp_decimal},"
                f" Metadata: {json.dumps(processed_metadata, default=str)}"
            )
            item = {
                self.partition_key_name: message_id,
                self.sort_key_name: timestamp_decimal,
                self.role_attribute_name: message.role.value,
                self.content_attribute_name: message.content,
                self.metadata_attribute_name: processed_metadata,
            }
            self._dynamodb_table.put_item(Item=item)
        except (ClientError, ValueError, TypeError) as e:
            logger.error(f"Error adding message to DynamoDB: {e}")
            raise DynamoDBMemoryError(f"Error adding message to DynamoDB: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error adding message to DynamoDB: {e}")
            try:
                logger.error(f"Problematic metadata (attempted): {json.dumps(message.metadata, default=str)}")
            except Exception:
                logger.error("Could not serialize problematic metadata for logging.")
            raise DynamoDBMemoryError(f"Unexpected error adding message: {e}") from e

    def _scan_and_process(
        self, limit: int | None = None, filters: dict | None = None, query: str | None = None
    ) -> list[Message]:
        """Scans table, applies filters/query client-side, sorts, and limits."""
        if self._dynamodb_table is None:
            raise DynamoDBMemoryError("DynamoDB table not initialized.")
        logger.warning(
            f"Performing DynamoDB Scan on table '{self.table_name}' for search/get_all. This can be slow and costly."
        )

        scan_params = {}
        processed_filters = _convert_floats_to_decimals(filters) if filters else None
        filter_expression_attr = self._build_filter_expression_attr(processed_filters)
        if filter_expression_attr:
            scan_params["FilterExpression"] = filter_expression_attr
            logger.debug(f"Using server-side FilterExpression object: {filter_expression_attr}")
        elif filters:
            logger.debug("No filters provided or FilterExpression could not be built.")

        all_items = []
        last_evaluated_key = None
        items_processed = 0
        scan_fetch_target = (limit * self.scan_fetch_target_multiplier) if limit else self.default_scan_fetch_target
        try:
            while True:
                if last_evaluated_key:
                    scan_params["ExclusiveStartKey"] = last_evaluated_key

                scan_params["Limit"] = (
                    min(self._MAX_SCAN_PAGE_LIMIT, scan_fetch_target - items_processed)
                    if limit
                    else self._MAX_SCAN_PAGE_LIMIT
                )
                if scan_params["Limit"] <= 0 and limit:
                    break

                logger.debug(f"Scanning page with params: {scan_params}")
                response = self._dynamodb_table.scan(**scan_params)
                page_items = response.get("Items", [])
                all_items.extend(page_items)
                items_processed += len(page_items)
                logger.debug(f"Scan page returned {len(page_items)} items. Total fetched: {items_processed}.")

                last_evaluated_key = response.get("LastEvaluatedKey")
                if not last_evaluated_key or (limit and items_processed >= scan_fetch_target):
                    if limit and items_processed >= scan_fetch_target:
                        logger.debug(f"Reached scan fetch target ({scan_fetch_target}), stopping pagination.")
                    break

            logger.info(f"Scan completed. Fetched {len(all_items)} total items.")

            messages = [self._deserialize_item(item) for item in all_items]
            valid_messages = [msg for msg in messages if msg is not None]
            logger.debug(f"Deserialized {len(valid_messages)} valid messages.")

            if filters:
                original_count = len(valid_messages)
                valid_messages = self._apply_filters_client_side_messages(valid_messages, filters)
                logger.debug(
                    f"Applied client-side filters ({filters}). "
                    f"Count changed from {original_count} to {len(valid_messages)}."
                )

            if query:
                original_count = len(valid_messages)
                query_lower = query.lower()
                valid_messages = [msg for msg in valid_messages if query_lower in msg.content.lower()]
                logger.debug(
                    f"Applied client-side text query ('{query}')."
                    f" Count changed from {original_count} to {len(valid_messages)}."
                )

            valid_messages.sort(key=lambda m: m.metadata.get(self.sort_key_name, 0))
            logger.debug("Sorted messages by timestamp.")

            if limit is not None and limit > 0:
                original_count = len(valid_messages)
                final_results = valid_messages[-limit:]
                logger.debug(
                    f"Applied final limit ({limit}). Count changed from {original_count} to {len(final_results)}."
                )
            else:
                final_results = valid_messages
                logger.debug("No final limit applied.")

            return final_results

        except ClientError as e:
            logger.error(f"Error scanning DynamoDB table '{self.table_name}': {e}")
            logger.error(f"Scan parameters used: {scan_params}")  # Log params on error
            raise DynamoDBMemoryError(f"Error scanning DynamoDB table: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during scan processing: {e}")
            raise DynamoDBMemoryError(f"Unexpected error during scan processing: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Retrieves messages via table scan, sorts chronologically."""
        return self._scan_and_process(limit=limit)

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Searches messages via table scan, applying filters and query client-side."""
        return self._scan_and_process(limit=limit, filters=filters, query=query)

    def _build_filter_expression_attr(self, filters: dict | None) -> Attr | None:
        """Builds a combined Attr object for FilterExpression."""
        if not filters:
            return None

        fe: Attr | None = None
        for key, value in filters.items():
            try:
                current_cond = Attr(f"{self.metadata_attribute_name}.{key}").eq(value)
            except Exception as e:
                logger.error(f"Could not build Attr condition for filter key '{key}': {e}. Skipping this filter.")
                continue

            if fe is None:
                fe = current_cond
            else:
                fe = fe & current_cond

        if fe:
            logger.debug(f"Built FilterExpression Attr object: {fe}")
        return fe

    def _apply_filters_client_side_messages(self, messages: list[Message], filters: dict) -> list[Message]:
        """Applies filters to a list of Message objects (client-side)."""
        if not filters:
            return messages
        filtered_messages = []
        processed_filters = _convert_floats_to_decimals(filters)

        for msg in messages:
            match = True
            for key, filter_value in processed_filters.items():
                metadata_value = msg.metadata.get(key)
                try:
                    if isinstance(filter_value, Decimal) and isinstance(metadata_value, (float, int)):
                        metadata_value_cmp = Decimal(str(metadata_value))
                    else:
                        metadata_value_cmp = metadata_value
                    if isinstance(filter_value, list):
                        if metadata_value_cmp not in filter_value:
                            match = False
                            break
                    elif metadata_value_cmp != filter_value:
                        match = False
                        break
                except (TypeError, ValueError, InvalidOperation):
                    match = False
                    break
            if match:
                filtered_messages.append(msg)
        return filtered_messages

    def is_empty(self) -> bool:
        if self._dynamodb_table is None:
            raise DynamoDBMemoryError("DynamoDB table not initialized.")
        try:
            response = self._dynamodb_table.scan(Limit=1, Select="COUNT")
            return response.get("Count", 0) == 0
        except ClientError as e:
            logger.error(f"Error checking emptiness for DynamoDB table '{self.table_name}': {e}")
            raise DynamoDBMemoryError(f"Error checking if DynamoDB memory is empty: {e}") from e

    def clear(self) -> None:
        """Clears memory by deleting all items via scan (slow, costly)."""
        if self._dynamodb_table is None:
            raise DynamoDBMemoryError("DynamoDB table not initialized.")
        logger.warning(
            f"Clearing all items from DynamoDB table '{self.table_name}'"
            f" via Scan/Delete. This may take time and consume capacity."
        )
        try:
            with self._dynamodb_table.batch_writer() as batch:
                scan_params = {"ProjectionExpression": f"{self.partition_key_name}, {self.sort_key_name}"}
                while True:
                    response = self._dynamodb_table.scan(**scan_params)
                    items = response.get("Items", [])
                    if not items:
                        break
                    for item in items:
                        batch.delete_item(
                            Key={
                                self.partition_key_name: item[self.partition_key_name],
                                self.sort_key_name: item[self.sort_key_name],
                            }
                        )
                    if "LastEvaluatedKey" in response:
                        scan_params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                    else:
                        break
            logger.info(f"DynamoDB Memory ({self.table_name}): Finished clearing items.")
        except ClientError as e:
            logger.error(f"Error clearing DynamoDB table '{self.table_name}': {e}")
            raise DynamoDBMemoryError(f"Error clearing DynamoDB memory: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error clearing DynamoDB memory: {e}")
            raise DynamoDBMemoryError(f"Unexpected error clearing DynamoDB memory: {e}") from e
