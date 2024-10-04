import json
from typing import Any

from dynamiq.utils import JsonWorkflowEncoder


class BaseSerializer:
    """
    Base class for serializers providing interface for dumps and loads methods.
    """

    def dumps(self, value: Any) -> str:
        """
        Serialize the given value to a string.

        Args:
            value (Any): The value to be serialized.

        Returns:
            str: The serialized string representation of the value.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def loads(self, value: Any) -> Any:
        """
        Deserialize the given value from a string.

        Args:
            value (Any): The serialized string to be deserialized.

        Returns:
            Any: The deserialized value.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError


class StringSerializer(BaseSerializer):
    """
    Serializer that converts values to and from strings.
    """

    def dumps(self, value: Any) -> str:
        """
        Convert the given value to a string.

        Args:
            value (Any): The value to be converted to a string.

        Returns:
            str: The string representation of the value.
        """
        return str(value)

    def loads(self, value: Any) -> Any:
        """
        Return the input value as is, since it's already a string.

        Args:
            value (Any): The value to be deserialized (expected to be a string).

        Returns:
            Any: The input value, unchanged.
        """
        return value


class JsonSerializer(BaseSerializer):
    """
    Serializer that converts values to and from JSON format.
    """

    def dumps(self, value: Any) -> str:
        """
        Serialize the given value to a JSON string.

        Args:
            value (Any): The value to be serialized to JSON.

        Returns:
            str: The JSON string representation of the value.
        """
        return json.dumps(value, cls=JsonWorkflowEncoder)

    def loads(self, value: str | None) -> Any:
        """
        Deserialize the given JSON string to a Python object.

        Args:
            value (str | None): The JSON string to be deserialized, or None.

        Returns:
            Any: The deserialized Python object, or None if the input is None.
        """
        if value is None:
            return None
        return json.loads(value)


class JsonPickleSerializer(BaseSerializer):
    """
    Serializer that uses jsonpickle to convert complex Python objects to and from JSON format.
    """

    def dumps(self, value: Any) -> str:
        """
        Serialize the given value to a JSON string using jsonpickle.

        Args:
            value (Any): The value to be serialized.

        Returns:
            str: The JSON string representation of the value.
        """
        import jsonpickle

        return jsonpickle.encode(value)

    def loads(self, value: str) -> Any:
        """
        Deserialize the given JSON string to a Python object using jsonpickle.

        Args:
            value (str): The JSON string to be deserialized.

        Returns:
            Any: The deserialized Python object.
        """
        import jsonpickle

        return jsonpickle.decode(value)  # nosec
