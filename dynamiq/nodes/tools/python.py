"""The shared input schema for the Python code-execution nodes.

The RestrictedPython ``Python`` and ``PythonCodeExecutor`` nodes were removed: running untrusted code
under RestrictedPython shares the interpreter with the code it tries to restrict, which is not a real
security boundary. Code execution now runs in the Dynamiq runtime's OS-level sandbox. Only this input
schema remains here, still used by the Monty executor and by the runtime's sandbox node.
"""

from pydantic import BaseModel, ConfigDict

from dynamiq.utils import format_value


class PythonInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    def to_dict(self, **kwargs) -> dict:
        return {field: format_value(value, **kwargs) for field, value in self.model_extra.items()}
