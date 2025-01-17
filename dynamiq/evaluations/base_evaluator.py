from functools import cached_property

from pydantic import BaseModel, ConfigDict, computed_field


class BaseEvaluator(BaseModel):
    """
    Base class for evaluators.

    Attributes:
        name (str): Name of the evaluator.
    """

    name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def type(self) -> str:
        """
        Compute the type identifier for the evaluator.

        Returns:
            str: A string representing the module and class name.
        """
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def run(self) -> list[float]:
        """
        Executes the evaluator.
        Must be overridden by subclasses.

        Returns:
            list[float]: Scores for each reference/answer pair.
        """
        raise NotImplementedError("Subclasses must implement this method.")
