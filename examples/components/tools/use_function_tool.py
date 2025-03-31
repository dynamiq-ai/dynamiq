from pydantic import BaseModel

from dynamiq.nodes.tools.function_tool import FunctionTool, function_tool


class AddNumbersInputSchema(BaseModel):
    a: int = -1
    b: int = -1


# Example usage without decorator
class AddNumbersTool(FunctionTool):
    name: str = "Add Numbers Tool"
    description: str = "A tool that adds two numbers together."

    def run_func(self, input_data: AddNumbersInputSchema, **kwargs) -> int:
        """Add two numbers together."""
        return input_data.a + input_data.b


# Example usage with decorator
@function_tool
def multiply_numbers(a: int, b: int, **kwargs) -> int:
    """Multiply two numbers together."""
    return a * b


if __name__ == "__main__":
    # Usage
    add_tool = AddNumbersTool()

    input_data = AddNumbersInputSchema()
    input_data.a = 3
    input_data.b = 5

    result = add_tool.execute(input_data=input_data)
    print(result)  # Output: {"content": 8}

    # Usage
    multiply_tool = multiply_numbers()
    result = multiply_tool.execute(input_data=input_data)
    print(result)  # Output: {"content": 15}
