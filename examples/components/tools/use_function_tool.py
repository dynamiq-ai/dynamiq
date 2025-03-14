from dynamiq.nodes.tools.function_tool import FunctionTool, function_tool

if __name__ == "__main__":
    # Example usage without decorator
    class AddNumbersTool(FunctionTool[int]):
        name: str = "Add Numbers Tool"
        description: str = "A tool that adds two numbers together."

        def run_tool(self, a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

    # Usage
    add_tool = AddNumbersTool()
    result = add_tool.execute({"a": 5, "b": 3})
    print(result)  # Output: {"content": 8}

    @function_tool
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    # Usage
    multiply_tool = multiply_numbers()
    result = multiply_tool.execute({"a": 5, "b": 3})
    print(result)  # Output: {"content": 15}

    # Get schema
    print(add_tool.get_schema())
    print(multiply_tool.get_schema())
