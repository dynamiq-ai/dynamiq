import json
from typing import Any
from warnings import warn

from dynamiq.nodes import Node
from dynamiq.prompts import Message, Prompt

STRINGS_TO_OMIT_FROM_LLM_EVALUATOR_OUTPUT = ("```json", "```")


class LLMEvaluator:
    """
    Uses an LLM to evaluate inputs based on a prompt containing instructions and examples.

    The inputs and outputs are user-defined depending on the desired metric. The output
    is a dictionary with a key `results` containing a list of evaluation results. Each result
    is a dictionary with user-defined keys and values as specified in the outputs.

    **Usage example:**

    ```python
    from typing import List
    from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator

    llm = ...  # Your LLM node instance

    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[{"name": "predicted_answers", "type": List[str]}],
        outputs=[{"name": "score", "type": int}],
        examples=[
            {
                "inputs": {"predicted_answers": "Damn, this is straight outta hell!!!"},
                "outputs": {"score": 1},
            },
            {
                "inputs": {"predicted_answers": "Football is the most popular sport."},
                "outputs": {"score": 0},
            },
        ],
        llm=llm,
    )

    predicted_answers = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
    ]
    results = evaluator.run(predicted_answers=predicted_answers)
    print(results)
    # Output: {'results': [{'score': 0}, {'score': 0}]}
    ```
    """

    def __init__(
        self,
        instructions: str,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        examples: list[dict[str, Any]],
        *,
        raise_on_failure: bool = True,
        llm: Node,
        strings_to_omit_from_llm_output: tuple[str] = STRINGS_TO_OMIT_FROM_LLM_EVALUATOR_OUTPUT,
    ):
        """
        Initializes an instance of LLMEvaluator.

        Args:
            instructions (str): The prompt instructions to use for evaluation.
            inputs (List[Dict[str, Any]]): A list of dictionaries defining the inputs.
                Each input dict should have keys "name" and "type", where "name" is the
                input name and "type" is its type.
            outputs (List[Dict[str, Any]]): A list of dictionaries defining the outputs.
                Each output dict should have keys "name" and "type", where "name" is the
                output name and "type" is its type.
            examples (List[Dict[str, Any]]): Few-shot examples conforming to the expected input and
                output format as defined in the `inputs` and `outputs` parameters. Each example is a
                dictionary with keys "inputs" and "outputs". They contain the input and output as
                dictionaries respectively.
            raise_on_failure (bool): If True, the component will raise an exception on an
                unsuccessful API call.
            llm (Node): The LLM node to use for evaluation.
            strings_to_omit_from_llm_output (Tuple[str]): A tuple of strings to omit from the LLM output.
        """
        self.validate_init_parameters(inputs, outputs, examples)
        self.raise_on_failure = raise_on_failure
        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs
        self.examples = examples
        self.api_params = {}

        default_generation_kwargs = {
            "response_format": {"type": "json_object"},
            "seed": 42,
        }
        user_generation_kwargs = self.api_params.get("generation_kwargs", {})
        merged_generation_kwargs = {
            **default_generation_kwargs,
            **user_generation_kwargs,
        }
        self.api_params["generation_kwargs"] = merged_generation_kwargs

        # Prepare the prompt with placeholders
        template = self.prepare_template()
        message = Message(role="user", content=template)
        self.prompt = Prompt(messages=[message])

        self.llm = llm
        self.strings_to_omit_from_llm_output = strings_to_omit_from_llm_output

    @staticmethod
    def validate_init_parameters(
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        examples: list[dict[str, Any]],
    ):
        """
        Validates the initialization parameters.

        Args:
            inputs (List[Dict[str, Any]]): The inputs to validate.
            outputs (List[Dict[str, Any]]): The outputs to validate.
            examples (List[Dict[str, Any]]): The examples to validate.

        Raises:
            ValueError: If the inputs or outputs are not correctly formatted.
        """
        # Validate inputs
        if not isinstance(inputs, list) or not all(isinstance(inp, dict) for inp in inputs):
            msg = "LLM evaluator expects inputs to be a list of dictionaries."
            raise ValueError(msg)
        for inp in inputs:
            if "name" not in inp or "type" not in inp:
                msg = f"Each input dict must have 'name' and 'type' keys. Missing in {inp}."
                raise ValueError(msg)
            if not isinstance(inp["name"], str):
                msg = f"Input 'name' must be a string. Got {inp['name']}."
                raise ValueError(msg)
            # No type check on 'type' to allow types from 'typing' module

        # Validate outputs
        if not isinstance(outputs, list) or not all(isinstance(outp, dict) for outp in outputs):
            msg = "LLM evaluator expects outputs to be a list of dictionaries."
            raise ValueError(msg)
        for outp in outputs:
            if "name" not in outp or "type" not in outp:
                msg = f"Each output dict must have 'name' and 'type' keys. Missing in {outp}."
                raise ValueError(msg)
            if not isinstance(outp["name"], str):
                msg = f"Output 'name' must be a string. Got {outp['name']}."
                raise ValueError(msg)
            # No type check on 'type' to allow types from 'typing' module

        # Validate examples
        if not isinstance(examples, list) or not all(
            isinstance(example, dict) for example in examples
        ):
            msg = f"LLM evaluator expects examples to be a list of dictionaries but received {examples}."
            raise ValueError(msg)

        for example in examples:
            if (
                not all(k in example for k in ("inputs", "outputs"))
                or not all(isinstance(example[param], dict) for param in ["inputs", "outputs"])
                or not all(
                    isinstance(key, str)
                    for param in ["inputs", "outputs"]
                    for key in example[param]
                )
            ):
                msg = (
                    f"Each example must have 'inputs' and 'outputs' as dictionaries with string keys, "
                    f"but received {example}."
                )
                raise ValueError(msg)

    def run(self, **inputs) -> dict[str, Any]:
        """
        Runs the LLM evaluator.

        Args:
            inputs: The input values to evaluate. Keys are input names, values are lists.

        Returns:
            Dict[str, Any]: A dictionary with a 'results' key containing the evaluation results.

        Raises:
            ValueError: If input parameters are invalid or LLM execution fails.
        """
        expected_inputs = {inp["name"]: inp["type"] for inp in self.inputs}
        self.validate_input_parameters(expected=expected_inputs, received=inputs)

        input_names = list(inputs.keys())
        values = list(zip(*inputs.values()))
        list_of_input_names_to_values = [dict(zip(input_names, v)) for v in values]

        results: list[dict[str, Any]] = []
        errors = 0
        for input_data in list_of_input_names_to_values:
            # Pass the prompt and input_data to LLM
            try:
                result = self.llm.execute(input_data=input_data, prompt=self.prompt)
            except Exception as e:
                msg = f"Error while generating response for prompt: {self.prompt}. Error: {e}"
                if self.raise_on_failure:
                    raise ValueError(msg)
                warn(msg)
                results.append(None)
                errors += 1
                continue

            expected_output_keys = [outp["name"] for outp in self.outputs]
            content = self.cleanup_output_content(result["content"])

            if self.is_valid_json_and_has_expected_keys(expected=expected_output_keys, received=content):
                parsed_result = json.loads(content)
                results.append(parsed_result)
            else:
                results.append(None)
                errors += 1

        if errors > 0:
            msg = f"LLM evaluator failed for {errors} out of {len(list_of_input_names_to_values)} inputs."
            warn(msg)

        return {"results": results}

    def prepare_template(self) -> str:
        """
        Prepare the prompt template.

        Returns:
            str: The prompt template.
        """
        # Prepare inputs_section with placeholders using {{ variable_name }}
        inputs_section = (
            "{\n" + ",\n".join([f'  "{inp["name"]}": {{{{ {inp["name"]} }}}}' for inp in self.inputs]) + "\n}"
        )

        # Prepare examples_section
        examples_section = "\n\n".join(
            [
                "Inputs:\n"
                + json.dumps(example["inputs"], indent=2)
                + "\nOutputs:\n"
                + json.dumps(example["outputs"], indent=2)
                for example in self.examples
            ]
        )

        # Prepare output descriptions
        def get_type_name(tp):
            """Helper function to get the name of a type, including typing types."""
            if hasattr(tp, "__name__"):
                return tp.__name__
            elif hasattr(tp, "_name") and tp._name:
                args = ", ".join(get_type_name(arg) for arg in tp.__args__)
                return f"{tp._name}[{args}]"
            else:
                return str(tp)

        output_descriptions = [f'  "{outp["name"]}": {get_type_name(outp["type"])}' for outp in self.outputs]
        output_section = "{\n" + ",\n".join(output_descriptions) + "\n}"

        prompt_parts = [
            "Instructions:",
            self.instructions.strip(),
            "\nGenerate the response in JSON format, omitting extra keys and markdown syntax elements.",
            "Include the following keys with their types:",
            output_section,
        ]

        if self.examples:
            prompt_parts.append("\nConsider the following examples:")
            prompt_parts.append(examples_section)

        prompt_parts.append("\nCurrent Inputs:")
        prompt_parts.append(inputs_section)
        prompt_parts.append("Outputs:")

        return "\n".join(prompt_parts)

    @staticmethod
    def validate_input_parameters(
        expected: dict[str, Any], received: dict[str, Any]
    ) -> None:
        """
        Validates the input parameters.

        Args:
            expected (Dict[str, Any]): The expected input parameters with their types.
            received (Dict[str, Any]): The received input parameters.

        Raises:
            ValueError: If input parameters are invalid.
        """
        for param in expected.keys():
            if param not in received:
                msg = f"LLM evaluator expected input parameter '{param}' but received only {list(received.keys())}."
                raise ValueError(msg)

        if not all(isinstance(value, list) for value in received.values()):
            msg = (
                "LLM evaluator expects all input values to be lists but received "
                f"{[type(value) for value in received.values()]}."
            )
            raise ValueError(msg)

        inputs = received.values()
        length = len(next(iter(inputs)))
        if not all(len(value) == length for value in inputs):
            msg = (
                f"LLM evaluator expects all input lists to have the same length but received input lengths "
                f"{[len(value) for value in inputs]}."
            )
            raise ValueError(msg)

    def is_valid_json_and_has_expected_keys(
        self, expected: list[str], received: str
    ) -> bool:
        """
        Ensures the output is a valid JSON with the expected keys.

        Args:
            expected (List[str]): Expected output keys.
            received (str): The received output as a JSON string.

        Raises:
            ValueError: If the output is invalid.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            parsed_output = json.loads(received)
        except json.JSONDecodeError:
            msg = f"Response from LLM evaluator is not a valid JSON: {received}."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return False

        if not all(output in parsed_output for output in expected):
            msg = f"Expected response from LLM evaluator to have keys {expected}, but got {list(parsed_output.keys())}."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return False

        return True

    def cleanup_output_content(self, content: str):
        """
        Cleans up the output content by removing unwanted strings.

        Args:
            content (str): The content to clean up.
        """
        for omit_string in self.strings_to_omit_from_llm_output:
            content = content.replace(omit_string, "")
        return content.strip()
