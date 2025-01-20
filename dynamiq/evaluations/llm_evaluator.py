import json
from typing import Any
from warnings import warn

from dynamiq.nodes import Node
from dynamiq.prompts import Message, Prompt
from dynamiq.utils.json_parser import parse_llm_json_output

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
        inputs=[{"name": "answer", "type": str}],
        outputs=[{"name": "score", "type": int}],
        examples=[
            {
                "inputs": {"answer": "Damn, this is straight outta hell!!!"},
                "outputs": {"score": 1},
            },
            {
                "inputs": {answer": "Football is the most popular sport."},
                "outputs": {"score": 0},
            },
        ],
        llm=llm,
    )

    predicted_answers = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
    ]
    results = evaluator.run(answer=predicted_answers)
    print(results)
    # Output: {'results': [{'score': 0}, {'score': 0}]}
    ```
    """

    def __init__(
        self,
        instructions: str,
        outputs: list[dict[str, Any]],
        examples: list[dict[str, Any]] | None = None,
        inputs: list[dict[str, Any]] | None = None,
        *,
        raise_on_failure: bool = True,
        llm: Node,
        strings_to_omit_from_llm_output: tuple[str] = STRINGS_TO_OMIT_FROM_LLM_EVALUATOR_OUTPUT,
    ):
        """
        Initializes an instance of LLMEvaluator.

        Args:
            instructions (str): The prompt instructions to use for evaluation.
            outputs (List[Dict[str, Any]]): A list of dictionaries defining the outputs.
                Each output dict should have keys "name" and "type", where "name" is the
                output name and "type" is its type.
            examples (Optional[List[Dict[str, Any]]]): Few-shot examples conforming to the expected input and
                output format as defined in the `inputs` and `outputs` parameters. Each example is a
                dictionary with keys "inputs" and "outputs". They contain the input and output as
                dictionaries respectively.
            inputs (Optional[List[Dict[str, Any]]]): A list of dictionaries defining the inputs.
                Each input dict should have keys "name" and "type", where "name" is the
                input name and "type" is its type. Defaults to None.
            raise_on_failure (bool): If True, the component will raise an exception on an
                unsuccessful API call.
            llm (Node): The LLM node to use for evaluation.
            strings_to_omit_from_llm_output (Tuple[str]): A tuple of strings to omit from the LLM output.
        """
        if inputs is None:
            inputs = []
        if examples is None:
            examples = []
        self._validate_init_parameters(inputs, outputs, examples)
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
        template = self._prepare_prompt_template()
        message = Message(role="user", content=template)
        self.prompt = Prompt(messages=[message])

        self.llm = llm
        self.strings_to_omit_from_llm_output = strings_to_omit_from_llm_output

    @staticmethod
    def _validate_init_parameters(
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
        if not isinstance(examples, list) or not all(isinstance(example, dict) for example in examples):
            msg = f"LLM evaluator expects examples to be a list of dictionaries but received {examples}."
            raise ValueError(msg)

        for example in examples:
            if (
                not all(k in example for k in ("inputs", "outputs"))
                or not all(isinstance(example[param], dict) for param in ["inputs", "outputs"])
                or not all(isinstance(key, str) for param in ["inputs", "outputs"] for key in example[param])
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
        self._validate_input_parameters(expected=expected_inputs, received=inputs)

        if self.inputs:
            input_names = list(inputs.keys())
            values = list(zip(*inputs.values()))
            list_of_input_data = [dict(zip(input_names, v)) for v in values]
        else:
            # If no inputs are provided, create a list with a single empty dictionary
            list_of_input_data = [{}]

        results: list[dict[str, Any]] = []
        errors = 0

        for input_data in list_of_input_data:
            # Pass the prompt and input_data to LLM
            try:
                result = self.llm.execute(input_data=input_data, prompt=self.prompt)
            except Exception as e:
                msg = f"Error while generating response for input {input_data}: {e}"
                if self.raise_on_failure:
                    raise ValueError(msg)
                warn(msg)
                results.append(None)
                errors += 1
                continue

            expected_output_keys = [outp["name"] for outp in self.outputs]
            content = self._cleanup_output_content(result["content"])

            parsed_result = self._parse_and_validate_json_output(expected_keys=expected_output_keys, content=content)
            if parsed_result is not None:
                results.append(parsed_result)
            else:
                results.append(None)
                errors += 1

        if errors > 0:
            msg = f"LLM evaluator failed for {errors} out of {len(list_of_input_data)} inputs."
            warn(msg)

        return {"results": results}

    def _prepare_prompt_template(self) -> str:
        """
        Prepares the prompt template.

        Returns:
            str: The prompt template.
        """
        prompt_parts = [
            "Instructions:",
            self.instructions.strip(),
        ]

        # Check if outputs are provided
        if self.outputs:
            # Prepare expected_output_section
            expected_output_dict = {outp["name"]: self._get_placeholder_for_type(outp["type"]) for outp in self.outputs}
            expected_output = json.dumps(expected_output_dict, indent=2)
            prompt_parts.extend(
                [
                    (
                        "\nYour task is to generate a JSON object that contains the following keys "
                        "and their corresponding values."
                    ),
                    "The output must be a valid JSON object and should exactly match the specified structure.",
                    "Do not include any additional text, explanations, or markdown.",
                    "Expected JSON format:",
                    expected_output,
                ]
            )

        if self.examples:
            # Prepare examples_section with explicit labels
            examples_parts = []
            for idx, example in enumerate(self.examples, start=1):
                example_input = json.dumps(example["inputs"], indent=2)
                example_output = json.dumps(example["outputs"], indent=2)
                example_text = f"Example {idx}:\n" f"Input:\n{example_input}\n" f"Expected Output:\n{example_output}"
                examples_parts.append(example_text)
            examples_section = "\n\n".join(examples_parts)
            prompt_parts.append("\nHere are some examples:")
            prompt_parts.append(examples_section)

        if self.inputs:
            # Prepare inputs_section with placeholders using {{ variable_name }}
            inputs_section = (
                "{\n" + ",\n".join([f'  "{inp["name"]}": {{{{ {inp["name"]} }}}}' for inp in self.inputs]) + "\n}"
            )
            prompt_parts.append("\nNow, process the following input:")
            prompt_parts.append(inputs_section)

        prompt_parts.append("\nProvide the output as per the format specified above.")

        return "\n".join(prompt_parts)

    def _get_placeholder_for_type(self, tp):
        """
        Generates a placeholder value based on the type.

        Args:
            tp: The type to generate a placeholder for.

        Returns:
            An example value corresponding to the type.
        """
        if tp == str:
            return "string_value"
        elif tp == int:
            return 0
        elif tp == float:
            return 0.0
        elif tp == bool:
            return True
        elif tp == list:
            return []
        elif tp == dict:
            return {}
        else:
            return f"{tp}"

    @staticmethod
    def _get_type_name(tp):
        """
        Helper function to get the name of a type, including typing types.

        Args:
            tp: The type to get the name of.

        Returns:
            str: The name of the type.
        """
        if hasattr(tp, "__name__"):
            return tp.__name__
        elif hasattr(tp, "_name") and tp._name:
            args = ", ".join(LLMEvaluator._get_type_name(arg) for arg in tp.__args__)
            return f"{tp._name}[{args}]"
        else:
            return str(tp)

    @staticmethod
    def _validate_input_parameters(expected: dict[str, Any], received: dict[str, Any]) -> None:
        """
        Validates the input parameters.

        Args:
            expected (Dict[str, Any]): The expected input parameters with their types.
            received (Dict[str, Any]): The received input parameters.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if expected:
            for param in expected.keys():
                if param not in received:
                    msg = f"LLM evaluator expected input parameter '{param}' but received {list(received.keys())}."
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
                    "LLM evaluator expects all input lists to have the same length but received input lengths "
                    f"{[len(value) for value in inputs]}."
                )
                raise ValueError(msg)
        else:
            if received:
                msg = f"LLM evaluator does not expect any input parameters but received {list(received.keys())}."
                raise ValueError(msg)

    def _parse_and_validate_json_output(self, expected_keys: list[str], content: str) -> dict[str, Any]:
        """
        Parses the LLM output content as JSON and validates that it contains the expected keys.

        Args:
            expected_keys (List[str]): Expected output keys.
            content (str): The received output as a JSON string.

        Returns:
            Dict[str, Any]: The parsed JSON output if valid, otherwise None.

        Raises:
            ValueError: If the output is invalid and raise_on_failure is True.
        """
        try:
            parsed_output = parse_llm_json_output(content)
        except json.JSONDecodeError:
            msg = f"Response from LLM evaluator is not a valid JSON: {content}."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return None

        if not all(key in parsed_output for key in expected_keys):
            msg = (
                f"Expected response from LLM evaluator to have keys {expected_keys}, "
                f"but got {list(parsed_output.keys())}."
            )
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return None

        return parsed_output

    def _cleanup_output_content(self, content: str) -> str:
        """
        Cleans up the output content by removing unwanted strings.

        Args:
            content (str): The content to clean up.

        Returns:
            str: The cleaned content.
        """
        for omit_string in self.strings_to_omit_from_llm_output:
            content = content.replace(omit_string, "")
        return content.strip()
