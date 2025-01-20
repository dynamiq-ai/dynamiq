from dynamiq.evaluations.llm_evaluator import LLMEvaluator


def test_prepare_prompt_template():
    # Sample data for the test
    instructions = "Evaluate the following answers for correctness."
    inputs = [
        {"name": "question", "type": str},
        {"name": "answer", "type": str},
    ]
    outputs = [
        {"name": "is_correct", "type": bool},
        {"name": "feedback", "type": str},
    ]
    examples = [
        {
            "inputs": {"question": "What is 2 + 2?", "answer": "4"},
            "outputs": {"is_correct": True, "feedback": "Correct answer."},
        },
        {
            "inputs": {"question": "What is the capital of France?", "answer": "Berlin"},
            "outputs": {"is_correct": False, "feedback": "Incorrect. The capital of France is Paris."},
        },
    ]

    # Create an instance of LLMEvaluator
    evaluator = LLMEvaluator(
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        llm=None,  # LLM is not needed for this test
    )

    # Call the method to test
    prompt_template = evaluator._prepare_prompt_template()

    # Prepare the expected prompt
    expected_prompt = """Instructions:
Evaluate the following answers for correctness.

Your task is to generate a JSON object that contains the following keys and their corresponding values.
The output must be a valid JSON object and should exactly match the specified structure.
Do not include any additional text, explanations, or markdown.
Expected JSON format:
{
  "is_correct": true,
  "feedback": "string_value"
}

Here are some examples:
Example 1:
Input:
{
  "question": "What is 2 + 2?",
  "answer": "4"
}
Expected Output:
{
  "is_correct": true,
  "feedback": "Correct answer."
}

Example 2:
Input:
{
  "question": "What is the capital of France?",
  "answer": "Berlin"
}
Expected Output:
{
  "is_correct": false,
  "feedback": "Incorrect. The capital of France is Paris."
}

Now, process the following input:
{
  "question": {{ question }},
  "answer": {{ answer }}
}

Provide the output as per the format specified above."""

    # Normalize line endings to ensure consistency
    prompt_template_normalized = prompt_template.replace("\r\n", "\n").strip()
    expected_prompt_normalized = expected_prompt.replace("\r\n", "\n").strip()

    # Assert that the generated prompt matches the expected prompt
    assert prompt_template_normalized == expected_prompt_normalized


def test_prepare_prompt_template_no_examples():
    # Sample data for the test
    instructions = "Evaluate the following answers for correctness."
    inputs = [
        {"name": "question", "type": str},
        {"name": "answer", "type": str},
    ]
    outputs = [
        {"name": "is_correct", "type": bool},
        {"name": "feedback", "type": str},
    ]
    examples = []  # No examples provided

    # Create an instance of LLMEvaluator
    evaluator = LLMEvaluator(
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        llm=None,  # LLM is not needed for this test
    )

    # Call the method to test
    prompt_template = evaluator._prepare_prompt_template()

    # Prepare the expected prompt
    expected_prompt = """Instructions:
Evaluate the following answers for correctness.

Your task is to generate a JSON object that contains the following keys and their corresponding values.
The output must be a valid JSON object and should exactly match the specified structure.
Do not include any additional text, explanations, or markdown.
Expected JSON format:
{
  "is_correct": true,
  "feedback": "string_value"
}

Now, process the following input:
{
  "question": {{ question }},
  "answer": {{ answer }}
}

Provide the output as per the format specified above."""

    # Normalize line endings to ensure consistency
    prompt_template_normalized = prompt_template.replace("\r\n", "\n").strip()
    expected_prompt_normalized = expected_prompt.replace("\r\n", "\n").strip()

    # Assert that the generated prompt matches the expected prompt
    assert prompt_template_normalized == expected_prompt_normalized


def test_prepare_prompt_template_single_input():
    # Sample data for the test
    instructions = "Determine if the provided text is positive or negative."
    inputs = [
        {"name": "text", "type": str},
    ]
    outputs = [
        {"name": "sentiment", "type": str},
    ]
    examples = [
        {
            "inputs": {"text": "I love this product!"},
            "outputs": {"sentiment": "positive"},
        },
        {
            "inputs": {"text": "This is the worst experience ever."},
            "outputs": {"sentiment": "negative"},
        },
    ]

    # Create an instance of LLMEvaluator
    evaluator = LLMEvaluator(
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        llm=None,  # LLM is not needed for this test
    )

    # Call the method to test
    prompt_template = evaluator._prepare_prompt_template()

    # Prepare the expected prompt
    expected_prompt = """Instructions:
Determine if the provided text is positive or negative.

Your task is to generate a JSON object that contains the following keys and their corresponding values.
The output must be a valid JSON object and should exactly match the specified structure.
Do not include any additional text, explanations, or markdown.
Expected JSON format:
{
  "sentiment": "string_value"
}

Here are some examples:
Example 1:
Input:
{
  "text": "I love this product!"
}
Expected Output:
{
  "sentiment": "positive"
}

Example 2:
Input:
{
  "text": "This is the worst experience ever."
}
Expected Output:
{
  "sentiment": "negative"
}

Now, process the following input:
{
  "text": {{ text }}
}

Provide the output as per the format specified above."""

    # Normalize line endings to ensure consistency
    prompt_template_normalized = prompt_template.replace("\r\n", "\n").strip()
    expected_prompt_normalized = expected_prompt.replace("\r\n", "\n").strip()

    # Assert that the generated prompt matches the expected prompt
    assert prompt_template_normalized == expected_prompt_normalized


def test_prepare_prompt_template_no_inputs():
    # Sample data for the test
    instructions = "Provide a random motivational quote."
    inputs = []  # No inputs provided
    outputs = [
        {"name": "quote", "type": str},
    ]
    examples = []  # No examples provided

    # Create an instance of LLMEvaluator
    evaluator = LLMEvaluator(
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        llm=None,  # LLM is not needed for this test
    )

    # Call the method to test
    prompt_template = evaluator._prepare_prompt_template()

    # Prepare the expected prompt
    expected_prompt = """Instructions:
Provide a random motivational quote.

Your task is to generate a JSON object that contains the following keys and their corresponding values.
The output must be a valid JSON object and should exactly match the specified structure.
Do not include any additional text, explanations, or markdown.
Expected JSON format:
{
  "quote": "string_value"
}

Provide the output as per the format specified above."""

    # Normalize line endings to ensure consistency
    prompt_template_normalized = prompt_template.replace("\r\n", "\n").strip()
    expected_prompt_normalized = expected_prompt.replace("\r\n", "\n").strip()

    # Assert that the generated prompt matches the expected prompt
    assert prompt_template_normalized == expected_prompt_normalized


def test_prepare_prompt_template_no_outputs():
    # Sample data for the test
    instructions = "Process the input and do nothing."
    inputs = [
        {"name": "data", "type": str},
    ]
    outputs = []  # No outputs provided
    examples = []  # No examples provided

    # Create an instance of LLMEvaluator
    evaluator = LLMEvaluator(
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        llm=None,  # LLM is not needed for this test
    )

    # Call the method to test
    prompt_template = evaluator._prepare_prompt_template()

    # Prepare the expected prompt
    expected_prompt = """Instructions:
Process the input and do nothing.

Now, process the following input:
{
  "data": {{ data }}
}

Provide the output as per the format specified above."""

    # Normalize line endings to ensure consistency
    prompt_template_normalized = prompt_template.replace("\r\n", "\n").strip()
    expected_prompt_normalized = expected_prompt.replace("\r\n", "\n").strip()

    # Assert that the generated prompt matches the expected prompt
    assert prompt_template_normalized == expected_prompt_normalized
