from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator


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
