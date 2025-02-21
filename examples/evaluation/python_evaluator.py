from dynamiq.evaluations import PythonEvaluator


def run_metric(example_name, user_code, input_data_list):
    """
    Helper function to run a metric via PythonEvaluator.
    It prints the input data and resulting score.
    """
    print(f"\n=== Running {example_name} Metric ===")
    evaluator = PythonEvaluator(code=user_code)
    scores = evaluator.run(input_data_list=input_data_list)
    for idx, score in enumerate(scores):
        print(f"Sample {idx}: Data: {input_data_list[idx]} -> Score: {score}")


def run_exact_match_metric():
    user_code_exact = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    input_data_exact = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Barcelona"},
    ]
    run_metric("Exact Match", user_code_exact, input_data_exact)


def run_email_presence_metric():
    user_code_email = """
import re
def evaluate(answer):
    # Default email regex pattern
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+"
    return 1.0 if re.search(email_pattern, answer) else 0.0
"""
    input_data_email = [
        {"answer": "Please contact us at support@example.com for details."},
        {"answer": "My email is user.name123@domain.net."},
        {"answer": "No email provided here."},
    ]
    run_metric("Email Presence", user_code_email, input_data_email)


def run_phone_presence_metric():
    user_code_phone = """
import re
def evaluate(answer):
    # US-style phone number pattern with separators '-' or '.'
    phone_pattern = r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b"
    return 1.0 if re.search(phone_pattern, answer) else 0.0
"""
    input_data_phone = [
        {"answer": "Call me at 123-456-7890 for more info."},
        {"answer": "You can reach me at 987.654.3210."},
        {"answer": "No contact number here."},
    ]
    run_metric("Phone Presence", user_code_phone, input_data_phone)


def run_yes_presence_metric():
    user_code_yes = """
import re
def evaluate(answer):
    # Use word boundaries to detect 'Yes' as an independent word
    return 1.0 if re.search(r'\\bYes\\b', answer) else 0.0
"""
    input_data_yes = [
        {"answer": "Yes, I agree with your proposal."},
        {"answer": "Absolutely,Yes indeed!"},
        {"answer": "Yesman is not acceptable."},
        {"answer": "No, I don't."},
    ]
    run_metric("Standalone 'Yes' Presence", user_code_yes, input_data_yes)


def run_arithmetic_sum_metric():
    user_code_sum = """
def evaluate(a, b, answer):
    try:
        expected_sum = float(a) + float(b)
        # Allow a small floating-point tolerance
        return 1.0 if abs(expected_sum - float(answer)) < 1e-6 else 0.0
    except Exception:
        return 0.0
"""
    input_data_sum = [
        {"a": 3, "b": 4, "answer": "7"},
        {"a": 10, "b": 5, "answer": "15"},
        {"a": 2.5, "b": 3.1, "answer": "5.6"},
        {"a": 2, "b": 2, "answer": "5"},
    ]
    run_metric("Arithmetic Sum", user_code_sum, input_data_sum)


def run_pydantic_response_validation_metric():
    user_code_pydantic = """
from pydantic import BaseModel, ValidationError
import json
class ResponseModel(BaseModel):
    name: str
    email: str
    phone: str

def evaluate(answer):
    try:
        data = json.loads(answer)
        ResponseModel(**data)
        return 1.0
    except (ValidationError, Exception):
        return 0.0
"""
    input_data_pydantic = [
        {"answer": '{"name": "Alice", "email": "alice@example.com", "phone": "123-456-7890"}'},
        {"answer": '{"name": "Bob", "email": "bob_at_example.com", "phone": "1234567890"}'},
        {"answer": '{"name": "Charlie", "email": "charlie@example.com"}'},
    ]
    run_metric("Pydantic Response Validation", user_code_pydantic, input_data_pydantic)


def main():
    run_exact_match_metric()
    run_email_presence_metric()
    run_phone_presence_metric()
    run_yes_presence_metric()
    run_arithmetic_sum_metric()
    run_pydantic_response_validation_metric()


if __name__ == "__main__":
    main()
