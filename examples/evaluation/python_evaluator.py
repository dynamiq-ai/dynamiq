from dynamiq.evaluations import PythonEvaluator


def main():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""

    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Barcelona"},
    ]

    evaluator = PythonEvaluator(code=user_code)
    scores = evaluator.run(input_data_list=input_data_list)

    for idx, score in enumerate(scores, start=0):
        print(
            f"Pair {idx}: Answer: {input_data_list[idx]['answer']}, "
            f"Expected: {input_data_list[idx]['expected']}, Score: {score}"
        )


if __name__ == "__main__":
    main()
