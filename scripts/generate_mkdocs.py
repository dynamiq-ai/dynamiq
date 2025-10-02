import glob
import os


def create_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_python_files(directory):
    """Get all Python files in the specified directory, excluding __init__.py files and tests."""
    py_files = glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)

    def is_not_test_file(path: str) -> bool:
        # Exclude anything under dynamiq/tests/** to avoid documenting test modules
        parts = path.split(os.sep)
        # Find a segment named 'tests' after the top-level package directory
        return "tests" not in parts[1:]

    return [file for file in py_files if "__init__.py" not in file and is_not_test_file(file)]


def generate_documentation_file(file_path):
    """Generate a documentation file for the given Python file."""
    file_path_splits = file_path.split("/")

    # Create the documentation folder
    docs_folder = os.path.join("mkdocs", *file_path_splits[:-1])
    create_directory(docs_folder)

    # Generate the new documentation file name
    file_name = file_path_splits[-1].replace(".py", ".md")
    new_docs_file = os.path.join(docs_folder, file_name)

    # Generate the file content
    file_content = ":::" + ".".join(file_path_splits).replace(".py", "\n")

    # Write the content to the new documentation file
    with open(new_docs_file, "w") as f:
        f.write(file_content)


def main():
    """Main function to generate documentation for Python files."""
    source_directory = "dynamiq"
    python_files = get_python_files(source_directory)
    for file in python_files:
        generate_documentation_file(file)

    print(f"Documentation generated for {len(python_files)} Python files.")


if __name__ == "__main__":
    main()
