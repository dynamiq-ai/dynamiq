# Contributing to Dynamiq

Thank you for your interest in contributing to Dynamiq! We welcome contributions in various forms, including bug reports, feature requests, documentation improvements, and code contributions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on our GitHub repository. Provide as much information as possible, including:

- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected and actual behavior.
- Screenshots, logs, or code snippets, if applicable.

### Suggesting Features

If you have an idea for a new feature or an improvement, please open an issue with the following details:

- A clear and descriptive title.
- A detailed description of the feature.
- Any relevant use cases or examples.

### Improving Documentation

Good documentation is key to a successful project. If you find areas in our documentation that need improvement, feel free to submit a pull request. Here are some ways you can help:

- Fix typos or grammatical errors.
- Clarify confusing sections.
- Add missing information.

### Contributing Code

1. **Fork the Repository:** Fork the [repository](https://github.com/dynamiq-ai/dynamiq) to your own GitHub account.

2. **Clone the Fork:** Clone your fork to your local machine:
   ```sh
   git clone https://github.com/YOUR-USERNAME/dynamiq
   ```

3. Create a virtaul environment and install project requirements:
   ```sh
   make install
   ```

4. **Create a Branch:** Create a new branch for your work:
   ```sh
   git checkout -b feature-name
   ```

5. **Make Changes:** Make your changes in your branch.

6. **Write Tests:** If applicable, write tests for your changes.

7. **Commit Changes:** Commit your changes with a descriptive commit message:
   ```sh
   git commit -m "Description of the feature or fix"
   ```

8. Make sure you run tests before pushing your changes:
   ```sh
   make prepare test
   ```

9. **Push to Fork:** Push your changes to your forked repository:
   ```sh
   git push origin feature-name
   ```

10. **Open a Pull Request:** Open a pull request from your fork to the main repository. Include a detailed description of your changes and any related issues.

## Code Style

Please follow the code style used in the project. We use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Ensure your code passes all linting checks before submitting a pull request.

## Review Process

All pull requests will be reviewed by our maintainers. We aim to provide feedback within a few days. Please be responsive to any feedback or questions and be ready to make changes if necessary.
