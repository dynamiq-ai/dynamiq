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

3. Create a virtual environment and install project requirements:
   ```sh
   make install
   ```

4. **Create a Branch:** Create a new branch for your work, named `<type>/<short-description>` with the type your commit will have:
   ```sh
   git checkout -b fix/qdrant-filter-search
   ```

5. **Make Changes:** Make your changes in your branch.

6. **Write Tests:** If applicable, write tests for your changes.

7. **Commit Changes:** Commit your changes with a message in [Conventional Commits](https://www.conventionalcommits.org/) form. The hook that `make install` sets up rejects any other, and CI checks the pull request title the same way:
   ```sh
   git commit -m "fix(retrievers): keep the filter on a Qdrant search"
   ```

8. Make sure you run the checks and the tests that CI runs before pushing your changes. `make test` would also run `tests/integration_with_creds`, which calls real providers with keys from `.env`:
   ```sh
   make prepare
   uv run pytest tests --ignore=tests/integration_with_creds -m "not smoke" -n auto --dist worksteal
   ```

9. **Push to Fork:** Push your changes to your forked repository:
   ```sh
   git push origin fix/qdrant-filter-search
   ```

10. **Open a Pull Request:** Open a pull request from your fork to the main repository, and fill in the template: what changes and why, whether saved workflows still load, what you ran, and any related issues.

## Code Style

Please follow the code style used in the project. We use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Ensure your code passes all linting checks before submitting a pull request. [AGENTS.md](AGENTS.md) lists the project's conventions, the checks CI runs and what to look for when reviewing.

## Review Process

All pull requests will be reviewed by our maintainers. We aim to provide feedback within a few days. Please be responsive to any feedback or questions and be ready to make changes if necessary.
