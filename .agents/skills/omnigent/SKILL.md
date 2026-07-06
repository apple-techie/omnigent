```markdown
# omnigent Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `omnigent` Python codebase. You will learn about file organization, import/export styles, commit message formatting, and how to write and locate tests. These guidelines ensure consistency and maintainability across the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all file and module names.
  - Example: `data_processor.py`, `user_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_data
    from ..models import User
    ```

### Export Style
- Use **named exports** by explicitly listing public objects in `__all__`.
  - Example:
    ```python
    __all__ = ["parse_data", "User"]
    ```

### Commit Messages
- Follow the **conventional commit** pattern.
- Use the `fix` prefix for bug fixes.
- Keep commit messages concise but descriptive (average length: ~95 characters).
  - Example:
    ```
    fix: correct data parsing logic in data_processor.py to handle empty inputs
    ```

## Workflows

### Code Contribution
**Trigger:** When adding new features or fixing bugs  
**Command:** `/contribute`

1. Create a new branch for your changes.
2. Follow coding conventions for file naming, imports, and exports.
3. Write or update tests as needed (see Testing Patterns).
4. Commit changes using the conventional commit format.
5. Open a pull request for review.

### Running Tests
**Trigger:** When verifying code correctness  
**Command:** `/test`

1. Identify test files (pattern: `*.test.*`).
2. Use the project's preferred test runner (framework is unspecified; check project docs or use `pytest` as a default).
3. Run tests:
    ```bash
    pytest
    ```
   or
    ```bash
    python -m unittest discover
    ```

### Reviewing Commits
**Trigger:** When reviewing code history or preparing a release  
**Command:** `/review-commits`

1. Check commit messages for the `fix` prefix and adherence to conventional commit style.
2. Ensure each commit message is clear and descriptive.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `data_processor.test.py`).
- The testing framework is unspecified; likely candidates are `pytest` or `unittest`.
- Place test files alongside implementation files or in a dedicated `tests/` directory.
- Example test file:
    ```python
    # data_processor.test.py

    from .data_processor import parse_data

    def test_parse_data_empty():
        assert parse_data('') == []
    ```

## Commands
| Command         | Purpose                                                |
|-----------------|--------------------------------------------------------|
| /contribute     | Start a new code contribution workflow                 |
| /test           | Run all tests in the codebase                          |
| /review-commits | Review commit messages for conventional formatting      |
```
