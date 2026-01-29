# Contributing Guidelines

Thank you for your interest in contributing to this template! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install dependencies: `make install`
4. Install pre-commit hooks: `make pre-commit-install`

## Code Style

- **Formatting**: Code is automatically formatted with `ruff` and `black` on save (VS Code/Cursor)
- **Linting**: Run `make lint` to check code quality
- **Pre-commit**: All commits are automatically checked by pre-commit hooks

## Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Ensure all tests pass: `make test`
4. Run linting: `make lint`
5. Commit your changes (pre-commit will run automatically)
6. Push to your fork and create a pull request

## Commit Messages

Use clear, descriptive commit messages following conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests

## Pull Requests

- Keep PRs focused and small when possible
- Ensure all pre-commit checks pass
- Update documentation if needed
- Add tests for new features

## Questions?

Feel free to open an issue for questions or discussions.
