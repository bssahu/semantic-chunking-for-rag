# Contributing to Semantic Document Chunking

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   pytest
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add your meaningful commit message here"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semantic_chunking.git
   cd semantic_chunking
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

5. Run the setup script:
   ```bash
   bash setup.sh
   ```

## Code Style

We follow PEP 8 guidelines for Python code. Please ensure your code adheres to these standards.

You can use tools like `flake8` and `black` to check and format your code:

```bash
flake8 .
black .
```

## Testing

Please write tests for any new features or bug fixes. We use pytest for testing:

```bash
pytest
```

## Documentation

Please update the documentation when adding or modifying features. This includes:

- Docstrings for functions and classes
- README.md updates for user-facing changes
- Architecture documentation for significant changes

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation as necessary
3. Include a clear description of the changes in your PR
4. Link any related issues in your PR description

## Feature Requests and Bug Reports

Please use the GitHub issue tracker to submit feature requests and bug reports. Include as much detail as possible to help us understand your request or the bug you're experiencing.

## Questions

If you have any questions about contributing, please open an issue with your question.

Thank you for contributing to Semantic Document Chunking! 