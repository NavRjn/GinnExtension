# Contributing to GinnExtension

We welcome contributions to GinnExtension! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Provide detailed reproduction steps
4. Include system information (OS, Python version, PyTorch version)
5. Add relevant error messages and stack traces

### Submitting Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/GinnExtension.git
   cd GinnExtension
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   python -m pytest tests/
   python -m flake8 .
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use a descriptive title
   - Reference related issues
   - Provide detailed description of changes
   - Include test results

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```bash
pip install pytest pytest-cov flake8 black isort pre-commit
```

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black .`
- Use `isort` for import sorting: `isort .`
- Run `flake8` for linting: `flake8 .`

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Tuple
import torch

def train_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3
) -> Tuple[torch.nn.Module, float]:
    """Train a neural network model."""
    # Implementation
    return model, final_loss
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings where helpful

```python
def constraint_function(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute constraint loss for a given field.
    
    Args:
        field: Neural field values of shape (N, 1)
        coords: Coordinate points of shape (N, 2)
        
    Returns:
        Scalar constraint loss
        
    Example:
        >>> field = torch.randn(100, 1)
        >>> coords = torch.randn(100, 2)
        >>> loss = constraint_function(field, coords)
    """
    return field.pow(2).mean()
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_constraint_loss_computation`
- Test both successful cases and error conditions
- Use fixtures for common test data

```python
import pytest
import torch
from models import FieldMLP

class TestFieldMLP:
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = FieldMLP(in_dim=2, hidden=64, depth=3, out_dim=1)
        x = torch.randn(10, 2)
        output = model(x)
        assert output.shape == (10, 1)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = FieldMLP(in_dim=2, hidden=64, depth=3, out_dim=1)
        x = torch.randn(10, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_models.py

# Run specific test
python -m pytest tests/test_models.py::TestFieldMLP::test_forward_pass_shape
```

## Documentation

### Building Documentation

If we add documentation generation:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

### Updating README

When adding new features:
1. Update the relevant sections in README.md
2. Add code examples for new functionality
3. Update the API reference section
4. Add new dependencies to requirements.txt

## Research Contributions

### New Model Architectures

When contributing new neural architectures:

1. **Implement the model** in `models.py`
2. **Add comprehensive tests** in `tests/test_models.py`
3. **Update documentation** with architecture details
4. **Provide example usage** in notebooks or examples
5. **Include performance benchmarks** if applicable

### New Constraint Types

When adding new constraints:

1. **Implement constraint function** with proper signature
2. **Add to example constraints** in `utils.py`
3. **Write unit tests** for the constraint
4. **Document mathematical formulation**
5. **Provide usage examples**

### Research Papers

If implementing methods from research papers:

1. **Add paper citation** to README.md
2. **Include mathematical details** in documentation
3. **Provide comparative experiments** if possible
4. **Credit original authors** appropriately

## Release Process

### Version Numbering

We follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Creating a Release

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release branch
4. Tag the release: `git tag v1.0.0`
5. Push tags: `git push --tags`

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be:
- Listed in the contributors section
- Acknowledged in release notes
- Co-authors on academic publications (for significant contributions)

Thank you for contributing to GinnExtension!