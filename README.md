# GinnExtension: Neural Implicit Field Learning with Geometric Constraints

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

GinnExtension is a research framework for learning neural implicit representations of 2D scalar fields using geometric and physical constraints. The framework implements Physics-Informed Neural Networks (PINNs) with a flexible constraint system, enabling the learning of Signed Distance Functions (SDFs) and other implicit field representations through differentiable optimization.

### Key Features

- **Neural Implicit Representations**: Learn continuous field representations using neural networks
- **Constraint-Based Training**: Flexible framework for incorporating geometric, physical, and custom constraints
- **Multiple Architectures**: Support for both MLP-based and attention-based neural field models
- **Interactive Visualization**: Real-time field visualization and training progress monitoring
- **Experiment Tracking**: Integration with Weights & Biases for reproducible research

## Technical Approach

### Architecture

The framework provides two main neural architectures for field learning:

1. **FieldMLP**: Multi-layer perceptron with smooth activations optimized for gradient-based learning
2. **FieldAttention**: Attention-based architecture using learnable context vectors for enhanced expressiveness

### Constraint Framework

The constraint system allows researchers to incorporate domain knowledge through differentiable loss functions:

- **Area Constraints**: Control the area of learned shapes
- **Total Variation**: Regularize field smoothness
- **Continuity Constraints**: Enforce geometric properties
- **Symmetry Constraints**: Preserve geometric symmetries
- **Custom Constraints**: Extensible framework for domain-specific requirements

### Mathematical Foundation

Given a 2D coordinate space, the framework learns a function f: ℝ² → ℝ that represents implicit fields. The training objective combines multiple constraint terms:

```
L_total = Σᵢ wᵢ L_constraint_i(f(x), x)
```

where each constraint L_constraint_i can encode geometric, physical, or application-specific requirements.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Dependencies

```bash
pip install torch torchvision pytorch-lightning
pip install numpy matplotlib plotly
pip install wandb  # For experiment tracking
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/NavRjn/GinnExtension.git
cd GinnExtension
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create this file with above dependencies
```

3. Run the example notebook:
```bash
jupyter notebook notebook.ipynb
```

## Usage

### Basic Example

```python
from models import FieldMLP, FieldAttention
from utils import SDF, Constraints, ConstraintTrainer
import torch
import pytorch_lightning as pl

# Define model
model = FieldMLP(in_dim=2, hidden=128, depth=4, out_dim=1)

# Create SDF wrapper
sdf = SDF(model=model, xy_lims=(-1, 1, -1, 1))

# Define constraints
constraints = Constraints()

@constraints.add(weight=1.0, target_area=0.2)
def area_constraint(field, coords, target_area):
    p = torch.sigmoid(field)
    area_est = p.mean()
    return (area_est - target_area) ** 2

# Setup trainer
trainer_module = ConstraintTrainer(
    sdf=sdf,
    constraints=constraints,
    lr=1e-3,
    n_points=1024,
    domain=[-1, 1]
)

# Train model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(trainer_module)
```

### Custom Constraints

Define custom constraints by decorating functions with the constraint system:

```python
constraints = Constraints()

@constraints.add(weight=0.1)
def smoothness_constraint(field, coords):
    # Compute total variation
    grad = torch.autograd.grad(
        field, coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True, retain_graph=True
    )[0]
    return grad.pow(2).sum(dim=-1).sqrt().mean()

@constraints.add(weight=2.0, target_value=0.0)
def boundary_constraint(field, coords, target_value):
    # Enforce boundary conditions
    boundary_mask = (coords.norm(dim=-1) > 0.9)
    boundary_field = field[boundary_mask]
    return (boundary_field - target_value).pow(2).mean()
```

## Experimental Setup

### Reproducibility

For reproducible experiments, set seeds consistently:

```python
import pytorch_lightning as pl
pl.seed_everything(42)

trainer_module = ConstraintTrainer(
    sdf=sdf,
    constraints=constraints,
    seed=42  # Additional seed setting
)
```

### Experiment Tracking

Integration with Weights & Biases for experiment management:

```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="ginn-experiments",
    name="experiment-name",
    tags=["neural-fields", "constraints"]
)

trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=100
)
```

### Hyperparameter Tuning

Key hyperparameters for optimization:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Learning Rate | Optimizer learning rate | 1e-4 to 1e-2 |
| N Points | Collocation points per batch | 128 to 2048 |
| Hidden Size | Network width | 64 to 512 |
| Depth | Network depth | 3 to 8 |
| Constraint Weights | Relative importance of constraints | 0.01 to 10.0 |

## API Reference

### Core Classes

#### `SDF`
Signed Distance Function wrapper for neural field visualization and evaluation.

**Parameters:**
- `model`: Neural network model
- `xy_lims`: Spatial domain bounds (x_min, x_max, y_min, y_max)
- `device`: Computation device

**Methods:**
- `plot_field(plotly=True, layers=100)`: Visualize the learned field
- `update(model=None)`: Update field values from model

#### `ConstraintTrainer`
PyTorch Lightning module for constraint-based training.

**Parameters:**
- `sdf`: SDF instance
- `constraints`: Constraints object
- `lr`: Learning rate
- `n_points`: Number of collocation points
- `domain`: Spatial domain for sampling

#### `Constraints`
Constraint management system.

**Methods:**
- `add(weight=1.0, **params)`: Decorator for adding constraints
- `get_loss(field, coords)`: Compute total constraint loss

### Model Architectures

#### `FieldMLP`
Multi-layer perceptron for field learning.

**Parameters:**
- `in_dim`: Input dimension (default: 2)
- `hidden`: Hidden layer size (default: 128)
- `depth`: Number of layers (default: 4)
- `out_dim`: Output dimension (default: 1)

#### `FieldAttention`
Attention-based field learning model.

**Parameters:**
- `in_dim`: Input dimension (default: 2)
- `hidden`: Hidden layer size (default: 128)
- `context_dim`: Context vector dimension (default: 64)
- `num_context`: Number of context vectors (default: 16)

## Extensions and Customization

### Adding New Architectures

To implement custom neural architectures:

```python
import torch.nn as nn

class CustomFieldModel(nn.Module):
    def __init__(self, in_dim=2, out_dim=1):
        super().__init__()
        # Define your architecture
        self.layers = nn.Sequential(
            # Your custom layers
        )
    
    def forward(self, x):
        # x: (batch_size, in_dim) coordinates
        # return: (batch_size, out_dim) field values
        return self.layers(x)
```

### Custom Constraint Types

Implement domain-specific constraints:

```python
@constraints.add(weight=1.0, physics_param=0.1)
def physics_constraint(field, coords, physics_param):
    """
    Example: Implement physics-based constraints
    such as Poisson equations, heat equations, etc.
    """
    # Compute required derivatives
    grad = torch.autograd.grad(field, coords, 
                              create_graph=True)[0]
    laplacian = torch.autograd.grad(grad.sum(), coords, 
                                   create_graph=True)[0]
    
    # Define physics-based loss
    physics_loss = (laplacian + physics_param * field).pow(2).mean()
    return physics_loss
```

### 3D Extension

Extend to 3D fields by modifying input dimensions:

```python
model_3d = FieldMLP(in_dim=3, hidden=256, depth=6, out_dim=1)
sdf_3d = SDF(model=model_3d, xyz_lims=(-1, 1, -1, 1, -1, 1))
```

## Performance Optimization

### GPU Acceleration

- Use mixed precision training: `trainer = pl.Trainer(precision="bf16-mixed")`
- Batch collocation points efficiently
- Use gradient accumulation for large point sets

### Memory Optimization

- Adjust `n_points` based on GPU memory
- Use gradient checkpointing for deep networks
- Consider distributed training for large-scale experiments

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ginnextension2024,
  title={GinnExtension: Neural Implicit Field Learning with Geometric Constraints},
  author={[Author Names]},
  year={2024},
  url={https://github.com/NavRjn/GinnExtension}
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

### Development Setup

```bash
git clone https://github.com/NavRjn/GinnExtension.git
cd GinnExtension
pip install -e .[dev]  # Install in development mode
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for public functions
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- PyTorch Lightning for training infrastructure
- Weights & Biases for experiment tracking
- Research community working on neural implicit representations

## Related Work

- [Neural Implicit Representations](https://arxiv.org/abs/2106.05228)
- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561)
- [Signed Distance Functions](https://en.wikipedia.org/wiki/Signed_distance_function)

## Contact

For questions or collaboration opportunities, please open an issue or contact [maintainer email].

---

**Disclaimer**: This is research software. While we strive for correctness and usability, please validate results for your specific use case.