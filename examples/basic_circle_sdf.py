#!/usr/bin/env python3
"""
Basic Example: Learning a Circle SDF with Neural Networks

This example demonstrates how to use GinnExtension to learn a signed distance
function for a circle using constraint-based training.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Import GinnExtension components
from models import FieldMLP, FieldAttention
from utils import SDF, Constraints, ConstraintTrainer


def main():
    """Run basic circle SDF learning example."""
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Define the neural network model
    print("Creating neural network model...")
    model = FieldMLP(
        in_dim=2,      # 2D coordinates (x, y)
        hidden=128,    # Hidden layer size
        depth=4,       # Number of layers
        out_dim=1      # Single output value
    ).to(device)
    
    # 2. Create SDF wrapper for visualization
    print("Setting up SDF wrapper...")
    sdf = SDF(
        model=model,
        xy_lims=(-1.5, 1.5, -1.5, 1.5),  # Domain bounds
        device=device
    )
    
    # 3. Define constraints for circle learning
    print("Defining constraints...")
    circle_constraints = Constraints()
    
    @circle_constraints.add(weight=1.0, target_area=0.3)
    def area_constraint(field, coords, target_area):
        """Constraint to control the area of the learned shape."""
        p = torch.sigmoid(field)  # Convert to probabilities
        area_est = p.mean()
        return (area_est - target_area) ** 2
    
    @circle_constraints.add(weight=0.1)
    def smoothness_constraint(field, coords):
        """Total variation constraint for smoothness."""
        grad = torch.autograd.grad(
            field, coords,
            grad_outputs=torch.ones_like(field),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        tv = (grad.pow(2).sum(dim=-1) + 1e-8).sqrt().mean()
        return tv
    
    @circle_constraints.add(weight=5.0, radius=0.5)
    def circle_constraint(field, coords, radius):
        """Constraint to encourage circle-like shape."""
        r = coords.norm(dim=-1, keepdim=True)
        # Inside circle should have positive values, outside negative
        target = (r <= radius).float() * 2 - 1  # Map to [-1, 1]
        return torch.nn.functional.mse_loss(torch.tanh(field), target)
    
    # 4. Setup training
    print("Setting up trainer...")
    trainer_module = ConstraintTrainer(
        sdf=sdf,
        constraints=circle_constraints,
        lr=1e-3,
        n_points=512,  # Number of collocation points per batch
        domain=[-1.5, 1.5],  # Sampling domain
        seed=42
    )
    
    # 5. Configure PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_progress_bar=True,
        # logger=WandbLogger(project="ginn-example", name="circle-sdf")  # Uncomment for W&B logging
    )
    
    # 6. Train the model
    print("Starting training...")
    trainer.fit(trainer_module)
    
    # 7. Visualize results
    print("Training completed! Visualizing results...")
    sdf.update(model=model, device_=device)
    sdf.plot_field(plotly=True, layers=50)
    
    print("Example completed successfully!")
    print("The learned field should approximate a circle SDF.")


if __name__ == "__main__":
    main()