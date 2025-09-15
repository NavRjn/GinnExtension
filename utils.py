from symbol import decorator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from math import sqrt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import wandb

################# File setup  ####################
N = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pio.renderers.default = "browser"



# ------------------------------------------------- GENERAL CLASSES -------------------------------------------------- #

# ---------- Signed Distance Functions ---------- #
class SDF:
    """
        Signed Distance Field (SDF) representation.

        This class can store a scalar field either from:
        - an analytic function (`fun`)
        - a model (e.g., neural network) (`model`)
        - or directly provided field values (`values`).

        The field is stored on a grid defined by either:
        - limits `xy_lims` (automatic meshgrid generation), or
        - precomputed `grid_x`, `grid_y`.

        Attributes
        ----------
        grid_x, grid_y : torch.Tensor
            Meshgrid of x and y coordinates.
        values : torch.Tensor
            Computed scalar field values over the grid.
        model : callable or None
            Function/model that evaluates the field given input coordinates.
        device : str
            Torch device ("cpu" or "cuda").
        fig : plotly.graph_objs.Figure or None
            Cached plotly figure (if `plotly=True` was used in `plot_field`).
        """
    def __init__(self, fun=None, model=None, grid_x=None, grid_y=None, values=None, xy_lims=None, device="cpu"):
        if xy_lims is not None:
            self.grid_y, self.grid_x = torch.meshgrid(torch.linspace(xy_lims[0], xy_lims[1], N),
                                                      torch.linspace(xy_lims[2], xy_lims[3], N), indexing="ij")
        else:
            assert grid_x is not None and grid_y is not None
            self.grid_x, self.grid_y = grid_x, grid_y

        self.values, self.model, self.values = None, None, None
        self.update(fun, model, values, device_=device)

        self.device = device

        self.fig = None

    def plot_field(self, plotly=False, layers=100, preprocess=None, domain=None, newN=None):
        values = None
        if domain is None:
            grid_x, grid_y = self.grid_x, self.grid_y
            values = self.values if (preprocess is None) else preprocess(self.values)
        else:
            assert self.model is not None
            model = self.model.to(self.device)
            N_ = N if newN is None else newN
            grid_y, grid_x = torch.meshgrid(torch.linspace(domain[0], domain[1], N_),
                                            torch.linspace(domain[0], domain[1], N_))
            coords = torch.stack([self.grid_x.reshape(-1), self.grid_y.reshape(-1)], dim=-1).to(self.device)
            with torch.no_grad():
                values = self.model(coords).reshape(N_, N_)
        if not plotly:
            plt.contourf(grid_x.cpu().numpy(), grid_y.cpu().numpy(), values.cpu(), levels=layers)
            plt.colorbar()
            plt.contour(grid_x.cpu(), grid_y.cpu(), values.reshape(grid_x.shape).cpu(), levels=[0.0],
                        colors='black')  # plot zero level
            plt.show()
            return

        self.fig = go.Figure(
            data=go.Contour(
                x=grid_x[0, :].cpu().numpy(),  # X-axis from meshgrid
                y=grid_y[:, 0].cpu().numpy(),  # Y-axis from meshgrid
                z=values.cpu().numpy(),  # field values
                colorscale="Viridis",  # color map
                contours=dict(showlines=False, coloring="fill"),
                ncontours=layers,
                showscale=True
            )
        )

        # Add zero-level contour in black (like your plt.contour)
        self.fig.add_trace(
            go.Contour(
                x=grid_x[0, :].cpu().numpy(),
                y=grid_y[:, 0].cpu().numpy(),
                z=values.cpu().numpy(),
                contours=dict(start=-0.1, end=0.1, size=1, coloring="none"),
                line=dict(color="black", width=2),
                showscale=False
            )
        )

        self.fig.update_layout(
            title="Contour Plot",
            xaxis=dict(
                scaleanchor="y",  # Lock x and y axes together
            ),
            yaxis=dict(
                scaleanchor="x",  # Lock y and x axes together (optional, can be removed if set on xaxis)
            ),
            autosize=True
        )

        self.fig.show()

    def update(self, fun=None, model=None, values=None, device_="cuda"):
        if fun is None and model is None:
            assert values is not None
            self.values = values
            self.model = None
        else:
            if model is None:
                assert fun is not None
                model = lambda points: torch.asarray([fun(point) for point in points])
            self.model = model
            coords = torch.stack(
                [self.grid_x.reshape(-1), self.grid_y.reshape(-1)], dim=-1
            ).to(device_)
            with torch.no_grad():
                self.values = model(coords).reshape(N, N)


class CircleSDF(SDF):
    def __init__(self, x0=(0, 0), r=1, xy_lims=(-1, 1, -1, 1)):
        model = lambda points: torch.asarray([abs(sqrt((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2) - r) for x in points])
        super().__init__(model=model, xy_lims=xy_lims)

# TODO: Untested. LLM generated.
class TuringSDF(SDF):
    def __init__(self, model=None, xy_lims=(-1, 1, -1, 1),
                 Du=2.4e-5, Dv=1.2e-5, alpha=0.028, beta=0.057, device="cpu"):
        """
        A Turing pattern field (two channels: u, v).
        model: nn.Module taking coords [B,2] → [B,2] for (u,v).
        """
        super().__init__(model=model, xy_lims=xy_lims, device=device)
        self.Du, self.Dv = Du, Dv
        self.alpha, self.beta = alpha, beta

    def residuals(self, coords):
        """Compute PDE residuals at coords (for loss)."""
        coords.requires_grad_(True)
        uv = self.model(coords)              # shape [B,2], columns = (u,v)
        u, v = uv[:,0:1], uv[:,1:2]

        # Laplacians via autograd (vectorized trick: grad-of-grad)
        grads = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        lap_u = torch.autograd.grad(grads[:,0].sum(), coords, create_graph=True)[0][:,0] + \
                torch.autograd.grad(grads[:,1].sum(), coords, create_graph=True)[0][:,1]
        grads = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        lap_v = torch.autograd.grad(grads[:,0].sum(), coords, create_graph=True)[0][:,0] + \
                torch.autograd.grad(grads[:,1].sum(), coords, create_graph=True)[0][:,1]

        Ru = self.Du*lap_u.unsqueeze(-1) - u*v**2 + self.alpha*(1-u)
        Rv = self.Dv*lap_v.unsqueeze(-1) + u*v**2 - (self.alpha+self.beta)*v
        return Ru, Rv, u, v

    def update(self, fun=None, model=None, values=None, device_="cuda"):
        # Override: update self.values with just one of the fields (say u) for visualization
        if model is not None:
            self.model = model
            coords = torch.stack(
                [self.grid_x.reshape(-1), self.grid_y.reshape(-1)], dim=-1
            ).to(device_)
            with torch.no_grad():
                uv = model(coords)
                self.values = uv[:,0].reshape(N, N)  # take u field
        elif values is not None:
            self.values = values



# ---------- Constraints ---------- #

class Constraints:
    def __init__(self):
        self.constraints = {}

    def add(self, weight=1.0, needs_residuals=False, **params):
        def decorator(func):
            name = func.__name__
            self.constraints[name] = {
                "func": func,
                "weight": weight,
                "params": params,
                "needs_residuals": needs_residuals
            }
            return func

        return decorator

    def get_loss(self, field, coords, residuals=None, loss_type=None):
        # Pass residuals if constraint asked for them
        def call_constraint(data, name, d):
            args = {
                "field": field,
                "coords": coords,
                **d["params"]
            }
            if d["needs_residuals"]:
                args["residuals"] = residuals
            return d["func"](**args)

        if loss_type == "all":
            all_losses = {}
            for name, d in self.constraints.items():
                all_losses[name] = call_constraint(field, name, d)
            return all_losses
        elif loss_type is not None:
            return call_constraint(field, loss_type, self.constraints[loss_type])
        else:
            total_loss = 0.0
            for name, d in self.constraints.items():
                total_loss += d["weight"] * call_constraint(field, name, d)
            return total_loss


example_constraints = Constraints()

@example_constraints.add(4.0, target_area=0.2)
def area(self, field, target_area):
    p = torch.sigmoid(field)
    area_est = p.mean()
    return (area_est - target_area) ** 2
@example_constraints.add(0.1)
def tv(self, field, coords):
    # TV on the field via gradient magnitude ||∇f||; needs grads w.r.t. coords
    grad = torch.autograd.grad(
        field, coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]  # shape (B,2)
    tv = (grad.pow(2).sum(dim=-1) + 1e-8).sqrt().mean()
    return tv
@example_constraints.add(10.0, thresh=0.5)
def cont(self, coords, field, thresh):
    # Example: penalize predictions outside a disk of radius 1 (just a demo)
    # Encourage "inside" probs > thresh where r<=1, and < thresh outside
    r = coords.norm(dim=-1, keepdim=True)
    target = (r <= 1.0).float()
    p = torch.sigmoid(field)
    return ((p - (thresh * 0 + target)).abs()).mean()
@example_constraints.add(0.2)
def cent(self, field):
    return field.mean() ** 2
@example_constraints.add(0.0)
def symm(self, field, coords):
    return 0.0


# ---------- Lightning trainer ---------- #

class ConstraintTrainer(pl.LightningModule):
    def __init__(self, sdf, constraints, lr=1e-3, n_points=1024, domain=[-1, 1], seed=None):
        super().__init__()
        self.sdf = sdf
        self.constraints = constraints
        self.model = sdf.model
        self.lr = lr
        self.n_points = n_points
        self.register_buffer('domain_min', torch.tensor([domain[0], domain[0]]), persistent=False)
        self.register_buffer('domain_max', torch.tensor([domain[1], domain[1]]), persistent=False)
        if seed is not None:
            pl.seed_everything(seed)

        self.all_frames = []

    def sample_coords(self, n):
        # Uniform in the box
        u = torch.rand(n, 2, device=self.device)
        return self.domain_min + (self.domain_max - self.domain_min) * u

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=1)

    # --------------------- HOOKS ---------------------------------
    def training_step(self, batch, batch_idx):
        # No dataloader needed; we draw fresh collocation points each step
        coords = self.sample_coords(self.n_points).requires_grad_()
        field = self.model(coords)

        # If sdf has residuals, compute once
        residuals = self.sdf.residuals(coords) if hasattr(self.sdf, "residuals") else None

        # loss = abs(field.mean()) # Dummy loss
        cl = self.constraints.get_loss(field=field, coords=coords, residuals=residuals, loss_type="all")
        loss = self.constraints.get_loss(field, coords)

        self.log_dict(
            {"loss": loss, **cl},
            prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def on_train_start(self) -> None:
        self.sdf.update(model=self.model, device_=self.device)
        self.sdf.plot_field(True, 50)

    def on_train_epoch_end(self):
        self.sdf.update(model=self.model, device_=self.device)
        # self.sdf.plot_field(True)
        self.all_frames.append(self.sdf.values.cpu().numpy())

    def on_train_end(self):
        # Build plotly animation
        frames = [go.Frame(
            data=[go.Contour(z=z, x=self.sdf.grid_x[0, :].cpu().numpy(),
                             y=self.sdf.grid_y[:, 0].cpu().numpy())],
            name=f"epoch{k}"
        ) for k, z in enumerate(self.all_frames)]

        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )

        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 200, "redraw": True},
                                     "fromcurrent": True}], "label": "▶ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}], "label": "⏸ Pause", "method": "animate"}
                ]
            }]
        )
        fig.show()


# -------------------------------------------------- TORCH HELPERS --------------------------------------------------- #

class DummyDataset(Dataset):
    def __len__(self): return 10

    def __getitem__(self, idx): return torch.tensor(0.0)

