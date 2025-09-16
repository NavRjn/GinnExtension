import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, depth=4, out_dim=1, init="xavier", use_periodic=False):
        super().__init__()
        self.use_periodic = use_periodic

        # input dimension: 2 (raw coords) or 4 (periodic encoding)
        eff_in_dim = 4 if use_periodic else in_dim

        layers = []
        dims = [eff_in_dim] + [hidden] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())  # smooth activations

        self.net = nn.Sequential(*layers)

        # apply initialization
        if init == "xavier":
            self._init_xavier()
        elif init == "kaiming":
            self._init_kaiming()

    def forward(self, x):
        if self.use_periodic:
            x = self._periodic_encoding(x)
        return self.net(x)

    @staticmethod
    def _periodic_encoding(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,2) with values in [0,1]
        returns: (B,4) with sine/cosine encoding
        """
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    # --- init schemes ---
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


class FieldAttention(nn.Module):
    def __init__(self, in_dim=2, hidden=128, context_dim=64, num_context=16, out_dim=1):
        super().__init__()
        self.context = nn.Parameter(torch.randn(num_context, context_dim))  # (N_ctx, C)
        self.query_proj = nn.Linear(in_dim, context_dim)
        self.key_proj = nn.Linear(context_dim, context_dim)
        self.value_proj = nn.Linear(context_dim, hidden)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):  # x: (B, 2)
        q = self.query_proj(x)  # (B, C)
        k = self.key_proj(self.context)  # (N_ctx, C)
        v = self.value_proj(self.context)  # (N_ctx, H)

        attn = F.softmax(q @ k.T / k.shape[-1] ** 0.5, dim=-1)  # (B, N_ctx)
        ctx = attn @ v  # (B, H)

        return self.mlp(ctx)  # (B, 1)


# ------------------------------------------------ Turing models ---------------------------------------------------- #
# Untested: LLM Generated
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, mapping_size) * scale, requires_grad=False)  # fixed
    def forward(self, x):  # x: [B, N, 2] or [N,2]
        x_proj = (2 * torch.pi * x) @ self.B  # [N, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SmallDecoder(nn.Module):
    def __init__(self, latent_dim=0, ff_dim=64, hidden=128, layers=4):
        super().__init__()
        input_dim = ff_dim * 2 + latent_dim  # if ff created with mapping_size=ff_dim
        seq = []
        seq.append(nn.Linear(input_dim, hidden))
        seq.append(nn.ReLU())
        for _ in range(layers-2):
            seq.append(nn.Linear(hidden, hidden))
            seq.append(nn.ReLU())
        seq.append(nn.Linear(hidden, 2))  # outputs u, v
        self.net = nn.Sequential(*seq)

    def forward(self, coords, z=None):
        # coords: [N,2] or [B,N,2] depending on batching
        ff = self.fourier(coords)  # attach module instance -- # PROBLEM!!!: TODO: debug.
        if z is not None:
            # broadcast z to per-point
            if z.dim()==2 and ff.dim()==2:
                z_rep = z.repeat(ff.shape[0], 1)
            else:
                # handle batch dims as needed
                pass
            inp = torch.cat([ff, z_rep], dim=-1)
        else:
            inp = ff
        return self.net(inp)  # shape [N,2] with columns u,v
