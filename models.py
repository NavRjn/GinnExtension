import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, depth=4, out_dim=1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden ] *(depth -1) + [out_dim]
        for i in range(len(dims ) -1):
            layers += [nn.Linear(dims[i], dims[ i +1])]
            if i < len(dims ) -2:
                layers += [nn.SiLU()]     # smooth activation is helpful for gradients
        self.net = nn.Sequential(*layers)

    def forward(self, x):   # x: (B, 2) points in [-1,1]^2
        return self.net(x)  # (B, 1)


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
