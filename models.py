import torch.nn as nn



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
