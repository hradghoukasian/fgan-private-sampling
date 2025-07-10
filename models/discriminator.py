try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("Make sure PyTorch is installed. Run `pip install torch`. Error: {}".format(e))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


