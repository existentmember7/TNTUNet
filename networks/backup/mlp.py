import torch
import torch.nn.functional as F
from torch import nn, einsum

class MLP(nn.Sequential):

    def __init__(
        self,
        feature_dim,
        num_classes
    ):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.patch_dim, self.num_classes)
        )