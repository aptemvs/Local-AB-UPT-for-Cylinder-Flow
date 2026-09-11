import torch
from torch import nn

from abupt.nn.attention import AnchorAttention


class Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AnchorAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        num_anchors: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs=freqs, num_anchors=num_anchors, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x
