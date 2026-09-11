import torch
import torch.nn.functional as F
from torch import nn

from abupt.nn.positional import apply_rope, merge_heads, split_heads


class AnchorAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        num_anchors: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_anchors is None or num_anchors == x.shape[1]:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            k_freqs = freqs
        else:
            anchors, queries = x[:, :num_anchors], x[:, num_anchors:]
            q, k, v = self.qkv(anchors).chunk(3, dim=-1)
            q_weight, q_bias = self.qkv.weight[: self.dim], self.qkv.bias[: self.dim]
            q = torch.cat([q, F.linear(queries, q_weight, q_bias)], dim=1)
            k_freqs = freqs[:, :num_anchors]
        q, k = split_heads(q, self.num_heads), split_heads(k, self.num_heads)
        q, k = apply_rope(q, freqs), apply_rope(k, k_freqs)
        v = split_heads(v, self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(merge_heads(out))
