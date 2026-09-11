import torch
import torch.nn.functional as F
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: float):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        dim_per_axis = (dim - dim % ndim) // ndim
        self.padding = dim % ndim + (dim_per_axis % 2) * ndim
        effective = (dim - self.padding) // ndim
        assert effective > 0, f"dim={dim} too small for ndim={ndim}"
        frequencies = torch.arange(0, effective, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (frequencies / effective))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=coords.device.type, enabled=False):
            angles = coords.float().unsqueeze(-1) * self.omega
            embedding = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(start_dim=-2)
        if self.padding:
            embedding = F.pad(embedding, (0, self.padding))
        return embedding


class RopeFrequency(nn.Module):
    def __init__(self, head_dim: int, ndim: int, max_wavelength: float):
        super().__init__()
        dim_per_axis = (head_dim - head_dim % ndim) // ndim
        self.padding = head_dim % ndim + (dim_per_axis % 2) * ndim
        assert self.padding % 2 == 0, f"head_dim={head_dim} incompatible with ndim={ndim}"
        effective = (head_dim - self.padding) // ndim
        assert effective > 0, f"head_dim={head_dim} too small for ndim={ndim}"
        frequencies = torch.arange(0, effective, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (frequencies / effective))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=coords.device.type, enabled=False):
            angles = coords.float().unsqueeze(-1) * self.omega
            angles = angles.flatten(start_dim=-2)
            if self.padding:
                angles = F.pad(angles, (0, self.padding // 2))
            return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(pairs * freqs.unsqueeze(1))
    return rotated.flatten(start_dim=-2).type_as(x)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, dim = x.shape
    return x.view(batch, seq_len, num_heads, dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).flatten(start_dim=2)
