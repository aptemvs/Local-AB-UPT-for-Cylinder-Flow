import torch
from torch import nn

from abupt.dataset import NUM_NODE_TYPES
from abupt.nn.blocks import TransformerBlock
from abupt.nn.normalizer import Normalizer
from abupt.nn.positional import ContinuousSincosEmbed, RopeFrequency


class AnchoredBranchedUPT(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        blocks: str,
        local_radius: float,
        pos_embed: str = "sincos",
        position_scale: float = 1000.0,
        position_max_wavelength: float = 1.0e4,
        target: str = "delta",
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        assert dim % num_heads == 0 and head_dim % 2 == 0, (
            f"RoPE needs an even head_dim; got dim={dim}, heads={num_heads}"
        )
        if not set(blocks) <= {"s", "l"}:
            raise ValueError(f"blocks must consist of 's' and 'l', got {blocks!r}")
        if pos_embed not in ("sincos", "none"):
            raise ValueError(f"pos_embed must be 'sincos' or 'none', got {pos_embed!r}")
        if target not in ("delta", "absolute"):
            raise ValueError(f"target must be 'delta' or 'absolute', got {target!r}")
        self.block_kinds = blocks
        self.local_radius = local_radius
        self.position_scale = position_scale
        self.use_abs_pos = pos_embed == "sincos"
        self.absolute_target = target == "absolute"

        self.rope = RopeFrequency(head_dim=head_dim, ndim=2, max_wavelength=position_max_wavelength)
        self.pos_embed = ContinuousSincosEmbed(
            dim=dim, ndim=2, max_wavelength=position_max_wavelength
        )

        node_features = 2 + NUM_NODE_TYPES
        self.anchor_encoder = nn.Sequential(
            nn.Linear(node_features, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(NUM_NODE_TYPES, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(TransformerBlock(dim, num_heads) for _ in blocks)
        self.decoder = nn.Linear(dim, 2)

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_weights)

        self.node_normalizer = Normalizer(size=node_features)
        self.output_normalizer = Normalizer(size=2)

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return self.output_normalizer(target, accumulate=self.training)

    def forward(
        self,
        anchor_pos: torch.Tensor,
        anchor_val: torch.Tensor,
        query_pos: torch.Tensor,
        query_val: torch.Tensor,
    ) -> torch.Tensor:
        num_anchors = anchor_pos.shape[1]
        local_mask = None
        if "l" in self.block_kinds:
            with torch.autocast(device_type=anchor_pos.device.type, enabled=False):
                all_pos = torch.cat([anchor_pos, query_pos], dim=1)
                dist = torch.cdist(all_pos.float(), anchor_pos.float())
                local_mask = dist <= self.local_radius
                local_mask.scatter_(-1, dist.argmin(dim=-1, keepdim=True), True)
                local_mask = local_mask.unsqueeze(1)
        anchor_pos = anchor_pos * self.position_scale
        query_pos = query_pos * self.position_scale

        anchor_feats = self.node_normalizer(anchor_val, accumulate=self.training)
        onehot_mean = self.node_normalizer.mean()[2:]
        onehot_std = self.node_normalizer.std_with_epsilon()[2:]
        query_feats = (query_val - onehot_mean) / onehot_std
        anchor_tokens = self.anchor_encoder(anchor_feats)
        query_tokens = self.query_encoder(query_feats)
        if self.use_abs_pos:
            anchor_tokens = anchor_tokens + self.pos_embed(anchor_pos)
            query_tokens = query_tokens + self.pos_embed(query_pos)
        x = torch.cat([anchor_tokens, query_tokens], dim=1)
        freqs = self.rope(torch.cat([anchor_pos, query_pos], dim=1))

        for kind, block in zip(self.block_kinds, self.blocks, strict=True):
            x = block(
                x,
                freqs=freqs,
                num_anchors=num_anchors,
                attn_mask=local_mask if kind == "l" else None,
            )
        return self.decoder(x)[:, num_anchors:]

    @torch.no_grad()
    def predict_next(
        self,
        anchor_pos: torch.Tensor,
        anchor_val: torch.Tensor,
        query_pos: torch.Tensor,
        query_val: torch.Tensor,
        current_query_velocity: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self(anchor_pos, anchor_val, query_pos, query_val)
        prediction = self.output_normalizer.inverse(normalized.float())
        if self.absolute_target:
            return prediction
        return current_query_velocity + prediction
