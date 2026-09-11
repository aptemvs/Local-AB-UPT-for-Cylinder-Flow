import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(
        self, size: int, max_accumulations: int = 10**6, std_epsilon: float = 1e-8
    ) -> None:
        super().__init__()
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon
        self.register_buffer("acc_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("num_accumulations", torch.zeros((), dtype=torch.float64))
        self.register_buffer("acc_sum", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("acc_sum_squared", torch.zeros(size, dtype=torch.float64))
        self._num_accumulations = 0
        self._accumulating = True

    def forward(self, batched_data: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        if accumulate and self._accumulating:
            self._accumulate(batched_data.detach())
        return (batched_data - self.mean()) / self.std_with_epsilon()

    @torch._dynamo.disable
    @torch.no_grad()
    def _accumulate(self, batched_data: torch.Tensor) -> None:
        flat = batched_data.reshape(-1, batched_data.shape[-1]).to(torch.float64)
        self.acc_sum += flat.sum(dim=0)
        self.acc_sum_squared += flat.square().sum(dim=0)
        self.acc_count += flat.shape[0]
        self.num_accumulations += 1
        self._num_accumulations += 1
        if self._num_accumulations >= self.max_accumulations:
            self._accumulating = False

    def inverse(self, normalized_data: torch.Tensor) -> torch.Tensor:
        return normalized_data * self.std_with_epsilon() + self.mean()

    def mean(self) -> torch.Tensor:
        safe_count = self.acc_count.clamp_min(1.0)
        return (self.acc_sum / safe_count).to(torch.float32)

    def std_with_epsilon(self) -> torch.Tensor:
        safe_count = self.acc_count.clamp_min(1.0)
        variance = self.acc_sum_squared / safe_count - (self.acc_sum / safe_count).square()
        std = variance.clamp_min(0.0).sqrt().to(torch.float32)
        return std.clamp_min(self.std_epsilon)
