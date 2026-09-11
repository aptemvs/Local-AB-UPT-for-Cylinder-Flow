import numpy as np
import torch

from abupt.nn.normalizer import Normalizer


def test_statistics_match_numpy():
    torch.manual_seed(0)
    normalizer = Normalizer(size=3)
    chunks = [torch.randn(50, 3) * 2.5 + 1.0 for _ in range(4)]
    for chunk in chunks:
        normalizer(chunk, accumulate=True)
    stacked = torch.cat(chunks).numpy()
    np.testing.assert_allclose(normalizer.mean().numpy(), stacked.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(
        normalizer.std_with_epsilon().numpy(), stacked.std(axis=0), rtol=1e-4
    )


def test_normalized_output_is_standardized():
    normalizer = Normalizer(size=2)
    data = torch.randn(1000, 2) * 3.0 - 5.0
    out = normalizer(data, accumulate=True)
    assert out.mean(dim=0).abs().max().item() < 1e-4
    assert (out.std(dim=0, unbiased=False) - 1).abs().max().item() < 1e-3


def test_inverse_roundtrip():
    normalizer = Normalizer(size=2)
    normalizer(torch.randn(100, 2) * 4 + 2, accumulate=True)
    x = torch.randn(10, 2)
    torch.testing.assert_close(normalizer.inverse(normalizer(x)), x, rtol=1e-5, atol=1e-5)


def test_no_accumulation_when_disabled():
    normalizer = Normalizer(size=2)
    normalizer(torch.randn(10, 2), accumulate=True)
    before = normalizer.acc_count.item()
    normalizer(torch.randn(10, 2), accumulate=False)
    assert normalizer.acc_count.item() == before


def test_freeze_at_max_accumulations():
    normalizer = Normalizer(size=2, max_accumulations=2)
    normalizer(torch.randn(10, 2), accumulate=True)
    normalizer(torch.randn(10, 2), accumulate=True)
    frozen_mean = normalizer.mean().clone()
    normalizer(torch.randn(10, 2) + 100, accumulate=True)
    torch.testing.assert_close(normalizer.mean(), frozen_mean)
    assert normalizer.num_accumulations.item() == 2


def test_state_dict_roundtrip():
    source = Normalizer(size=2)
    source(torch.randn(100, 2) * 3 + 1, accumulate=True)
    target = Normalizer(size=2)
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.mean(), source.mean())
    torch.testing.assert_close(target.std_with_epsilon(), source.std_with_epsilon())
