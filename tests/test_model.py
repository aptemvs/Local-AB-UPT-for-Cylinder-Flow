import pytest
import torch

from abupt.nn import AnchoredBranchedUPT
from abupt.nn.positional import ContinuousSincosEmbed

TINY = {"dim": 32, "num_heads": 2, "blocks": "ss", "local_radius": 0.05}
PAPER_MODELS = {
    "ssssssss": 2_050_850,
    "lllll": 1_298_738,
    "lllsss": 1_549_442,
    "sssss": 1_298_738,
    "ssssss": 1_549_442,
}


def make_inputs(batch=2, m=8, q=12):
    torch.manual_seed(0)
    anchor_pos = torch.rand(batch, m, 2) * torch.tensor([1.6, 0.41])
    anchor_val = torch.cat(
        [torch.randn(batch, m, 2), torch.eye(9)[torch.zeros(batch, m).long()]], dim=-1
    )
    query_pos = torch.rand(batch, q, 2) * torch.tensor([1.6, 0.41])
    query_onehot = torch.eye(9)[torch.zeros(batch, q).long()]
    return anchor_pos, anchor_val, query_pos, query_onehot


@pytest.fixture()
def tiny_model():
    return AnchoredBranchedUPT(**TINY)


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_forward_shapes(tiny_model):
    anchor_pos, anchor_val, query_pos, query_onehot = make_inputs()
    out = tiny_model(anchor_pos, anchor_val, query_pos, query_onehot)
    assert out.shape == (2, 12, 2)


def test_four_block_param_count_frozen():
    model = AnchoredBranchedUPT(dim=144, num_heads=3, blocks="ssss", local_radius=0.2)
    assert param_count(model) == 1_048_034


@pytest.mark.parametrize(("blocks", "expected"), PAPER_MODELS.items())
def test_paper_model_param_counts(blocks, expected):
    model = AnchoredBranchedUPT(
        dim=144, num_heads=3, blocks=blocks, local_radius=0.1, pos_embed="none", target="absolute"
    )
    assert param_count(model) == expected


def test_local_blocks_only_attend_within_radius():
    model = AnchoredBranchedUPT(**{**TINY, "blocks": "ll"})
    anchor_pos, anchor_val, query_pos, query_onehot = make_inputs(batch=1)
    model.train()
    model(anchor_pos, anchor_val, query_pos, query_onehot)
    model.eval()
    base = model(anchor_pos, anchor_val, query_pos, query_onehot)
    far = anchor_pos.clone()
    far[0, -1] = torch.tensor([5.0, 5.0])
    moved = model(far, anchor_val, query_pos, query_onehot)
    distances = torch.cdist(query_pos[0], anchor_pos[0, :-1])
    isolated = (distances > TINY["local_radius"]).all(dim=1)
    torch.testing.assert_close(moved[0, ~isolated], base[0, ~isolated])


def test_queries_do_not_influence_each_other(tiny_model):
    anchor_pos, anchor_val, query_pos, query_onehot = make_inputs(batch=1)
    tiny_model.train()
    tiny_model(anchor_pos, anchor_val, query_pos, query_onehot)
    tiny_model.eval()
    base = tiny_model(anchor_pos, anchor_val, query_pos, query_onehot)
    moved_pos = query_pos.clone()
    moved_pos[0, 0] = torch.tensor([0.01, 0.01])
    out = tiny_model(anchor_pos, anchor_val, moved_pos, query_onehot)
    torch.testing.assert_close(out[0, 1:], base[0, 1:])
    assert not torch.allclose(out[0, 0], base[0, 0])


def test_normalizers_accumulate_only_in_training(tiny_model):
    inputs = make_inputs()
    tiny_model.train()
    tiny_model(*inputs)
    count_after_train = tiny_model.node_normalizer.acc_count.item()
    assert count_after_train > 0
    tiny_model.eval()
    tiny_model(*inputs)
    assert tiny_model.node_normalizer.acc_count.item() == count_after_train


def test_predict_next_inverts_normalization(tiny_model):
    anchor_pos, anchor_val, query_pos, query_onehot = make_inputs()
    tiny_model.train()
    tiny_model.normalize_target(torch.randn(2, 12, 2) * 0.01)
    tiny_model.eval()
    current = torch.randn(2, 12, 2)
    nxt = tiny_model.predict_next(anchor_pos, anchor_val, query_pos, query_onehot, current)
    delta = tiny_model(anchor_pos, anchor_val, query_pos, query_onehot)
    torch.testing.assert_close(nxt, current + tiny_model.output_normalizer.inverse(delta.float()))


def test_absolute_predict_next_ignores_current():
    model = AnchoredBranchedUPT(**TINY, target="absolute")
    anchor_pos, anchor_val, query_pos, query_onehot = make_inputs()
    model.train()
    model.normalize_target(torch.randn(2, 12, 2))
    model.eval()
    a = model.predict_next(anchor_pos, anchor_val, query_pos, query_onehot, torch.zeros(2, 12, 2))
    b = model.predict_next(anchor_pos, anchor_val, query_pos, query_onehot, torch.ones(2, 12, 2))
    torch.testing.assert_close(a, b)
    normalized = model(anchor_pos, anchor_val, query_pos, query_onehot)
    torch.testing.assert_close(a, model.output_normalizer.inverse(normalized.float()))


def test_odd_head_dim_rejected():
    with pytest.raises(AssertionError):
        AnchoredBranchedUPT(dim=39, num_heads=3, blocks="ss", local_radius=0.1)


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        AnchoredBranchedUPT(**{**TINY, "blocks": "sx"})
    with pytest.raises(ValueError):
        AnchoredBranchedUPT(**TINY, pos_embed="learned")
    with pytest.raises(ValueError):
        AnchoredBranchedUPT(**TINY, target="velocity")


def test_rope_only_keeps_absolute_embedding_in_state_dict():
    model = AnchoredBranchedUPT(**TINY, pos_embed="none")
    assert not model.use_abs_pos
    assert "pos_embed.omega" in model.state_dict()


def test_positional_encoding_resolves_mesh_spacing():
    embed = ContinuousSincosEmbed(dim=144, ndim=2, max_wavelength=1.0e4)
    a = embed(torch.tensor([[0.400, 0.200]]) * 1000.0)
    b = embed(torch.tensor([[0.405, 0.200]]) * 1000.0)
    assert torch.nn.functional.cosine_similarity(a, b).item() < 0.99
