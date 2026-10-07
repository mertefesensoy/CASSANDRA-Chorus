from __future__ import annotations

import dataclasses
import hashlib
import warnings

import pytest
import torch
import torch.nn.functional as F

from cassandra_chorus import repro
from cassandra_chorus.config import load_config
from cassandra_chorus.model import (
    ModelSection,
    MoELayer,
    MoETransformer,
    count_parameters,
    expected_parameter_count,
    expert_param_owner,
    switch_balance_loss,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TINY = ModelSection(vocab_size=11, context_length=16, d_model=32, n_layers=2, n_heads=4, expert_hidden=48, n_experts=4)


def make(cfg: ModelSection = TINY, held=None, seed: int = 0, device: str = "cpu") -> MoETransformer:
    torch.manual_seed(seed)
    return MoETransformer(cfg, held).to(device)


def tokens(batch: int = 3, length: int = 12, vocab: int = 11, seed: int = 1, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g).to(device)


def slice_from_full(full: MoETransformer, held) -> MoETransformer:
    """Build a slice and load it from the full model's state dict, filtered by name."""
    part = MoETransformer(full.cfg, held).to(next(full.parameters()).device)
    state = {
        name: value
        for name, value in full.state_dict().items()
        if (owner := expert_param_owner(name)) is None or owner[1] in held[owner[0]]
    }
    part.load_state_dict(state, strict=True)
    return part


# ---------------------------------------------------------------- shapes and contract


def test_output_shapes_and_routing_counts():
    model = make()
    idx = tokens()
    out = model(idx, targets=idx, return_routing=True)
    assert out.logits.shape == (3, 12, 11)
    assert out.loss is not None and out.ce_loss is not None and out.balance_loss.dim() == 0
    assert out.expert_counts.shape == (2, 4)
    assert out.expert_counts.sum(dim=1).tolist() == [3 * 12 * 2] * 2  # every token goes to k = 2 experts
    assert [r.shape for r in out.routing] == [torch.Size([3, 12, 2])] * 2
    assert model(idx).loss is None and model(idx).routing is None


def test_loss_is_ce_plus_weighted_balance():
    cfg = dataclasses.replace(TINY, balance_coef=0.3)
    out = make(cfg)(tokens(), targets=tokens(seed=2))
    assert torch.allclose(out.loss, out.ce_loss + 0.3 * out.balance_loss)
    plain = make(TINY)(tokens(), targets=tokens(seed=2))
    assert torch.equal(plain.loss, plain.ce_loss)  # coefficient 0 by default


def test_ignored_targets():
    model = make()
    idx, targets = tokens(), tokens(seed=2)
    masked = targets.clone()
    masked[:, :5] = -100
    out = model(idx, targets=masked)
    logits = out.logits[:, 5:].reshape(-1, 11)
    assert torch.allclose(out.ce_loss, F.cross_entropy(logits, targets[:, 5:].reshape(-1)))


def test_causality():
    model = make().eval()
    a = tokens()
    b = a.clone()
    b[:, 7] = (b[:, 7] + 1) % 11
    oa, ob = model(a, return_routing=True), model(b, return_routing=True)
    # Routing decisions before the change are identical. Logits there agree to
    # rounding only, not bitwise: changing token 7 regroups the tokens each
    # expert processes, and a matrix product over a different number of rows
    # may round differently (measured: 3e-8 on CPU).
    assert all(torch.equal(x[:, :7], y[:, :7]) for x, y in zip(oa.routing, ob.routing))
    assert torch.allclose(oa.logits[:, :7], ob.logits[:, :7], atol=1e-6, rtol=0)
    assert (oa.logits[:, 7:] - ob.logits[:, 7:]).abs().max() > 1e-2


def test_sequence_too_long():
    with pytest.raises(ValueError, match="exceeds context_length"):
        make()(tokens(length=17))


# ---------------------------------------------------------------- configuration checks


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"top_k": 1}, "router gets no gradient"),
        ({"top_k": 5}, "cannot exceed n_experts"),
        ({"n_heads": 5}, "divisible by n_heads"),
        ({"d_model": 24, "n_heads": 8}, "must be even"),
        ({"dropout": 1.0}, "dropout"),
        ({"balance_coef": -0.1}, "balance_coef"),
        ({"vocab_size": 0}, "vocab_size"),
    ],
)
def test_invalid_model_config(changes, message):
    with pytest.raises(ValueError, match=message):
        MoETransformer(dataclasses.replace(TINY, **changes))


@pytest.mark.parametrize(
    ("held", "message"),
    [
        ({0: [0, 1]}, "must name every layer"),
        ({0: [0, 1], 1: [0, 1], 2: [0, 1]}, "must name every layer"),
        ({0: [0, 4], 1: [0, 1]}, "must be in 0..3"),
        ({0: [2], 1: [0, 1]}, "fewer than top_k"),
    ],
)
def test_invalid_held(held, message):
    with pytest.raises(ValueError, match=message):
        MoETransformer(TINY, held)


def test_model_section_loads_from_toml(tmp_path):
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        model: ModelSection

    path = tmp_path / "m.toml"
    path.write_text(
        "[model]\nvocab_size = 11\ncontext_length = 16\nd_model = 32\nn_layers = 2\n"
        'n_heads = 4\nexpert_hidden = 48\nn_experts = 4\nattention = "sdpa"\n',
        encoding="utf-8",
    )
    cfg = load_config(path, Cfg)
    assert cfg.model == dataclasses.replace(TINY, attention="sdpa")


# ---------------------------------------------------------------- routing and masking


def test_gates_sum_to_one_and_only_available_experts_are_chosen():
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, hidden=8, n_experts=6, top_k=3, held=[0, 1, 2, 4, 5])
    x = torch.randn(2, 9, 16)
    full = layer(x)
    assert torch.allclose(full.gates.sum(-1), torch.ones(2, 9))
    assert set(full.top_idx.unique().tolist()) <= {0, 1, 2, 4, 5}  # 3 is not held
    masked = layer(x, allowed=[1, 4, 5])
    assert set(masked.top_idx.unique().tolist()) <= {1, 4, 5}
    assert torch.allclose(masked.gates.sum(-1), torch.ones(2, 9))


def test_too_few_available_experts_at_forward():
    model = make()
    with pytest.raises(ValueError, match="only 1 experts are available"):
        model(tokens(), expert_mask={0: [3]})
    with pytest.raises(ValueError, match="layers that do not exist"):
        model(tokens(), expert_mask={5: [0, 1]})


def test_mask_allowing_everything_equals_no_mask():
    model = make()
    idx = tokens()
    all_allowed = {layer: range(4) for layer in range(2)}
    assert torch.equal(model(idx).logits, model(idx, expert_mask=all_allowed).logits)


def test_sparse_dispatch_matches_dense_reference():
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, hidden=24, n_experts=5, top_k=2, held=range(5))
    x = torch.randn(3, 7, 16)
    result = layer(x, allowed=[0, 2, 3, 4])
    flat = x.reshape(-1, 16)
    dense_gate = torch.zeros(flat.shape[0], 5)
    dense_gate.scatter_(1, result.top_idx.reshape(-1, 2), result.gates.reshape(-1, 2))
    dense = sum(dense_gate[:, e : e + 1] * layer.experts[str(e)](flat) for e in range(5))
    assert torch.allclose(result.output.reshape(-1, 16), dense, atol=1e-6, rtol=1e-5)


def test_router_rows_of_unavailable_experts_get_no_gradient():
    model = make()
    out = model(tokens(), targets=tokens(seed=2), expert_mask={0: [0, 1], 1: [0, 1]})
    out.loss.backward()
    for block in model.blocks:
        grad = block.moe.router.weight.grad
        assert torch.count_nonzero(grad[2:]) == 0
        assert torch.count_nonzero(grad[:2]) > 0
        assert all(block.moe.experts[str(e)].w1.weight.grad is None for e in (2, 3))


def test_balance_loss_values():
    uniform_idx = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    uniform_p = torch.full((4, 4), 0.25)
    assert torch.allclose(switch_balance_loss(uniform_idx, uniform_p, 4), torch.tensor(1.0))
    collapsed_idx = torch.tensor([[0, 1]] * 4)
    collapsed_p = torch.tensor([[0.5, 0.5, 0.0, 0.0]] * 4)
    assert torch.allclose(switch_balance_loss(collapsed_idx, collapsed_p, 4), torch.tensor(2.0))


def test_balance_loss_is_one_when_router_is_uniform():
    model = make()
    with torch.no_grad():
        for block in model.blocks:
            block.moe.router.weight.zero_()
    out = model(tokens())
    assert torch.allclose(out.balance_loss, torch.tensor(1.0))


# ---------------------------------------------------------------- slices


HELD = {0: [1, 3], 1: [0, 2, 3]}


@pytest.mark.parametrize("dispatch", ["sparse", "dense"])
def test_slice_equals_full_model_under_matching_mask(dispatch):
    full = make(dataclasses.replace(TINY, dispatch=dispatch))
    part = slice_from_full(full, HELD)
    assert part.held_experts() == {0: (1, 3), 1: (0, 2, 3)}
    idx, targets = tokens(), tokens(seed=2)
    a = full(idx, targets=targets, expert_mask=HELD, return_routing=True)
    b = part(idx, targets=targets, return_routing=True)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.expert_counts, b.expert_counts)
    assert all(torch.equal(x, y) for x, y in zip(a.routing, b.routing))
    a.loss.backward()
    b.loss.backward()
    full_grads = dict(full.named_parameters())
    for name, param in part.named_parameters():
        assert torch.equal(param.grad, full_grads[name].grad), name


def test_slice_state_dict_is_a_filtered_full_state_dict():
    full_keys = set(make().state_dict())
    part_keys = set(MoETransformer(TINY, HELD).state_dict())
    assert part_keys < full_keys
    dropped = full_keys - part_keys
    assert {expert_param_owner(k) for k in dropped} == {(0, 0), (0, 2), (1, 1)}


def test_expert_param_owner():
    assert expert_param_owner("blocks.2.moe.experts.5.w1.weight") == (2, 5)
    assert expert_param_owner("blocks.11.moe.experts.0.w2.weight") == (11, 0)
    assert expert_param_owner("blocks.2.moe.router.weight") is None
    assert expert_param_owner("embed.weight") is None
    assert expert_param_owner("blocks.0.attn.qkv.weight") is None


# ---------------------------------------------------------------- parameter counts


@pytest.mark.parametrize("held", [None, HELD])
def test_parameter_count_matches_formula(held):
    assert count_parameters(MoETransformer(TINY, held)) == expected_parameter_count(TINY, held)


def test_worked_examples_in_the_implementation_doc():
    task_a = ModelSection(vocab_size=34, context_length=64, d_model=128, n_layers=4, n_heads=4, expert_hidden=256)
    task_b = ModelSection(vocab_size=27, context_length=256, d_model=384, n_layers=6, n_heads=6, expert_hidden=768)
    assert expected_parameter_count(task_a)["total"] == 3_421_824
    assert expected_parameter_count(task_b)["total"] == 46_050_432
    assert count_parameters(MoETransformer(task_a))["total"] == 3_421_824


# ---------------------------------------------------------------- attention implementations


def test_sdpa_matches_explicit_attention():
    explicit = make().eval()
    fused = make(dataclasses.replace(TINY, attention="sdpa")).eval()
    fused.load_state_dict(explicit.state_dict())
    idx = tokens()
    assert torch.allclose(explicit(idx).logits, fused(idx).logits, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- CUDA: determinism


def train_step_checksum(device: str, dispatch: str = "sparse") -> str:
    model = make(dataclasses.replace(TINY, dispatch=dispatch), device=device)
    out = model(tokens(device=device), targets=tokens(seed=2, device=device))
    out.loss.backward()
    digest = hashlib.sha256(out.logits.detach().cpu().numpy().tobytes())
    for _, param in sorted(model.named_parameters()):
        digest.update(param.grad.cpu().numpy().tobytes())
    return digest.hexdigest()


@requires_cuda
@pytest.mark.gpu
@pytest.mark.usefixtures("restore_torch_determinism")
@pytest.mark.parametrize("dispatch", ["sparse", "dense"])
def test_strict_mode_runs_and_is_bit_identical_on_cuda(dispatch):
    # strict mode raises on any operation without a deterministic kernel, so this
    # also proves that forward and backward use only deterministic kernels.
    repro.apply_determinism("strict")
    assert train_step_checksum("cuda", dispatch) == train_step_checksum("cuda", dispatch)


@requires_cuda
@pytest.mark.gpu
@pytest.mark.usefixtures("restore_torch_determinism")
def test_warn_mode_reports_nothing_for_explicit_attention_on_cuda():
    repro.apply_determinism("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_step_checksum("cuda")
    assert not [w for w in caught if "deterministic" in str(w.message)]


@requires_cuda
@pytest.mark.gpu
@pytest.mark.usefixtures("restore_torch_determinism")
@pytest.mark.parametrize("dispatch", ["sparse", "dense"])
def test_slice_equals_full_model_on_cuda(dispatch):
    repro.apply_determinism("strict")
    full = make(dataclasses.replace(TINY, dispatch=dispatch), device="cuda")
    part = slice_from_full(full, HELD)
    idx, targets = tokens(device="cuda"), tokens(seed=2, device="cuda")
    a = full(idx, targets=targets, expert_mask=HELD)
    b = part(idx, targets=targets)
    assert torch.equal(a.logits, b.logits)
    a.loss.backward()
    b.loss.backward()
    full_grads = dict(full.named_parameters())
    for name, param in part.named_parameters():
        assert torch.equal(param.grad, full_grads[name].grad), name


# ---------------------------------------------------------------- dispatch modes and the select hook


def test_dense_dispatch_matches_sparse():
    sparse = make()
    dense = make(dataclasses.replace(TINY, dispatch="dense"))
    dense.load_state_dict(sparse.state_dict())
    idx, targets = tokens(), tokens(seed=2)
    a = sparse(idx, targets=targets, expert_mask=HELD, return_routing=True)
    b = dense(idx, targets=targets, expert_mask=HELD, return_routing=True)
    assert all(torch.equal(x, y) for x, y in zip(a.routing, b.routing))  # same selection
    assert torch.allclose(a.logits, b.logits, atol=1e-5, rtol=1e-4)
    a.loss.backward()
    b.loss.backward()
    dense_params = dict(dense.named_parameters())
    for name, param in sparse.named_parameters():
        other = dense_params[name].grad
        if param.grad is None:  # experts masked out: no gradient in either mode
            assert other is None, name
        else:
            assert torch.allclose(param.grad, other, atol=1e-5, rtol=1e-3), name


def test_select_hook_reproducing_topk_changes_nothing():
    model = make()
    idx = tokens()
    topk = lambda layer, logits: logits.topk(2, dim=-1).indices  # noqa: E731
    assert torch.equal(model(idx).logits, model(idx, select=topk).logits)


def test_select_hook_controls_routing_and_gates_use_router_logits():
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, hidden=8, n_experts=4, top_k=2, held=range(4))
    x = torch.randn(1, 5, 16)
    forced = torch.tensor([[2, 3]] * 5)
    result = layer(x, select=lambda logits: forced)
    assert torch.equal(result.top_idx.reshape(-1, 2), forced)
    router_logits = layer.router(x.reshape(-1, 16))
    assert torch.allclose(result.gates.reshape(-1, 2), torch.softmax(router_logits[:, [2, 3]], dim=-1))


@pytest.mark.parametrize(
    ("chosen", "message"),
    [
        (lambda n: torch.tensor([[0, 1, 2]] * n), "shape"),
        (lambda n: torch.tensor([[0, 3]] * n), "unavailable"),
        (lambda n: torch.tensor([[1, 1]] * n), "distinct"),
    ],
)
def test_select_hook_is_validated(chosen, message):
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, hidden=8, n_experts=4, top_k=2, held=[0, 1, 2])
    with pytest.raises(ValueError, match=message):
        layer(torch.randn(1, 3, 16), select=lambda logits: chosen(3))


def test_invalid_dispatch():
    with pytest.raises(ValueError, match="dispatch"):
        MoELayer(d_model=16, hidden=8, n_experts=4, top_k=2, held=range(4), dispatch="fast")
