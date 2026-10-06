from __future__ import annotations

import dataclasses

import pytest
import torch

from cassandra_chorus.config import load_config
from cassandra_chorus.data import (
    IGNORE,
    TaskA,
    TaskABatch,
    TaskASection,
    bayes_optimal_accuracy,
    first_occurrence,
)
from cassandra_chorus.model import ModelSection, MoETransformer

MARKED = TaskA(TaskASection())
UNMARKED = TaskA(TaskASection(marked=False))


def gen(seed: int = 7) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- maps and vocabulary


def test_maps_depend_only_on_task_seed():
    assert MARKED.maps.shape == (8, 26)
    assert int(MARKED.maps.min()) >= 0 and int(MARKED.maps.max()) <= 25
    assert torch.equal(MARKED.maps, TaskA(TaskASection()).maps)
    assert torch.equal(MARKED.maps, UNMARKED.maps)  # the variant does not change the maps
    assert not torch.equal(MARKED.maps, TaskA(TaskASection(task_seed=1)).maps)


def test_vocabulary():
    assert MARKED.vocab_size == UNMARKED.vocab_size == 34
    assert MARKED.marker(0) == 26 and MARKED.marker(7) == 33


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"seq_len": 63}, "even"), ({"n_symbols": 1}, "n_symbols"), ({"n_maps": 0}, "n_maps")],
)
def test_invalid_config(changes, message):
    with pytest.raises(ValueError, match=message):
        TaskA(dataclasses.replace(TaskASection(), **changes))


def test_section_loads_from_toml(tmp_path):
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        task_a: TaskASection

    path = tmp_path / "t.toml"
    path.write_text("[task_a]\nmarked = false\nmap_weights = [1, 1, 1, 1, 0, 0, 0, 2]\n", encoding="utf-8")
    cfg = load_config(path, Cfg)
    assert cfg.task_a == TaskASection(marked=False, map_weights=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0))


# ---------------------------------------------------------------- token layout


def check_scored_targets_follow_the_map(task: TaskA, batch: TaskABatch) -> None:
    rows, cols = batch.scored.nonzero(as_tuple=True)
    keys_at = batch.inputs[rows, cols]
    assert torch.equal(batch.targets[rows, cols], task.maps[batch.map_ids[rows], keys_at])


def test_marked_layout():
    batch = MARKED.sample(50, gen())
    assert batch.inputs.shape == batch.targets.shape == (50, 64)
    assert torch.equal(batch.inputs[:, 0], 26 + batch.map_ids)
    assert torch.equal(batch.inputs[:, 1::2], batch.keys)  # k1..k32 at positions 1, 3, ..., 63
    assert torch.equal(batch.inputs[:, 2::2], batch.values[:, :31])  # v1..v31 at positions 2, ..., 62
    assert bool((batch.inputs[:, 1:] < 26).all())  # no marker after position 0
    assert bool((batch.targets[:, 0::2] == IGNORE).all())  # only key positions can be scored
    check_scored_targets_follow_the_map(MARKED, batch)


def test_unmarked_layout():
    batch = UNMARKED.sample(50, gen())
    assert torch.equal(batch.inputs[:, 0::2], batch.keys)  # k1..k32 at positions 0, 2, ..., 62
    assert torch.equal(batch.inputs[:, 1::2], batch.values)  # v1..v32 at positions 1, ..., 63
    assert bool((batch.inputs < 26).all())  # markers never appear
    assert bool((batch.targets[:, 1::2] == IGNORE).all())
    check_scored_targets_follow_the_map(UNMARKED, batch)


@pytest.mark.parametrize("task", [MARKED, UNMARKED], ids=["marked", "unmarked"])
def test_only_first_occurrences_are_scored(task):
    batch = task.sample(200, gen())
    first = first_occurrence(batch.keys, 26)
    offset = 1 if task.cfg.marked else 0
    key_positions = offset + 2 * torch.arange(32)
    assert torch.equal(batch.scored[:, key_positions], first)
    assert torch.equal(batch.scored.sum(dim=1), torch.tensor([len(set(row.tolist())) for row in batch.keys]))


def test_first_occurrence_helper():
    keys = torch.tensor([[3, 1, 3, 2, 1, 0]])
    assert first_occurrence(keys, 4).tolist() == [[True, True, False, True, False, True]]


def test_mean_number_of_scored_positions_matches_formula():
    batch = MARKED.sample(4000, gen())
    expected = 26 * (1 - (25 / 26) ** 32)  # about 18.59
    assert abs(batch.scored.sum(dim=1).double().mean().item() - expected) < 0.15


# ---------------------------------------------------------------- determinism, weights, evaluation set


def test_same_generator_state_gives_same_batch():
    a, b, c = MARKED.sample(20, gen(7)), MARKED.sample(20, gen(7)), MARKED.sample(20, gen(11))
    for field in dataclasses.fields(TaskABatch):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name))
    assert not torch.equal(a.inputs, c.inputs)


def test_map_weights():
    only_first = MARKED.sample(100, gen(), map_weights=[1, 0, 0, 0, 0, 0, 0, 0])
    assert bool((only_first.map_ids == 0).all())
    skewed = TaskA(TaskASection(map_weights=(3, 1, 0, 0, 0, 0, 0, 0))).sample(8000, gen())
    share = (skewed.map_ids == 0).double().mean().item()
    assert set(skewed.map_ids.unique().tolist()) == {0, 1} and abs(share - 0.75) < 0.02
    for bad in ([1, 1], [1, -1, 0, 0, 0, 0, 0, 0], [0] * 8):
        with pytest.raises(ValueError, match="map_weights"):
            MARKED.sample(5, gen(), map_weights=bad)


def test_explicit_map_ids():
    batch = MARKED.sample(3, gen(), map_ids=[5, 5, 2])
    assert batch.map_ids.tolist() == [5, 5, 2]
    with pytest.raises(ValueError, match="map_ids"):
        MARKED.sample(2, gen(), map_ids=[0, 8])


def test_evaluation_set_is_fixed_and_balanced():
    a, b = MARKED.evaluation_set(16, seed=99), MARKED.evaluation_set(16, seed=99)
    assert torch.equal(a.inputs, b.inputs)
    assert torch.bincount(a.map_ids, minlength=8).tolist() == [16] * 8
    assert not torch.equal(a.inputs, MARKED.evaluation_set(16, seed=100).inputs)


# ---------------------------------------------------------------- reference accuracy


def test_bayes_optimal_marked_is_one():
    assert bayes_optimal_accuracy(MARKED, MARKED.sample(100, gen())) == 1.0


def test_bayes_optimal_unmarked_with_one_map_is_one():
    task = TaskA(TaskASection(n_maps=1, marked=False))
    assert bayes_optimal_accuracy(task, task.sample(50, gen())) == pytest.approx(1.0)


def test_bayes_optimal_hand_worked_example():
    # Two maps over two symbols: f0 = (0, 0), f1 = (0, 1). A map-1 sequence with
    # keys (0, 1): the first value is 0 under both maps (certain); after seeing
    # (0 -> 0) both maps remain possible, so the second value is a coin flip.
    task = TaskA(TaskASection(n_maps=2, n_symbols=2, seq_len=4, marked=False))
    task.maps = torch.tensor([[0, 0], [0, 1]])
    batch = TaskABatch(
        inputs=torch.zeros(1, 4, dtype=torch.long),
        targets=torch.zeros(1, 4, dtype=torch.long),
        scored=torch.ones(1, 4, dtype=torch.bool),
        map_ids=torch.tensor([1]),
        keys=torch.tensor([[0, 1]]),
        values=torch.tensor([[0, 1]]),
    )
    assert bayes_optimal_accuracy(task, batch) == pytest.approx(0.75)


def test_bayes_optimal_unmarked_reference_parameters():
    value = bayes_optimal_accuracy(UNMARKED, UNMARKED.evaluation_set(256, seed=1))
    assert 0.9 < value < 1.0


# ---------------------------------------------------------------- model compatibility


def test_batch_feeds_the_model():
    torch.manual_seed(0)
    model = MoETransformer(
        ModelSection(vocab_size=34, context_length=64, d_model=32, n_layers=2, n_heads=4, expert_hidden=32)
    )
    batch = MARKED.sample(4, gen())
    out = model(batch.inputs, targets=batch.targets)
    assert torch.isfinite(out.loss)
    assert out.ce_loss.item() == pytest.approx(3.526, abs=0.3)  # about ln(34) for an untrained model
