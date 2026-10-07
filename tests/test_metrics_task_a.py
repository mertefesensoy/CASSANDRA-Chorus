from __future__ import annotations

import types

import pytest
import torch

from cassandra_chorus.data import TaskA, TaskASection
from cassandra_chorus.metrics import evaluate_task_a, random_selector, routing_necessity
from cassandra_chorus.model import ModelSection, MoETransformer

TASK = TaskA(TaskASection())
EVAL = TASK.evaluation_set(4, seed=3)


class Oracle:
    """A stand-in model for the marked variant that reads the marker and looks the answer up.

    With ``wrong=True`` it predicts a value that is always wrong. It routes every
    token to experts (0, 1) in each of two layers.
    """

    def __init__(self, task: TaskA, wrong: bool = False) -> None:
        self.task, self.wrong, self.training = task, wrong, False
        self.cfg = types.SimpleNamespace(n_layers=2, n_experts=4)

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def __call__(self, inputs, expert_mask=None, return_routing=False, select=None):
        n, length = inputs.shape
        v = self.task.vocab_size
        map_ids = inputs[:, 0] - self.task.cfg.n_symbols
        answers = self.task.maps[map_ids.unsqueeze(1), inputs.clamp(max=self.task.cfg.n_symbols - 1)]
        if self.wrong:
            answers = (answers + 1) % self.task.cfg.n_symbols
        logits = torch.nn.functional.one_hot(answers, v).float()
        routing = torch.tensor([0, 1]).expand(n, length, 2)
        counts = torch.tensor([[n * length, n * length, 0, 0]] * 2)
        return types.SimpleNamespace(logits=logits, expert_counts=counts, routing=[routing, routing])


def test_perfect_and_wrong_predictors():
    perfect = evaluate_task_a(Oracle(TASK), EVAL, "cpu", n_maps=8)
    assert perfect["accuracy"] == 1.0 and perfect["per_map_accuracy"] == [1.0] * 8
    assert perfect["scored"] == int(EVAL.scored.sum())
    wrong = evaluate_task_a(Oracle(TASK, wrong=True), EVAL, "cpu", n_maps=8)
    assert wrong["accuracy"] == 0.0 and wrong["min_map_accuracy"] == 0.0


def test_map_expert_table_counts_scored_positions():
    result = evaluate_task_a(Oracle(TASK), EVAL, "cpu", n_maps=8, chunk=5)  # odd chunk exercises batching
    table = torch.tensor(result["map_expert_counts"])  # [layers, maps, experts]
    per_map_scored = torch.bincount(EVAL.map_ids.unsqueeze(1).expand_as(EVAL.targets)[EVAL.scored], minlength=8)
    assert torch.equal(table[0, :, 0], per_map_scored) and torch.equal(table[0, :, 1], per_map_scored)
    assert int(table[:, :, 2:].sum()) == 0


def test_real_model_evaluation_is_consistent():
    torch.manual_seed(0)
    model = MoETransformer(
        ModelSection(vocab_size=34, context_length=64, d_model=32, n_layers=2, n_heads=4, expert_hidden=32)
    )
    result = evaluate_task_a(model, EVAL, "cpu", n_maps=8)
    assert 0.0 <= result["accuracy"] <= 1.0 and result["scored_loss"] > 0
    table = torch.tensor(result["map_expert_counts"])
    assert torch.equal(table.sum(dim=(1, 2)), torch.full((2,), result["scored"] * 2))  # k = 2 per scored token
    assert model.training  # evaluation restored training mode


def test_random_selector_draws_distinct_available_experts_reproducibly():
    logits = torch.zeros(1000, 6)
    logits[:, 4] = float("-inf")
    a = random_selector(6, 2, seed=5, device="cpu")(0, logits)
    b = random_selector(6, 2, seed=5, device="cpu")(0, logits)
    assert torch.equal(a, b)
    assert not bool((a == 4).any()) and bool((a[:, 0] != a[:, 1]).all())
    share = torch.bincount(a.reshape(-1), minlength=6).double() / a.numel()
    assert torch.allclose(share[[0, 1, 2, 3, 5]], torch.full((5,), 0.2, dtype=torch.float64), atol=0.03)


def test_routing_necessity_structure():
    torch.manual_seed(0)
    model = MoETransformer(
        ModelSection(vocab_size=34, context_length=64, d_model=32, n_layers=2, n_heads=4, expert_hidden=32)
    )
    diag = routing_necessity(model, EVAL, "cpu", n_maps=8, seed=1)
    assert [r["expert"] for r in diag["leave_one_out"]] == list(range(8))
    for entry in [diag["trained"], diag["random_routing"], *diag["leave_one_out"]]:
        assert 0.0 <= entry["accuracy"] <= 1.0 and entry["min_map_accuracy"] <= entry["accuracy"] + 1e-12
