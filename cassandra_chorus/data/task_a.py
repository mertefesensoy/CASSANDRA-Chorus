"""Task A: synthetic lookup maps (SRS S0-F-15 to S0-F-18).

Each sequence is produced by one of M random maps over V symbols and consists
of 32 key-value pairs (for L = 64). Only the first occurrence of each key is
scored and trained on; a repeated key could be answered by copying from earlier
in the sequence instead of recalling the map. The maps come from ``task_seed``
alone, so they are the same for every run seed.

Token layout and all formulas: ``docs/implementations/2026-10-07-task-a-generator.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch
import torch.nn.functional as F

IGNORE = -100  # target value the model's loss ignores


@dataclasses.dataclass(frozen=True)
class TaskASection:
    """Task A configuration (the ``[task_a]`` section). Defaults are decision D5."""

    n_maps: int = 8
    n_symbols: int = 26
    seq_len: int = 64
    marked: bool = True
    task_seed: int = 20261007
    map_weights: tuple[float, ...] = ()  # empty means uniform


@dataclasses.dataclass
class TaskABatch:
    """One batch of Task A sequences (all CPU int64 tensors except ``scored``).

    ``inputs`` and ``targets`` are ``[n, L]``; ``targets`` is -100 wherever the
    position is not scored. ``scored`` is ``targets != -100``. ``map_ids`` is
    ``[n]``. ``keys`` and ``values`` are the ``[n, P]`` pairs, in order.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    scored: torch.Tensor
    map_ids: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor

    def to(self, device: torch.device | str) -> TaskABatch:
        return TaskABatch(**{f.name: getattr(self, f.name).to(device) for f in dataclasses.fields(self)})


def first_occurrence(keys: torch.Tensor, n_symbols: int) -> torch.Tensor:
    """``[n, P]`` bool: True where ``keys[:, j]`` does not appear in ``keys[:, :j]``."""
    counts = F.one_hot(keys, n_symbols).cumsum(dim=1)  # inclusive running count of each symbol
    return counts.gather(2, keys.unsqueeze(2)).squeeze(2) == 1


class TaskA:
    """The M maps of one task, and sampling of sequences from them."""

    def __init__(self, cfg: TaskASection) -> None:
        problems = []
        if cfg.n_maps < 1:
            problems.append("n_maps must be at least 1")
        if cfg.n_symbols < 2:
            problems.append("n_symbols must be at least 2")
        if cfg.seq_len < 2 or cfg.seq_len % 2 != 0:
            problems.append("seq_len must be an even number of at least 2")
        if problems:
            raise ValueError("invalid Task A configuration: " + "; ".join(problems))
        self.cfg = cfg
        self.n_pairs = cfg.seq_len // 2
        generator = torch.Generator().manual_seed(cfg.task_seed)
        self.maps = torch.randint(0, cfg.n_symbols, (cfg.n_maps, cfg.n_symbols), generator=generator)
        self.default_weights = self.normalize_weights(cfg.map_weights)

    @property
    def vocab_size(self) -> int:
        """Symbols plus one marker per map (reserved in both variants)."""
        return self.cfg.n_symbols + self.cfg.n_maps

    def marker(self, map_id: int) -> int:
        return self.cfg.n_symbols + map_id

    def normalize_weights(self, weights: Sequence[float]) -> torch.Tensor:
        """Map sampling probabilities. Empty means uniform."""
        m = self.cfg.n_maps
        if len(weights) == 0:
            return torch.full((m,), 1.0 / m, dtype=torch.float64)
        w = torch.tensor(list(weights), dtype=torch.float64)
        if w.shape != (m,):
            raise ValueError(f"map_weights must have {m} entries, got {len(weights)}")
        if bool((w < 0).any()) or not float(w.sum()) > 0:
            raise ValueError("map_weights must be non-negative with a positive sum")
        return w / w.sum()

    def sample(
        self,
        n: int,
        generator: torch.Generator,
        map_ids: Sequence[int] | torch.Tensor | None = None,
        map_weights: Sequence[float] | None = None,
    ) -> TaskABatch:
        """Draw ``n`` sequences using only ``generator`` (a CPU generator) for randomness.

        ``map_ids`` fixes which map produces each sequence; otherwise maps are
        drawn from ``map_weights`` (default: the configuration's weights).
        """
        cfg, pairs = self.cfg, self.n_pairs
        if map_ids is None:
            weights = self.default_weights if map_weights is None else self.normalize_weights(map_weights)
            map_ids = torch.multinomial(weights, n, replacement=True, generator=generator)
        else:
            map_ids = torch.as_tensor(map_ids, dtype=torch.long)
            if map_ids.shape != (n,) or bool((map_ids < 0).any()) or bool((map_ids >= cfg.n_maps).any()):
                raise ValueError(f"map_ids must be {n} integers in 0..{cfg.n_maps - 1}")

        n_keys = pairs if cfg.marked else pairs + 1  # unmarked needs one trailing key to complete the targets
        all_keys = torch.randint(0, cfg.n_symbols, (n, n_keys), generator=generator)
        keys = all_keys[:, :pairs]
        values = self.maps[map_ids.unsqueeze(1), keys]
        interleaved = torch.stack((keys, values), dim=2).reshape(n, 2 * pairs)
        if cfg.marked:
            tokens = torch.cat(((cfg.n_symbols + map_ids).unsqueeze(1), interleaved), dim=1)
        else:
            tokens = torch.cat((interleaved, all_keys[:, pairs:]), dim=1)

        offset = 1 if cfg.marked else 0
        key_positions = offset + 2 * torch.arange(pairs)  # input position of each key
        targets = torch.full((n, cfg.seq_len), IGNORE, dtype=torch.long)
        targets[:, key_positions] = torch.where(first_occurrence(keys, cfg.n_symbols), values, IGNORE)
        return TaskABatch(
            inputs=tokens[:, : cfg.seq_len].clone(),
            targets=targets,
            scored=targets != IGNORE,
            map_ids=map_ids,
            keys=keys,
            values=values,
        )

    def evaluation_set(self, n_per_map: int, seed: int) -> TaskABatch:
        """A fixed held-out set: exactly ``n_per_map`` sequences per map, from its own seed."""
        generator = torch.Generator().manual_seed(seed)
        map_ids = torch.arange(self.cfg.n_maps).repeat_interleave(n_per_map)
        return self.sample(len(map_ids), generator, map_ids=map_ids)


def bayes_optimal_accuracy(task: TaskA, batch: TaskABatch, prior: Sequence[float] | None = None) -> float:
    """Expected accuracy of the best possible predictor that knows the maps.

    Averaged over the scored positions of ``batch``. Marked: the marker
    identifies the map, so the result is 1. Unmarked: the posterior over maps
    after the pairs seen so far, with ``prior`` (default: the configuration's
    map weights), gives a predictive distribution over the value; the best
    prediction is right with probability equal to its largest mass.
    """
    keys, values = batch.keys, batch.values
    n, pairs = keys.shape
    m = task.cfg.n_maps
    f_keys = task.maps[:, keys].permute(1, 2, 0)  # [n, P, M]: value each map gives each key
    if task.cfg.marked:
        posterior = F.one_hot(batch.map_ids, m).to(torch.float64).unsqueeze(1).expand(n, pairs, m)
    else:
        prior_w = task.default_weights if prior is None else task.normalize_weights(prior)
        consistent = (f_keys == values.unsqueeze(2)).to(torch.float64)
        seen_before = torch.cat((torch.ones(n, 1, m, dtype=torch.float64), consistent.cumprod(dim=1)[:, :-1]), dim=1)
        unnormalized = prior_w * seen_before
        posterior = unnormalized / unnormalized.sum(dim=2, keepdim=True)
    predictive = torch.zeros(n, pairs, task.cfg.n_symbols, dtype=torch.float64)
    predictive.scatter_add_(2, f_keys, posterior)
    best = predictive.max(dim=2).values
    return float(best[first_occurrence(keys, task.cfg.n_symbols)].mean())
