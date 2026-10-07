"""Task-independent training loop shared by the centralized baseline, simulated workers and Stage 1 clients."""

from cassandra_chorus.train.loop import OptimSection, TrainStats, lr_at, make_optimizer, train_steps

__all__ = ["OptimSection", "TrainStats", "lr_at", "make_optimizer", "train_steps"]
