"""Mixture-of-experts transformer with a maskable router (PLAN step 2; SRS S0-F-01 to S0-F-03)."""

from cassandra_chorus.model.moe import MoELayer, MoEResult, switch_balance_loss
from cassandra_chorus.model.transformer import (
    ModelOutput,
    ModelSection,
    MoETransformer,
    count_parameters,
    expected_parameter_count,
    expert_param_owner,
    normalize_held,
    router_param_layer,
    validate_model_config,
)

__all__ = [
    "ModelOutput",
    "ModelSection",
    "MoELayer",
    "MoEResult",
    "MoETransformer",
    "count_parameters",
    "expected_parameter_count",
    "expert_param_owner",
    "normalize_held",
    "router_param_layer",
    "switch_balance_loss",
    "validate_model_config",
]
