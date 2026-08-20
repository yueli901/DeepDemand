"""Component ablations of DeepDemand.

Each class disables exactly one design choice so that its contribution can be measured on the
same protocol as the full model. They are selected at runtime through the MODEL_VARIANT
environment variable (full | shared_encoder | constant_deterrence | linear_aggregation) and
reuse the entire training pipeline, so the "full" variant is the reported model.

  AblationSharedEncoder      single shared origin/destination encoder in place of two towers
  AblationConstantDeterrence travel-time deterrence fixed to 1
  AblationLinearAggregation  linear final transform in place of the square root
"""

import torch
import torch.nn as nn

from model.deepdemand import DeepDemand


class _ConstantOne(nn.Module):
    """Deterrence stub returning 1 for every travel time."""
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(d)


class AblationSharedEncoder(DeepDemand):
    """Tie the destination encoder to the origin encoder, removing the origin/destination
    asymmetry."""
    def __init__(self, feature_bank=None, node_to_lsoa=None, od_cache=None, feature_matrix=None):
        super().__init__(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa,
                         od_cache=od_cache, feature_matrix=feature_matrix, shared_encoder=True)


class AblationConstantDeterrence(DeepDemand):
    """Replace the learned travel-time deterrence with a constant."""
    def __init__(self, feature_bank=None, node_to_lsoa=None, od_cache=None, feature_matrix=None):
        super().__init__(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa,
                         od_cache=od_cache, feature_matrix=feature_matrix)
        self.dist_head = _ConstantOne()


class AblationLinearAggregation(DeepDemand):
    """Replace the monotonic square-root transform with a linear one. Overriding _aggregate
    keeps every forward path consistent."""
    GAMMA = 1.0

    def _aggregate(self, S, eps=0.0):
        return self.GAMMA * S
