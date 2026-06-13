from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import corruption_aware_batch
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import FAULT_SPECS
from src.models.strong_backbones_v3 import SRAFOfficialStyleSTIDWrapperFactorAblation
from src.protocols.matched_protocol import (
    MATCHED_TRAIN_FAULTS,
    TEST_FAULTS,
    training_fault_for_step,
    training_fault_seed,
)


class CaptureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x[:, -1:, :, :1]


def sample_batch() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 12, 5, 3)).astype(np.float32)
    x[..., 1:] = rng.uniform(size=(4, 12, 5, 2)).astype(np.float32)
    y = rng.normal(size=(4, 12, 5, 1)).astype(np.float32)
    return x, y


def test_shared_fault_rotation_and_rm20_boundary() -> None:
    expected = list(MATCHED_TRAIN_FAULTS) * 3
    actual = [training_fault_for_step(step) for step in range(len(expected))]
    assert actual == expected
    assert "random_missing_20" not in MATCHED_TRAIN_FAULTS
    assert "random_missing_20" in TEST_FAULTS


def test_same_seed_and_batch_produce_identical_corruption() -> None:
    x, _ = sample_batch()
    for step, label in enumerate(MATCHED_TRAIN_FAULTS):
        setting = FAULT_SPECS[label]
        seed = training_fault_seed(42, step)
        a = corruption_aware_batch(x, setting, seed)
        b = corruption_aware_batch(x, setting, seed)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)


def test_fault_generator_does_not_modify_clean_target() -> None:
    x, y = sample_batch()
    y_before = y.copy()
    corruption_aware_batch(x, FAULT_SPECS["random_missing_40"], 42)
    np.testing.assert_array_equal(y, y_before)


def test_finite_faults_remain_observed() -> None:
    x, _ = sample_batch()
    for label in ("gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"):
        corrupted, _, observed = corruption_aware_batch(x, FAULT_SPECS[label], 42)
        assert np.isfinite(corrupted[..., :1]).all()
        assert np.all(observed == 1.0)


def test_finite_fault_inference_has_no_fault_mask_argument() -> None:
    signature = inspect.signature(SRAFOfficialStyleSTIDWrapperFactorAblation.forward)
    assert "fault_mask" not in signature.parameters
    assert "m_fault" not in signature.parameters


def test_two_way_softmax_and_identity_feature_bypass() -> None:
    x, _ = sample_batch()
    x_t = torch.from_numpy(x[:2])
    backbone = CaptureBackbone()
    model = SRAFOfficialStyleSTIDWrapperFactorAblation(
        sensors=5,
        backbone=backbone,
        tod_profile=torch.zeros((288, 5, 1)),
        temporal_mode="basic",
        spatial_mode="adjacency",
        fusion_mode="softmax",
        use_profile=False,
        observed_input_blend=0.5,
    )
    observed = torch.ones_like(x_t[..., :1])
    _, components = model(
        x_t,
        adjacency=torch.eye(5),
        observed_mask=observed,
        return_components=True,
    )
    weights = components["candidate_weights"]
    assert weights.shape[-1] == 2
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    torch.testing.assert_close(components["backbone_input"][..., 1:], x_t[..., 1:])
