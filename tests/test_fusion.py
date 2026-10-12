"""Smoke tests for the fusion and hard-switch baselines."""
import numpy as np
import pandas as pd

from crowdsafe.counting.fusion import Fusion, fit_hard_switch, hard_switch


def _synthetic(n: int = 300, seed: int = 42):
    """Detector is exact when sparse (low overlap) and undercounts when crowded; density is noisy but unbiased."""
    rng = np.random.default_rng(seed)
    gt = rng.integers(2, 60, n).astype(float)
    overlap = np.clip(gt / 60 + rng.normal(0, 0.05, n), 0, 1)
    det = np.where(gt < 25, gt, gt * (1 - 0.5 * overlap))
    dens = gt + rng.normal(0, 4, n)
    feats = np.column_stack([0.8 - 0.3 * overlap, det, overlap, 40 - 10 * overlap, dens])
    return feats, det, dens, gt


def test_fusion_beats_both_inputs_on_synthetic_data() -> None:
    feats, det, dens, gt = _synthetic()
    f = Fusion().fit(feats[:200], det[:200], dens[:200], gt[:200])
    fused = f(feats[200:], det[200:], dens[200:])
    mae = lambda p: np.mean(np.abs(p - gt[200:]))
    assert mae(fused) < mae(det[200:]) and mae(fused) < mae(dens[200:])
    w = f.weights(feats[200:])
    assert ((w >= 0) & (w <= 1)).all()


def test_hard_switch_threshold() -> None:
    feats, det, dens, gt = _synthetic()
    tau = fit_hard_switch(det, dens, gt)
    hs = hard_switch(det, dens, tau)
    assert np.mean(np.abs(hs - gt)) <= min(np.mean(np.abs(det - gt)), np.mean(np.abs(dens - gt)))


def test_time_block_split_never_interleaves() -> None:
    from scripts.fusion_ablation import split_time_blocks

    df = pd.DataFrame({"camera": ["cam01"] * 6 + ["cam02"] * 2, "t_s": [50, 10, 40, 20, 30, 60, 5, 6]})
    s = split_time_blocks(df, "cam01", 0.5)
    assert set(df.loc[s == "val", "t_s"]) == {10, 20, 30} and (s[df.camera == "cam02"] == "test").all()
