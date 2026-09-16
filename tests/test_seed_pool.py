import numpy as np

from dot_pe.mp_inference import restrict_seed_pool


def test_restrict_seed_pool_keeps_top_survivors_per_bank():
    inds = {"bank_0": np.arange(5), "bank_1": np.arange(3), "bank_2": np.arange(0)}
    lnl = {
        "bank_0": np.array([10.0, 30.0, 25.0, 5.0, 29.0]),
        "bank_1": np.array([28.0, 1.0, 31.0]),
        "bank_2": np.array([]),
    }
    out = restrict_seed_pool(inds, lnl, lnlike_drop=3.0, min_pool=1)
    assert set(out) == {"bank_0", "bank_1", "bank_2"}
    assert out["bank_0"].tolist() == [1, 4]  # 30, 29 >= 31 - 3
    assert out["bank_1"].tolist() == [0, 2]  # 28, 31
    assert out["bank_2"].size == 0


def test_restrict_seed_pool_falls_back_to_top_min_pool():
    inds = {"bank_0": np.arange(6)}
    lnl = {"bank_0": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])}
    out = restrict_seed_pool(inds, lnl, lnlike_drop=1.0, min_pool=3)
    assert out["bank_0"].tolist() == [3, 4, 5]
