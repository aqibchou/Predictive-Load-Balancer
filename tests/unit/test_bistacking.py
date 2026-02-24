import os
import sys

import numpy as np
import pytest

try:
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip("lightgbm not available (architecture or install issue)", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../load_balancer"))

from bistacking import BiStackingEnsemble, _ts_splits


def make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, 6)).astype(np.float32)
    # linear relationship so a stacking ensemble can capture it
    y   = (2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] +
           rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


# ── _ts_splits ────────────────────────────────────────────────────────────────

def test_ts_splits_no_data_leakage():
    for tr_idx, val_idx in _ts_splits(100, n_folds=5):
        assert tr_idx.max() < val_idx.min()


def test_ts_splits_at_least_one_fold():
    splits = list(_ts_splits(100, n_folds=5))
    assert len(splits) >= 1


def test_ts_splits_small_n_returns_no_folds():
    # n=10, fold_size=1 → train_end=1 < 10 minimum → no folds
    splits = list(_ts_splits(10, n_folds=5))
    assert len(splits) == 0


# ── BiStackingEnsemble ────────────────────────────────────────────────────────

def test_fit_predict_output_shape():
    X, y = make_dataset(150)
    m    = BiStackingEnsemble(n_folds=3)
    m.fit(X, y)
    assert m.predict(X[-10:]).shape == (10,)


def test_predict_proba_returns_all_learners():
    X, y = make_dataset(150)
    m    = BiStackingEnsemble(n_folds=3)
    m.fit(X, y)
    proba = m.predict_proba(X[-5:])
    assert set(proba.keys()) == {"lgbm", "ridge", "extra_trees", "elastic_net"}
    for v in proba.values():
        assert v.shape == (5,)


def test_predict_before_fit_raises():
    m = BiStackingEnsemble()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(np.ones((3, 4), dtype=np.float32))


def test_oof_predictions_shape():
    X, y = make_dataset(150)
    m    = BiStackingEnsemble(n_folds=3)
    m.fit(X, y)
    assert m.oof_predictions_ is not None
    assert m.oof_predictions_.shape == (len(X), 4)   # 4 base learners


def test_predictions_correlated_with_target():
    """Ensemble should capture the linear DGP — Pearson r > 0.9."""
    X, y = make_dataset(400)
    m    = BiStackingEnsemble(n_folds=5)
    m.fit(X[:-50], y[:-50])
    preds = m.predict(X[-50:])
    r     = float(np.corrcoef(preds, y[-50:])[0, 1])
    assert r > 0.9


def test_fit_returns_self():
    X, y = make_dataset(150)
    m    = BiStackingEnsemble(n_folds=3)
    ret  = m.fit(X, y)
    assert ret is m


def test_meta_learner_uses_oof_stack():
    """After fit, meta-learner input dimension equals number of base learners (4)."""
    X, y = make_dataset(150)
    m    = BiStackingEnsemble(n_folds=3)
    m.fit(X, y)
    # meta-learner was trained on (n_covered, 4) OOF stack
    assert m._meta_learner is not None
    # calling predict on a (1, 4) matrix should work
    fake_stack = np.zeros((1, 4), dtype=np.float32)
    result = m._meta_learner.predict(fake_stack)
    assert result.shape == (1,)
