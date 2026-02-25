import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb


def _ts_splits(n: int, n_folds: int = 5):
    fold_size = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        val_end   = min(train_end + fold_size, n)
        if train_end < 10 or val_end - train_end < 5:
            continue
        yield np.arange(train_end), np.arange(train_end, val_end)


def _make_base_learners():
    return {
        "lgbm":        lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                          num_leaves=31, subsample=0.8,
                                          colsample_bytree=0.8, verbose=-1),
        "ridge":       Ridge(alpha=10.0),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=4,
                                           n_jobs=-1, random_state=42),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
    }


def _clone_model(name: str, template):
    if name == "lgbm":
        return lgb.LGBMRegressor(**template.get_params())
    if name == "ridge":
        return Ridge(**template.get_params())
    if name == "extra_trees":
        params = template.get_params()
        params.pop("estimator", None)
        params.pop("estimators_", None)
        return ExtraTreesRegressor(**params)
    if name == "elastic_net":
        return ElasticNet(**template.get_params())
    raise ValueError(f"Unknown learner: {name}")


class BiStackingEnsemble:
    """Two-level stacking ensemble: L0 (LightGBM, Ridge, ExtraTrees, ElasticNet)
    → OOF stack → L1 LightGBM meta-learner. Expanding-window CV prevents leakage."""

    def __init__(self, n_folds: int = 5, meta_leaves: int = 15):
        self.n_folds            = n_folds
        self.meta_leaves        = meta_leaves
        self._base_learners:    dict = {}
        self._meta_learner      = None
        self._learner_names:    list[str] = []
        self.oof_predictions_:  np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BiStackingEnsemble":
        n    = len(X)
        base = _make_base_learners()
        self._learner_names = list(base.keys())

        oof     = np.zeros((n, len(self._learner_names)))
        splits  = list(_ts_splits(n, self.n_folds))
        if not splits:
            raise ValueError(f"Not enough samples ({n}) to form {self.n_folds} CV folds.")

        covered = np.zeros(n, dtype=bool)
        for tr_idx, val_idx in splits:
            for col, (name, model) in enumerate(base.items()):
                clone = _clone_model(name, model)
                clone.fit(X[tr_idx], y[tr_idx])
                oof[val_idx, col] = clone.predict(X[val_idx])
            covered[val_idx] = True

        if not covered.all():
            oof[~covered] = oof[covered].mean(axis=0)
        self.oof_predictions_ = oof

        for name, model in base.items():
            model.fit(X, y)
            self._base_learners[name] = model

        self._meta_learner = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=self.meta_leaves,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        self._meta_learner.fit(oof[covered], y[covered])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._meta_learner is None:
            raise RuntimeError("Call fit() before predict().")
        return self._meta_learner.predict(self._l0_predict(X))

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        l0 = self._l0_predict(X)
        return {name: l0[:, col] for col, name in enumerate(self._learner_names)}

    def _l0_predict(self, X: np.ndarray) -> np.ndarray:
        if not self._base_learners:
            raise RuntimeError("Call fit() before predict().")
        return np.column_stack([self._base_learners[n].predict(X) for n in self._learner_names])
