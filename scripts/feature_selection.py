"""OOA-DPSO-GA hybrid feature selector (slow path — async worker only).
Runs Binary PSO with Latin Hypercube init, GA mutation, and single-point crossover."""

import numpy as np
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

W_MAX = 0.9  # inertia at start (exploration)
W_MIN = 0.4  # inertia at end   (exploitation)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _ts_cv_mae(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    """Time-series expanding-window CV MAE (no shuffle)."""
    n, fold_size, maes = len(X), len(X) // (n_folds + 1), []
    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        test_end  = min(train_end + fold_size, n)
        X_tr, y_tr = X[:train_end],         y[:train_end]
        X_te, y_te = X[train_end:test_end],  y[train_end:test_end]
        if len(X_tr) < 10 or len(X_te) < 5:
            continue
        model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, verbose=-1)
        model.fit(X_tr, y_tr)
        maes.append(mean_absolute_error(y_te, model.predict(X_te)))
    return float(np.mean(maes)) if maes else 1e9


def _fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray, sparsity_penalty: float = 0.05) -> float:
    """MAE + sparsity penalty. Lower is better."""
    selected = np.where(mask)[0]
    if len(selected) == 0:
        return 1e9
    return _ts_cv_mae(X[:, selected], y) + sparsity_penalty * len(selected) / X.shape[1]


def _lhs_binary_init(n_particles: int, n_feat: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling init: stratified, decorrelated binary positions."""
    strata = (np.arange(n_particles)[:, None] + rng.random((n_particles, n_feat))) / n_particles
    for j in range(n_feat):
        strata[:, j] = rng.permutation(strata[:, j])
    return (strata > 0.5).astype(float)


class OOADPSOGASelector:
    """Hybrid OOA-DPSO-GA feature selector (slow path — async worker only)."""

    def __init__(
        self,
        n_particles:  int   = 30,
        n_iterations: int   = 50,
        w_max:        float = W_MAX,
        w_min:        float = W_MIN,
        c1:           float = 1.5,
        c2:           float = 1.5,
        pc:           float = 0.8,
        pm_scale:     float = 1.0,
        sparsity_pen: float = 0.05,
        random_state: int   = 42,
    ):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w_max        = w_max
        self.w_min        = w_min
        self.c1           = c1
        self.c2           = c2
        self.pc           = pc
        self.pm_scale     = pm_scale
        self.sparsity_pen = sparsity_pen
        self.rng          = np.random.default_rng(random_state)

        self.mask_:              np.ndarray | None = None
        self.selected_features_: np.ndarray | None = None
        self.best_fitness_:      float = float("inf")
        self.history_:           list[float] = []

    def _w(self, it: int) -> float:
        return self.w_max - (self.w_max - self.w_min) * it / max(self.n_iterations - 1, 1)

    def _mutate(self, pos: np.ndarray, n_feat: int) -> np.ndarray:
        """Bit-flip mutation at rate pm_scale/n_features."""
        flip = self.rng.random((self.n_particles, n_feat)) < self.pm_scale / n_feat
        return np.abs(pos - flip)

    def _crossover(self, pos: np.ndarray, pbest: np.ndarray, n_feat: int) -> np.ndarray:
        """Single-point crossover between each particle and its personal best."""
        pos     = pos.copy()
        targets = self.rng.choice(self.n_particles, size=int(self.pc * self.n_particles), replace=False)
        for i in targets:
            k = int(self.rng.integers(1, n_feat))
            pos[i, k:] = pbest[i, k:]
        return pos

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OOADPSOGASelector":
        n_feat = X.shape[1]
        print(f"\n  OOA-DPSO-GA | particles={self.n_particles} "
              f"iters={self.n_iterations} features={n_feat}")
        print(f"  mutation p={self.pm_scale/n_feat:.4f} | crossover p={self.pc:.2f} | "
              f"inertia {self.w_max}→{self.w_min}\n")

        pos = _lhs_binary_init(self.n_particles, n_feat, self.rng)
        vel = self.rng.uniform(-1, 1, size=(self.n_particles, n_feat))

        pbest     = pos.copy()
        pbest_fit = np.array([_fitness(p.astype(bool), X, y, self.sparsity_pen) for p in pbest])
        gbest_idx = int(np.argmin(pbest_fit))
        gbest     = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        print(f"  [init]  best_fitness={gbest_fit:.4f}  features={int(gbest.sum())}/{n_feat}")

        for it in range(self.n_iterations):
            w  = self._w(it)
            r1 = self.rng.random((self.n_particles, n_feat))
            r2 = self.rng.random((self.n_particles, n_feat))

            vel = w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = (self.rng.random((self.n_particles, n_feat)) < _sigmoid(vel)).astype(float)
            pos = self._mutate(pos, n_feat)
            pos = self._crossover(pos, pbest, n_feat)

            fits              = np.array([_fitness(p.astype(bool), X, y, self.sparsity_pen) for p in pos])
            improved          = fits < pbest_fit
            pbest[improved]   = pos[improved]
            pbest_fit[improved] = fits[improved]

            gi = int(np.argmin(pbest_fit))
            if pbest_fit[gi] < gbest_fit:
                gbest     = pbest[gi].copy()
                gbest_fit = pbest_fit[gi]
            self.history_.append(gbest_fit)

            if (it + 1) % 5 == 0 or it == 0:
                print(f"  [iter {it+1:3d}/{self.n_iterations}]  w={w:.3f}  "
                      f"best_fitness={gbest_fit:.4f}  features={int(gbest.sum())}/{n_feat}")

        self.mask_              = gbest.astype(bool)
        self.selected_features_ = np.where(self.mask_)[0]
        self.best_fitness_      = gbest_fit
        print(f"\n  Converged: {len(self.selected_features_)}/{n_feat} features "
              f"({len(self.selected_features_)/n_feat*100:.1f}% kept) | fitness={gbest_fit:.4f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mask_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.mask_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
