"""Ensemble utilities for XGBoost and CatBoost alongside LightGBM.

Provides fit/predict/save/load wrappers and hyperparameters for the three model
types.  The pipeline in train.py trains all three and blends predictions with
optimised per-stage weights.
"""
import gc
import json

import numpy as np

from config import TRAIN_DEVICE


# ── XGBoost hyperparameters ──────────────────────────────────────────────────
# Analogous to S1_PARAMS / S2_PARAMS in stack.py (LightGBM).

XGB_S1_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.15, max_depth=7, min_child_weight=200,
    colsample_bytree=0.8, subsample=0.7, reg_lambda=1.0,
    max_bin=128, tree_method='hist', verbosity=0,
)
XGB_S1_ROUNDS = 800

XGB_S2_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.08, max_depth=6, min_child_weight=200,
    colsample_bytree=0.8, subsample=0.7, reg_lambda=1.0,
    max_bin=128, tree_method='hist', verbosity=0,
)
XGB_S2_ROUNDS = 1000


# ── CatBoost hyperparameters ────────────────────────────────────────────────

CAT_S1_PARAMS = dict(
    loss_function='Logloss', learning_rate=0.15, depth=7,
    min_data_in_leaf=200, rsm=0.8, subsample=0.7,
    bootstrap_type='Bernoulli', l2_leaf_reg=1.0,
)
CAT_S1_ROUNDS = 800

CAT_S2_PARAMS = dict(
    loss_function='Logloss', learning_rate=0.08, depth=6,
    min_data_in_leaf=200, rsm=0.8, subsample=0.7,
    bootstrap_type='Bernoulli', l2_leaf_reg=1.0,
)
CAT_S2_ROUNDS = 1000


# ── XGBoost fit / predict / save / load ─────────────────────────────────────

def fit_xgb(params, X, y, Xv, yv, rounds, n_jobs, seed):
    """Train an XGBoost model with early stopping."""
    import xgboost as xgb
    params = dict(params, nthread=n_jobs, seed=seed)
    if TRAIN_DEVICE == 'cuda':
        params['device'] = 'cuda'
    dtrain = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(Xv, label=yv)
    model = xgb.train(params, dtrain, num_boost_round=rounds,
                      evals=[(dval, 'val')], early_stopping_rounds=50,
                      verbose_eval=100)
    del dtrain, dval
    gc.collect()
    return model


def predict_xgb(model, X, n_threads=1):
    """Predict probability with an XGBoost Booster."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X, nthread=n_threads)
    best_it = getattr(model, 'best_iteration', None)
    if best_it is not None:
        p = model.predict(dmat, iteration_range=(0, best_it + 1))
    else:
        p = model.predict(dmat)
    del dmat
    return p.astype(np.float32)


def save_xgb(model, path):
    """Save XGBoost model + its best_iteration metadata."""
    model.save_model(str(path))
    with open(str(path) + '.meta', 'w') as f:
        json.dump({'best_iteration': int(model.best_iteration)}, f)


def load_xgb(path):
    """Load an XGBoost Booster and restore best_iteration."""
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(path))
    try:
        with open(str(path) + '.meta') as f:
            model.best_iteration = json.load(f)['best_iteration']
    except FileNotFoundError:
        pass
    return model


# ── CatBoost fit / predict / save / load ────────────────────────────────────

def fit_cat(params, X, y, Xv, yv, rounds, n_jobs, seed):
    """Train a CatBoost classifier with early stopping."""
    from catboost import CatBoostClassifier, Pool
    params = dict(params)
    if TRAIN_DEVICE == 'cuda':
        params.pop('rsm', None)
        params.pop('min_data_in_leaf', None)
    model = CatBoostClassifier(
        iterations=rounds, early_stopping_rounds=50,
        thread_count=n_jobs, random_seed=seed, verbose=100,
        task_type='GPU' if TRAIN_DEVICE == 'cuda' else 'CPU',
        **params
    )
    eval_pool = Pool(Xv, yv)
    model.fit(X, y, eval_set=eval_pool)
    del eval_pool
    gc.collect()
    return model


def predict_cat(model, X, n_threads=1):
    """Predict probability with a CatBoost classifier."""
    return model.predict_proba(X, thread_count=n_threads)[:, 1].astype(np.float32)


def save_cat(model, path):
    model.save_model(str(path))


def load_cat(path):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


# ── Blending ────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {'lgb': 0.40, 'xgb': 0.35, 'cat': 0.25}


def blend(preds, weights=None):
    """Weighted average of prediction arrays.
    preds:   dict  model_type -> np.ndarray of probabilities
    weights: dict  model_type -> float (auto-normalised)
    """
    if weights is None:
        weights = {k: DEFAULT_WEIGHTS.get(k, 1.0) for k in preds}
    total = sum(weights[k] for k in preds)
    return sum(weights[k] / total * preds[k] for k in preds).astype(np.float32)
