"""The two credit models and their scores. Risk is the predicted probability of default; lower is better."""

import pickle

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier

from aa_gate.settings import APPROVAL_RATE, BLOCKS, CACHE, MONOTONE, SEED


def fit(kind: str, X: pd.DataFrame, y: pd.Series):
    """EBM without interaction terms, or XGBoost whose trees may combine features only within a block, so that both
    add up over the blocks exactly; both with the monotone constraints of settings. Fitted once, kept in
    cache/models/."""
    path = CACHE / "models" / f"{kind}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    mono = [MONOTONE.get(c, 0) for c in X.columns]
    if kind == "ebm":
        model = ExplainableBoostingClassifier(interactions=0, monotone_constraints=mono, random_state=SEED)
    elif kind == "xgb":
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=4, monotone_constraints=tuple(mono),
                              interaction_constraints=[list(cols) for cols in BLOCKS.values()])
    else:
        raise ValueError(kind)
    model.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    return model


def risk(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def logit(model, X: pd.DataFrame) -> np.ndarray:
    p = np.clip(risk(model, X), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def cutoff(model, X_train: pd.DataFrame) -> float:
    """The highest risk still approved: the APPROVAL_RATE quantile of the training applicants' risk."""
    return float(np.quantile(risk(model, X_train), APPROVAL_RATE))


def cuts(model, feature: str) -> np.ndarray:
    """The values of `feature` at which the model's output can change: XGBoost's split points, the EBM's bin edges.
    Between two of them the model is constant in that feature."""
    if isinstance(model, XGBClassifier):
        trees = model.get_booster().trees_to_dataframe()
        return np.unique(trees.loc[trees.Feature == feature, "Split"].to_numpy(float))
    return np.asarray(model.bins_[list(model.feature_names_in_).index(feature)][0], dtype=float)
