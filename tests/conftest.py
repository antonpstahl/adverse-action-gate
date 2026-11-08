"""A bank on synthetic data with a logistic regression, whose exact values can be computed by hand."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aa_gate.model import cutoff, risk
from aa_gate.queries import Bank
from aa_gate.reference import groups
from aa_gate.settings import FEATURES


@pytest.fixture(scope="session")
def train() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(3000, len(FEATURES))), columns=FEATURES)


@pytest.fixture(scope="session")
def bank(train) -> Bank:
    rng = np.random.default_rng(1)
    y = (train.to_numpy() @ rng.normal(size=len(FEATURES)) + rng.logistic(size=len(train)) > 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(train, y)
    cut = cutoff(model, train)
    declined = train[risk(model, train) > cut].head(40)
    return Bank("test", model, cut, groups(model, train, cut), train.min(), train.max(), declined)
