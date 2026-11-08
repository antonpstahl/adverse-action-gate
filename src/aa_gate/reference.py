"""Reg B's reference groups and the shortfall of each block (plan §3.5): its exact Shapley value in the log-odds gap
between the applicant and the reference group, with the blocks as players. A positive shortfall pushed the applicant
toward a decline."""

from math import factorial

import numpy as np
import pandas as pd

from aa_gate.model import logit, risk
from aa_gate.settings import APPROVAL_RATE, BAND, BLOCKS, MAX_REASONS, N_REF, SEED


def groups(model, X_train: pd.DataFrame, cut: float) -> dict[int, pd.DataFrame]:
    """Comment 9(b)(2)-5. Method 1: the approved applicants just inside the cutoff, whose total score was "at or
    slightly above the minimum passing score". Method 2: all applicants. N_REF rows of each, drawn once."""
    r = risk(model, X_train)
    near = X_train[(r >= np.quantile(r, APPROVAL_RATE - BAND)) & (r <= cut)]
    return {m: X.sample(n=min(N_REF, len(X)), random_state=SEED) for m, X in ((1, near), (2, X_train))}


def shapley(model, x: pd.Series, ref: pd.DataFrame, blocks: dict = BLOCKS) -> pd.Series:
    """Exact: the value of a coalition of blocks is the mean log-odds over the reference rows with those blocks set
    to the applicant's values; all 2^n coalitions are scored in one call. The values add up to the applicant's
    log-odds minus the reference rows' mean log-odds."""
    names = list(blocks)
    n = len(names)
    cols = [[ref.columns.get_loc(c) for c in blocks[b]] for b in names]
    base, xv = ref.to_numpy(dtype=float), x[ref.columns].to_numpy(dtype=float)
    rows = []
    for mask in range(2**n):
        z = base.copy()
        for i in range(n):
            if mask >> i & 1:
                z[:, cols[i]] = xv[cols[i]]
        rows.append(z)
    v = logit(model, pd.DataFrame(np.vstack(rows), columns=ref.columns)).reshape(2**n, len(ref)).mean(axis=1)
    phi = np.zeros(n)
    for mask in range(2**n):
        size = bin(mask).count("1")
        for i in range(n):
            if not mask >> i & 1:
                phi[i] += factorial(size) * factorial(n - size - 1) / factorial(n) * (v[mask | 1 << i] - v[mask])
    return pd.Series(phi, index=names)


def reasons(phi: pd.Series) -> list[str]:
    """The principal reasons: blocks with a positive shortfall, largest first, at most MAX_REASONS."""
    return list(phi[phi > 0].sort_values(ascending=False).index[:MAX_REASONS])
