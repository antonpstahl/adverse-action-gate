"""TaiwanCredit without SEX, MARRIAGE and AGE (ECOA prohibited bases, and age) and without EDUCATION (decision 31),
split once. Only the features of settings.BLOCKS are loaded."""

from functools import cache

import pandas as pd
from sklearn.model_selection import train_test_split

from aa_gate.settings import DATA, FEATURES, SEED, TEST_SHARE

# Payment status codes are categories below 1 and months late from 1 (UCI documents only -1 and 1-9). The model uses
# the codes; statements use the kinds and months late, never a mean of codes (decisions 28, 33).
KINDS = {-2: "card not used", -1: "paid in full", 0: "paid on time"}


def kind(code) -> str:
    return KINDS.get(int(code), "late")


def label(code) -> str:
    return KINDS.get(int(code), f"{int(code)} month{'s' if int(code) > 1 else ''} late")


# What each column means, for the writer's tools and the reader's prompt (UCI dataset 350; amounts in NT dollars).
DESCRIPTIONS = {
    "LIMIT_BAL": "credit limit",
    **{c: f"payment status {m}: card not used, paid in full, paid on time, or k months late"
       for c, m in zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
                       ["Sep", "Aug", "Jul", "Jun", "May", "Apr"])},
    **{f"BILL_AMT{i}": f"statement balance {m}" for i, m in enumerate(["Sep", "Aug", "Jul", "Jun", "May", "Apr"], 1)},
    **{f"PAY_AMT{i}": f"amount paid {m}" for i, m in enumerate(["Sep", "Aug", "Jul", "Jun", "May", "Apr"], 1)},
}


@cache
def load() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_excel(DATA, header=1)
    return raw[FEATURES].astype(float), raw["default payment next month"].rename("default")


@cache
def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train and test rows, stratified on the label. Index labels are the rows of the source file."""
    X, y = load()
    return train_test_split(X, y, test_size=TEST_SHARE, stratify=y, random_state=SEED)
