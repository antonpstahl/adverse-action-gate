"""The reference and the action layer against values computed by hand (plan §3.8)."""

from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from aa_gate import actions
from aa_gate.model import logit, risk
from aa_gate.reference import reasons, shapley
from aa_gate.settings import APPROVAL_RATE, BAND, BILLS, BLOCKS, FEATURES, MAX_REASONS, PAYMENTS, STATUS


def test_shapley_of_a_linear_model_is_coefficient_times_difference_from_the_reference(bank):
    """For a logistic regression the log-odds are linear, so a block's exact Shapley value is the sum over its
    features of coefficient times (applicant value - reference mean)."""
    coef = pd.Series(bank.model.coef_[0], index=FEATURES)
    x = bank.declined.iloc[0]
    for ref in bank.refs.values():
        expected = pd.Series({b: float((coef[c] * (x[c] - ref[c].mean())).sum()) for b, c in BLOCKS.items()})
        assert np.allclose(shapley(bank.model, x, ref), expected, atol=1e-8)


def test_shapley_matches_the_average_over_all_orders_for_a_model_with_interactions(bank):
    """An independent computation: the average marginal contribution over every order of the blocks."""
    X = pd.concat([bank.refs[2], bank.declined])
    model = GradientBoostingClassifier(max_depth=3, n_estimators=30, random_state=0).fit(X, risk(bank.model, X) > 0.5)
    x, ref = bank.declined.iloc[1], bank.refs[2].head(100)

    def value(coalition):
        z = ref.copy()
        for b in coalition:
            z.loc[:, BLOCKS[b]] = np.tile(x[BLOCKS[b]].to_numpy(), (len(z), 1))
        return logit(model, z).mean()

    total = dict.fromkeys(BLOCKS, 0.0)
    orders = list(permutations(BLOCKS))
    for order in orders:
        for k, b in enumerate(order):
            total[b] += value(order[: k + 1]) - value(order[:k])
    phi = shapley(model, x, ref)
    assert np.allclose(phi, pd.Series(total) / len(orders), atol=1e-9)
    assert np.isclose(phi.sum(), logit(model, x.to_frame().T)[0] - logit(model, ref).mean())


def test_method_1_holds_the_approved_applicants_just_inside_the_cutoff(bank, train):
    r = risk(bank.model, bank.refs[1])
    assert (r <= bank.cutoff).all()
    assert (r >= np.quantile(risk(bank.model, train), APPROVAL_RATE - BAND)).all()
    assert bank.refs[2].index.isin(train.index).all()


def test_reasons_are_the_positive_shortfalls_largest_first_at_most_four():
    phi = pd.Series({"a": 0.2, "b": -0.1, "c": 0.5, "d": 0.0, "e": 0.3, "f": 0.1})
    assert reasons(phi) == ["c", "e", "a", "f"]
    assert len(reasons(phi)) == MAX_REASONS
    assert reasons(pd.Series({"a": -1.0})) == []


def _x() -> pd.Series:
    x = pd.Series(0.0, index=FEATURES)
    x[STATUS] = [3, 2, 1, 0, -1, -2]
    x[BILLS] = [10, 20, 30, 40, 50, 60]
    x[PAYMENTS] = [1, 2, 3, 4, 5, 6]
    return x


def test_on_time_moves_each_monthly_series_back():
    y = actions.on_time(_x(), 2)
    assert list(y[STATUS]) == [0, 0, 3, 2, 1, 0]
    assert list(y[BILLS]) == [10, 10, 10, 20, 30, 40]
    assert list(y[PAYMENTS]) == [4, 4, 1, 2, 3, 4]  # the new months pay the six-month average, 3.5 (decision 43)
    assert all(actions.on_time(_x(), m) is None for m in (0, 7, 2.5))


def test_both_pays_on_time_and_then_more_in_the_last_month():
    y = actions.both(_x(), 4, 2)
    assert list(y[STATUS]) == [0, 0, 3, 2, 1, 0]
    assert list(y[PAYMENTS]) == [8, 4, 1, 2, 3, 4]
    assert list(y[BILLS]) == [6, 10, 10, 20, 30, 40]
    x = _x()
    assert actions.apply(x, "both", 4, x - 100, x + 100) is None  # without the months


def test_pay_more_raises_the_payment_and_lowers_the_balance_into_a_credit_balance():
    y = actions.pay_more(_x(), 4)
    assert (y["PAY_AMT1"], y["BILL_AMT1"]) == (5, 6)
    assert actions.pay_more(_x(), 15)["BILL_AMT1"] == -5
    assert actions.pay_more(_x(), 0) is None


def test_an_action_outside_the_observed_range_is_not_possible():
    x = _x()
    lo, hi = x - 100, x + 100
    assert actions.apply(x, "pay_more", 50, lo, hi) is not None
    assert actions.apply(x, "pay_more", 500, lo, hi) is None
    assert actions.apply(x, "move_house", 1, lo, hi) is None
