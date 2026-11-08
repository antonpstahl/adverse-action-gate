"""The exact least action and the constraints, on the study's own models (decisions 23–25, 27)."""

import numpy as np
import pandas as pd
import pytest

from aa_gate import actions
from aa_gate.model import logit, risk
from aa_gate.queries import applicant_risk, build, least, run
from aa_gate.reference import shapley
from aa_gate.settings import BLOCKS, SEED


@pytest.fixture(scope="module", params=["xgb", "ebm"])
def bank(request):
    return build(request.param)


def _every_amount(bank, x):
    """Brute force, without cut points: the risk after paying every whole amount from 1 to the end of the data."""
    top = int(np.floor(min(bank.hi["PAY_AMT1"] - x["PAY_AMT1"], x["BILL_AMT1"] - bank.lo["BILL_AMT1"])))
    amounts = np.arange(1, top + 1, dtype=float)
    frame = pd.DataFrame(np.repeat(x.to_numpy()[None, :], len(amounts), axis=0), columns=x.index)
    frame["PAY_AMT1"] += amounts
    frame["BILL_AMT1"] -= amounts
    return amounts, risk(bank.model, frame)


def test_the_least_payment_equals_brute_force_and_paying_more_never_raises_the_risk(bank):
    for _, x in bank.declined.sample(3, random_state=SEED).iterrows():
        amounts, r = _every_amount(bank, x)
        ok = r <= bank.cutoff
        assert least(bank, x, "pay_more")["least"] == (int(amounts[np.argmax(ok)]) if ok.any() else None)
        assert np.all(np.diff(r) <= 1e-9)


def test_the_fewest_months_on_time_equal_brute_force(bank):
    for _, x in bank.declined.sample(3, random_state=SEED).iterrows():
        approved = [k for k in range(1, 7) if (y := actions.apply(x, "on_time", k, bank.lo, bank.hi)) is not None
                    and applicant_risk(bank, y) <= bank.cutoff]
        assert least(bank, x, "on_time")["least"] == (approved[0] if approved else None)


def test_the_least_payment_after_months_on_time_equals_brute_force_and_no_way_means_not_even_together(bank):
    """Decision 43: both is k months on time, then the extra payment; its least is 0 if the months are enough alone.
    For an applicant whom neither action gets approved alone, "neither alone" is true and "no way" is not."""
    x = bank.declined.sample(1, random_state=SEED).iloc[0]
    for k in range(1, 7):
        y = actions.apply(x, "on_time", k, bank.lo, bank.hi)
        if y is None:
            assert least(bank, x, "both", k)["least"] is None
            continue
        if applicant_risk(bank, y) <= bank.cutoff:  # the months alone are enough, and "0 more" is a valid both
            assert least(bank, x, "both", k)["least"] == 0
            assert run(bank, x, {"type": "action", "action": "both", "months": k, "amount": 0,
                                 "decision": "approved"}) is True
            continue
        amounts, r = _every_amount(bank, y)
        ok = r <= bank.cutoff
        assert least(bank, x, "both", k)["least"] == (int(amounts[np.argmax(ok)]) if ok.any() else None)
    stuck = next(x for _, x in bank.declined.sample(300, random_state=SEED).iterrows()
                 if least(bank, x, "on_time")["least"] is None and least(bank, x, "pay_more")["least"] is None)
    q = {"type": "least", "decision": "declined"}
    assert run(bank, stuck, {**q, "action": "on_time"}) is True and run(bank, stuck, {**q, "action": "pay_more"}) is True
    assert run(bank, stuck, {**q, "action": "both"}) is False
    assert run(bank, stuck, {"type": "least", "action": "both", "months": 7, "value": 500, "decimals": 0}) is None


def test_both_models_add_up_over_the_blocks(bank):
    """No interactions between blocks: each block's Shapley value is its effect when moved alone."""
    x, ref = bank.declined.iloc[0], bank.refs[1]
    base = logit(bank.model, ref).mean()
    alone = {}
    for b, cols in BLOCKS.items():
        z = ref.copy()
        z.loc[:, cols] = np.tile(x[cols].to_numpy(), (len(z), 1))
        alone[b] = logit(bank.model, z).mean() - base
    assert np.allclose(shapley(bank.model, x, ref), pd.Series(alone), atol=1e-4)


def test_a_stated_least_payment_is_true_only_if_it_is_never_below_the_true_least(bank):
    """The stated least must be the true least rounded up at the precision written (decision 29): rounded down it
    would not be enough. With a least of 20,564, "20,570" is true and "20,560" false."""
    for _, x in bank.declined.sample(20, random_state=SEED).iterrows():
        best = least(bank, x, "pay_more")["least"]
        if best is None or best < 100:
            continue
        query = {"type": "least", "action": "pay_more"}
        assert run(bank, x, {**query, "value": best, "decimals": 0}) is True
        assert run(bank, x, {**query, "value": best - 1, "decimals": 0}) is False
        up, down = np.ceil(best / 10) * 10, np.floor(best / 10) * 10
        if up != best:
            assert run(bank, x, {**query, "value": up, "decimals": -1}) is True
            assert run(bank, x, {**query, "value": down, "decimals": -1}) is False
        return
    pytest.skip("no applicant with a least payment of at least NT$100 in the sample")
