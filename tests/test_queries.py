"""The queries and the gate's verdicts against values computed by hand (plan §3.8)."""

from aa_gate import actions
from aa_gate.gate import check, stated_reasons
from aa_gate.queries import DEPENDS, QUERY_FIELDS, applicant_risk, combine, complete, number, run, shortfalls
from aa_gate.reference import reasons
from aa_gate.settings import BLOCKS, STATUS


def q(**fields) -> dict:
    return {**dict.fromkeys(QUERY_FIELDS), **fields}


def test_a_number_is_true_at_the_precision_written():
    assert number(0.31, 2, 0.305) and number(0.31, 2, 0.3149)
    assert not number(0.31, 2, 0.3151)
    assert number(26542, 0, 26542.5) and number(26543, 0, 26542.5)  # halfway rounds either way (decision 56)
    assert number(150000, -4, 149873) and not number(150000, -4, 155001)
    assert number(0.31, None, 0.31) is None


def test_methods_that_disagree_make_a_claim_reference_dependent():
    assert combine(True, True) is True and combine(False, False) is False
    assert combine(True, False) == DEPENDS
    assert combine(None, True) is None


def test_outcome(bank):
    x = bank.declined.iloc[0]
    r = applicant_risk(bank, x)
    assert run(bank, x, q(type="outcome", quantity="decision", decision="declined")) is True
    assert run(bank, x, q(type="outcome", quantity="decision", decision="approved")) is False
    assert run(bank, x, q(type="outcome", quantity="risk", value=round(r, 3), decimals=3)) is True
    assert run(bank, x, q(type="outcome", quantity="risk", value=round(r, 3) + 0.002, decimals=3)) is False
    assert run(bank, x, q(type="outcome", quantity="gap", value=round(r - bank.cutoff, 2), decimals=2)) is True
    assert run(bank, x, q(type="outcome", quantity="risk")) is None


def test_action(bank):
    x = bank.declined.iloc[0]
    y = actions.apply(x, "on_time", 3, bank.lo, bank.hi)
    after, before = applicant_risk(bank, y), applicant_risk(bank, x)
    assert run(bank, x, q(type="action", action="on_time", amount=3, decision="approved")) == (after <= bank.cutoff)
    assert run(bank, x, q(type="action", action="on_time", amount=3, direction="lower")) == (after < before)
    assert run(bank, x, q(type="action", action="on_time", amount=3, quantity="risk", value=round(after, 3),
                          decimals=3)) is True
    assert run(bank, x, q(type="action", action="on_time", amount=9, decision="approved")) is None


def test_reasons_and_comparisons_follow_both_reference_groups_and_measurements_their_named_group(bank):
    x = bank.declined.iloc[2]
    phi1, phi2 = shortfalls(bank, x, 1), shortfalls(bank, x, 2)
    top1, top2 = reasons(phi1), reasons(phi2)
    for b in BLOCKS:  # with four blocks, a principal reason is exactly a block that raised the risk (decision 36)
        assert (b in top1, b in top2) == (bool(phi1[b] > 0), bool(phi2[b] > 0))
        assert run(bank, x, q(type="reason", block=b, direction="higher")) == combine(bool(phi1[b] > 0),
                                                                                     bool(phi2[b] > 0))
    assert run(bank, x, q(type="reason", block=top1[0], rank=1)) == combine(True, top2[0] == top1[0])
    last = phi1.sort_values().index[0]  # the block that lowered the risk most ranks last
    assert run(bank, x, q(type="reason", block=last, rank=len(BLOCKS)), methods=(1,)) is True
    a, b = "balance", "credit limit"
    assert run(bank, x, q(type="compare", block=a, other_block=b)) == combine(bool(phi1[a] > phi1[b]),
                                                                              bool(phi2[a] > phi2[b]))
    cols = BLOCKS["balance"]
    assert run(bank, x, q(type="measure", feature="balance", value=round(x[cols].mean(), 3), decimals=3)) is True
    higher = bool(x[cols].mean() > bank.refs[1][cols].to_numpy().mean())  # the group it names: method 1 (decision 50)
    assert run(bank, x, q(type="measure", feature="balance", direction="higher")) is higher
    average = round(float(bank.refs[1]["LIMIT_BAL"].mean()), 2)
    assert run(bank, x, q(type="measure", feature="LIMIT_BAL", reference_value=average, decimals=2)) is True
    assert run(bank, x, q(type="measure", feature="income", direction="higher")) is None



def test_the_model_uses_its_four_blocks_and_nothing_else(bank):
    x = bank.declined.iloc[0]
    assert run(bank, x, q(type="used", feature="age", is_used=False)) is True
    assert run(bank, x, q(type="used", feature="income", is_used=True)) is False
    assert run(bank, x, q(type="used", feature="balance", is_used=True)) is True
    assert run(bank, x, q(type="used", feature="shoe size", is_used=False)) is None


def test_payment_status_is_read_as_kinds_of_months_and_counts_of_late_months(bank):
    """Decision 33: a month is card not used, paid in full, paid on time or k months late; the group is compared by
    its share late; the six months by the number of late months."""
    x = bank.declined.iloc[0].copy()
    x[STATUS] = [-1, 2, 0, 1, -2, 0]
    group = bank.refs[1]
    m = {"methods": (1,)}
    assert run(bank, x, q(type="measure", feature="PAY_0", status="paid in full"), **m) is True
    assert run(bank, x, q(type="measure", feature="PAY_0", status="paid on time"), **m) is False
    assert run(bank, x, q(type="measure", feature="PAY_2", status="late", value=2, decimals=0), **m) is True
    assert run(bank, x, q(type="measure", feature="PAY_0", value=0, decimals=0), **m) is True  # not late
    share = round(float((group["PAY_0"] >= 1).mean()), 3)
    assert run(bank, x, q(type="measure", feature="PAY_0", reference_value=share, decimals=3), **m) is True
    assert run(bank, x, q(type="measure", feature="payment history", value=2, decimals=0), **m) is True
    mean = float((group[STATUS] >= 1).sum(axis=1).mean())
    assert run(bank, x, q(type="measure", feature="payment history", reference_value=round(mean, 2), decimals=2),
               **m) is True
    assert run(bank, x, q(type="measure", feature="payment history", direction="higher"), **m) is (2 > mean)
    assert run(bank, x, q(type="measure", feature="balance", status="late"), **m) is None


def test_a_notice_must_state_every_principal_reason(bank):
    x = bank.declined.iloc[0]
    top1, top2 = (reasons(shortfalls(bank, x, m)) for m in (1, 2))
    assert complete(bank, x, set(top1), methods=(1,)) is True
    assert complete(bank, x, set(top1[1:]), methods=(1,)) is False
    assert complete(bank, x, set(top1)) == combine(True, set(top2) <= set(top1))
    sentences = [{"queries": [q(type="reason", block=b, direction="higher") for b in top1[:-1]]}]
    assert stated_reasons(sentences) == set(top1[:-1])


def test_the_gate_passes_only_quoted_sentences_with_true_queries_that_cover_the_answer(bank):
    x = bank.declined.iloc[0]
    declined = q(type="outcome", quantity="decision", decision="declined")
    approved = q(type="outcome", quantity="decision", decision="approved")
    answer = "Your request was declined. I hope this helps."
    claim = {"sentence": "Your request was declined.", "kind": "claims", "queries": [declined]}
    polite = {"sentence": "I hope this helps.", "kind": "no_claim", "queries": []}

    sentences, covered = check(bank, x, answer, {"sentences": [claim, polite]})
    assert covered and all(s["ok"] for s in sentences)
    assert not check(bank, x, answer, {"sentences": [{**claim, "queries": [approved]}, polite]})[0][0]["ok"]
    assert not check(bank, x, answer, {"sentences": [{**claim, "kind": "unmappable", "queries": []}, polite]})[0][0]["ok"]
    sentences, covered = check(bank, x, answer, {"sentences": [{**claim, "sentence": "It was refused."}, polite]})
    assert not sentences[0]["ok"] and not covered
    assert check(bank, x, answer, {"sentences": [claim]})[1] is False
