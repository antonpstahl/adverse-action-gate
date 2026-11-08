"""Every setting of the study in one place (plan §7). Nothing here changes once the confirmation run is frozen."""

from pathlib import Path

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "taiwan_credit.xls"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

SEED = 7
TEST_SHARE = 0.3
APPROVAL_RATE = 0.75  # share of training applicants approved; this fixes the cutoff on the predicted risk
BAND = 0.05  # Reg B method 1 until Sudjianto's weighting can be used: approved applicants in the 5 points of
# the risk ranking just below the cutoff, i.e. "at or slightly above the minimum passing score" (decision 5)
N_REF = 1000  # rows drawn once (SEED) from each reference group
MAX_REASONS = 4  # Reg B comment 9(b)(2)-1
N_CHECK = 15  # answers per model that Anton reads blind to check the scorer, half where the conditions differ

STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # repayment status, newest month first
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]  # statement balance, newest first
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]  # amount paid, newest first
BLOCKS = {
    "payment history": STATUS,
    "balance": BILLS,
    "payment amounts": PAYMENTS,
    "credit limit": ["LIMIT_BAL"],
}  # EDUCATION is left out like the protected attributes: lenders avoid it as a fair-lending risk (decision 31)
FEATURES = [c for cols in BLOCKS.values() for c in cols]
# Monotone constraints, a modelling assumption (decision 24): being later and owing more never lower the risk, paying
# more never raises it. XGBoost may also combine features only within a block (decision 25).
MONOTONE = {**dict.fromkeys(STATUS, 1), **dict.fromkeys(BILLS, 1), **dict.fromkeys(PAYMENTS, -1)}

WRITER = "claude-haiku-5-5"  # the cheap model a bank would run for millions of answers (decision 18)
READER = "claude-opus-5-5"  # Sonnet 5.5 dropped negations when writing queries (decision 52)
WRITER_EFFORT = "low"  # short chat answers
READER_EFFORT = "medium"
SCORER = "gpt-6.1-sol"  # scores every answer: another vendor's model, sharing neither model nor family with the
SCORER_EFFORT = "medium"  # gate (decision 45); OpenAI's reasoning effort
MAX_TOKENS = 16000
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),  # USD per million input / output tokens;
         "claude-haiku-5-5": (0.10, 0.50),  # Sonnet for the replays of pilots 4 and 5
         "gpt-6.1-sol": (2.0, 10.0)}  # Claude: price table of 6 Oct 2026; OpenAI: of 20 Oct 2026
