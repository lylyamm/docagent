"""Retrieval metrics, and paired tests to compare two systems on the same questions.

A question is answered when a returned passage comes from one of its expected
(document, page) pairs:
- hit@k: share of questions with a right passage in the top k;
- MRR@10: mean of 1/rank of the first right passage (0 if none in the top 10).

With a few dozen questions, a gap between two systems can be luck. Both systems
answer the SAME questions, so they are compared question by question (paired),
looking only at the questions where they disagree:
- McNemar's exact test on "found in the top k or not": with b questions won by
  A and c by B, if A and B were equally good each disagreement would be a coin
  flip, so min(b, c) follows Binomial(b + c, 1/2);
- the sign test on the rank of the right page: same coin flip, over every
  question where one system ranks it higher than the other.
The p-value is the probability of a split at least this lopsided between two
equally good systems (two-sided).
"""

from collections.abc import Sequence
from math import comb

from pydantic import BaseModel

Ranks = dict[str, int | None]  # question id -> rank of the first right passage (1-based)


def first_hit(pages: Sequence[tuple[str, int]], expected: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first (document, page) that answers, or None."""
    for rank, page in enumerate(pages, start=1):
        if page in expected:
            return rank
    return None


def summarise(ranks: Ranks, ids: Sequence[str] | None = None) -> dict[str, float]:
    values = [ranks[i] for i in (ids if ids is not None else list(ranks))]
    n = len(values) or 1
    return {
        "hit@1": sum(r is not None and r <= 1 for r in values) / n,
        "hit@5": sum(r is not None and r <= 5 for r in values) / n,
        "mrr@10": sum(1 / r for r in values if r is not None and r <= 10) / n,
    }


def binomial_p(wins: int, losses: int) -> float:
    """Two-sided exact test that wins and losses are a fair coin flip."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


class PairedComparison(BaseModel):
    hit_wins: int  # questions in the top k for A only
    hit_losses: int  # ... for B only
    hit_p: float  # McNemar exact
    rank_wins: int  # questions where A ranks the right page higher
    rank_losses: int
    rank_p: float  # sign test


def compare(a: Ranks, b: Ranks, k: int = 5, depth: int = 10) -> PairedComparison:
    """Paired comparison of A against B on the questions both answered.

    A right page beyond ``depth`` (or never found) counts as rank depth + 1.
    """
    ids = sorted(a.keys() & b.keys())

    def hit(r: int | None) -> bool:
        return r is not None and r <= k

    def rank(r: int | None) -> int:
        return r if r is not None and r <= depth else depth + 1

    hit_wins = sum(hit(a[i]) and not hit(b[i]) for i in ids)
    hit_losses = sum(hit(b[i]) and not hit(a[i]) for i in ids)
    rank_wins = sum(rank(a[i]) < rank(b[i]) for i in ids)
    rank_losses = sum(rank(a[i]) > rank(b[i]) for i in ids)
    return PairedComparison(
        hit_wins=hit_wins,
        hit_losses=hit_losses,
        hit_p=binomial_p(hit_wins, hit_losses),
        rank_wins=rank_wins,
        rank_losses=rank_losses,
        rank_p=binomial_p(rank_wins, rank_losses),
    )
