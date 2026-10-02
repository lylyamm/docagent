"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).

    rrf(d) = sum over rankings r of  1 / (k + rank_r(d))      (rank starts at 1)

Only ranks are used, never raw scores: a cosine similarity (0.83) and a BM25
score (14.2) live on different scales and can't be added, but "2nd here, 5th
there" can. k = 60 (the paper's value) flattens the curve so that being found by
both searches beats being first in only one.
"""

from collections import defaultdict
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fused (item, score) list, best first. Ties keep the first ranking's order."""
    scores: dict[Hashable, float] = defaultdict(float)
    order: dict[Hashable, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
