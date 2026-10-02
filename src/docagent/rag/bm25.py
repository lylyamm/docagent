"""BM25: lexical search, the "exact words" half of the hybrid search.

Score of a passage d for a query q (Robertson & Zaragoza, 2009):

    score(q, d) = sum over terms t of q of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

- tf: occurrences of t in d, with diminishing returns (k1 = 1.5: the 5th "CO2"
  adds much less than the 1st);
- idf: rare terms weigh more ("SSP1-2.6" in 12 passages beats "climate" in 2,000);
- |d| / avgdl: long passages are penalised (b = 0.75), or they would win just by
  containing more words.

Written by hand (about 60 lines) rather than taken from a library, to control
the tokenisation: climate reports are full of codes that must stay whole.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict

# Lower-cased, accent-free words; codes and numbers stay whole: "ssp1-2.6", "co2", "1,5".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,\-/][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can for from has have in into is it its of on
    or that the their there these this those to was were which will with within would
    au aux avec ce ces cette dans de des du elle en est et il ils la le les leur
    leurs mais ne ou par pas plus pour qu que qui sa se ses son sont sur un une vers
    d l s n c j m y ont ete etre fait aussi comme entre dont
    """.split()  # noqa: SIM905 - a word list reads better as text
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _stem(token: str) -> str:
    """Crude plural folding shared by French and English: "emissions" -> "emission".

    Codes and numbers are left alone. A real stemmer (Snowball) would fold more
    ("warming"/"warmed"), at the cost of a per-language dependency.
    """
    if len(token) > 4 and token.isalpha() and token[-1] in "sx" and token[-2] not in "su":
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(_strip_accents(text.lower()))
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in documents]
        self.lengths = [len(t) for t in tokenized]
        self.avgdl = sum(self.lengths) / len(tokenized) if tokenized else 0.0
        # Inverted index: term -> {document position: term frequency}.
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                self.postings[term][i] = tf
        n = len(tokenized)
        self.idf = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """(document position, score) of the best documents, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term].items():
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda item: -item[1])[:top_k]
