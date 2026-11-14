"""Serializable BM25 and its dependency-free runtime, also embedded in generated servers."""

import math
import re
from collections import Counter


def terms(text):
    return re.findall(r"\w+", text.lower())


def make_index(records):
    frequencies = [dict(Counter(terms(record["text"]))) for record in records]
    lengths = [sum(freq.values()) for freq in frequencies]
    document_frequency = Counter(term for freq in frequencies for term in freq)
    return {"version": 1, "k1": 1.5, "b": 0.75, "frequencies": frequencies,
            "lengths": lengths, "avg_length": sum(lengths) / max(1, len(lengths)),
            "document_frequency": dict(document_frequency),
            "chunk_ids": [record["chunk_id"] for record in records]}


def bm25_search(index, records, query, k=3):
    if not isinstance(query, str) or not query.strip():
        return []
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10")
    scores = []
    count = len(records)
    for i, freq in enumerate(index["frequencies"]):
        score = 0.0
        for term in set(terms(query)):
            tf = freq.get(term, 0)
            df = index["document_frequency"].get(term, 0)
            if tf:
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                denominator = tf + index["k1"] * (1 - index["b"] + index["b"] *
                                                  index["lengths"][i] / max(1, index["avg_length"]))
                score += idf * tf * (index["k1"] + 1) / denominator
        if score > 0:
            scores.append((score, i))
    return [records[i] for _, i in sorted(scores, key=lambda item: (-item[0], item[1]))[:k]]
