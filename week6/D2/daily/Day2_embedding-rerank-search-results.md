# Day 2 — Embedding-Based Re-Ranking of Search Results

A search API's own ranking is optimized for general popularity, not for the exact question your
agent is answering. Five results come back; often only one or two are actually relevant, and the
rest are noise that wastes tokens and can distract the LLM from the right answer. The fix reuses
a tool from the earlier retrieval lesson: turn text into vectors and score relevance by **cosine
similarity**, then apply that same trick to sort search hits instead of document chunks.

## A deterministic, offline "embedding"

A real embedding model calls out to an API. For teaching, a small hash-based bag-of-words
vectorizer is enough to demonstrate the mechanics with no network call and no randomness — same
input text always produces the same vector.

```python
import hashlib, re
from collections import Counter

VECTOR_SIZE = 64

def embed_text(text: str) -> list[float]:
    vec = [0.0] * VECTOR_SIZE
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word, count in Counter(words).items():
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % VECTOR_SIZE
        vec[bucket] += count
    return vec

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

## Re-ranking search hits

Score each `Hit` (title + snippet) against the query vector, then sort descending. This is the
same `cosine_similarity` helper from the RAG lesson — the only thing that changed is what we're
scoring: search-engine snippets instead of chunks of a local document.

```python
def rerank_hits(query: str, hits: list, top_k: int = 2) -> list:
    q_vec = embed_text(query)
    scored = [(cosine_similarity(q_vec, embed_text(h.title + " " + h.snippet)), h) for h in hits]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [h for _, h in scored[:top_k]]
```

## Before vs. after

Say a search API returns five hits for "python async timeout" in its own default order: a general
async tutorial, a `subprocess` guide, an `asyncio.wait_for` reference page, a forum post about
Django timeouts, and a changelog entry. In API order, the on-topic `asyncio.wait_for` reference is
buried third. Running `rerank_hits` against the query pulls it — and the closest runner-up — to
the top two slots, and the Django/changelog noise drops out of the bundle entirely before it ever
reaches the LLM.

```
Before (API order):        After (embedding re-rank, top_k=2):
1. general async tutorial   1. asyncio.wait_for reference
2. subprocess guide         2. general async tutorial
3. asyncio.wait_for ref
4. django timeout forum
5. changelog entry
```

**Takeaway:** re-run the same embed-and-cosine-sort trick from RAG on raw search hits, and only the results actually close to the query make it into the LLM's context.
