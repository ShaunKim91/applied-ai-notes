# Day 2 — Turning Text Into Numbers You Can Compare

To "retrieve the relevant handbook passage" from Day 1, we need a way to
measure how close a question is to a passage in *meaning*. That's what
embeddings are for.

## What an embedding is

An embedding is a vector (a list of numbers) that represents a piece of text,
positioned so that texts with similar meaning end up with similar vectors.
Real embedding models (accessed via an API, e.g. an embeddings endpoint) are
trained on huge amounts of text and genuinely capture meaning — "time off"
and "vacation days" land close together even though they share no words.

## A toy embedding you can build without any API

To build intuition without needing network calls, we can fake a simple
embedding using **character n-grams**: break the text into overlapping
3-character chunks and hash each chunk into a fixed-size vector.

```python
import hashlib

def toy_embed(text: str, dims: int = 64) -> list[float]:
    vec = [0.0] * dims
    text = text.lower().replace(" ", "_")
    for i in range(len(text) - 2):
        trigram = text[i:i + 3]
        bucket = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dims
        vec[bucket] += 1.0
    return vec
```

This produces *a* vector, and similar-spelled text really does get similar
vectors — but as we'll see, "similar spelling" is not the same thing as
"similar meaning."

## Cosine similarity

Given two vectors, cosine similarity measures the angle between them — 1.0
means identical direction (same meaning/topic), 0 means unrelated, and it
ignores vector length so longer documents aren't unfairly favored:

```python
import math

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

To rank candidate passages against a query: embed the query, embed every
candidate, compute cosine similarity against each, and sort descending.

```python
query_vec = toy_embed("how much paid time off do I get")
scored = sorted(
    ((cosine_similarity(query_vec, toy_embed(doc)), doc) for doc in passages),
    reverse=True,
)
top_match = scored[0]
```

## The limitation: spelling overlap ≠ meaning

Our n-gram toy embedding will happily rank "vacation days" close to
"vacation destination" (they share lots of 3-character chunks) while
possibly missing that "paid time off" means the *same thing* as "vacation
days" (almost no character overlap at all). Real embedding models don't have
this problem — they were trained to place semantically similar phrases
together regardless of shared spelling, which is precisely why production
RAG systems use trained embedding models rather than a hashing trick.

## Takeaway

Embeddings turn "is this relevant?" into a number you can sort by; a
character-hashing toy embedding shows the *mechanism*, but only a real,
trained embedding model captures actual *meaning*.
