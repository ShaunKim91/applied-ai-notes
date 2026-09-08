# Day 3 — Storing Embeddings at Scale: Vector Databases

Yesterday we compared a query vector against a handful of passages with a
Python loop. That works fine for 20 handbook sections. It falls apart at
20,000 manual pages across every product line the company sells.

## Why brute force stops working

Comparing a query against every stored vector one-by-one is O(n) per search —
fine for a small list, painfully slow once you have hundreds of thousands of
chunks, and it only gets worse as the document set grows. A **vector
database** solves this with indexing structures built specifically for
"find the nearest vectors to this one" at scale, turning a linear scan into a
much faster approximate search.

## Core operations

Regardless of which specific vector database you use, the same operations
show up:

```python
# 1. Create a collection (a named table of vectors)
collection = db.create_collection("handbook_sections")

# 2. Add items: each needs an id, the original text, its embedding, and metadata
collection.add(
    ids=["sec-4.2"],
    documents=["New hires accrue 12 vacation days in their first year."],
    embeddings=[embed("New hires accrue 12 vacation days in their first year.")],
    metadatas=[{"source": "handbook.pdf", "page": 14}],
)

# 3. Query: embed the question, ask for the top-k closest matches
results = collection.query(query_embeddings=[embed("how much PTO do new hires get")], n_results=3)
```

The metadata field matters as much as the vector itself — it's how you trace
an answer back to "handbook.pdf, page 14" later.

## Distance vs. similarity

Vector databases usually report a **distance** (how far apart two vectors
are — smaller is more relevant), while cosine **similarity** from Day 2 is
the opposite direction (larger is more relevant). Check which convention a
given database/query returns before assuming "highest score wins" — with a
distance metric, you want the *lowest* score.

## Fallback when the library isn't installed

Not every environment has a vector database library available. A reasonable
pattern is to fall back to the brute-force cosine search from Day 2 when the
import fails, so the rest of the code doesn't need to know which path ran:

```python
try:
    import vector_db_client as vdb
    HAS_VECTOR_DB = True
except ImportError:
    HAS_VECTOR_DB = False

def search(query: str, top_k: int = 3):
    if HAS_VECTOR_DB:
        return vdb_query(query, top_k)
    return brute_force_cosine_search(query, top_k)  # pure-Python fallback
```

This keeps a notebook or small script runnable anywhere, while still using
the real database when it's available.

## The production landscape

At production scale, teams typically reach for one of: a managed/hosted
vector database service, a self-hosted vector search engine, or a vector
extension bolted onto a database you already run. The right choice depends on
scale, latency needs, and operational preference — the concepts above (create
a collection, add with metadata, query top-k, mind distance vs. similarity)
carry over regardless of which one you pick.

## Takeaway

A vector database is what turns "compare against everything" into "find the
few that matter" — the API shape is nearly universal even though the
specific engine you plug in can vary.
