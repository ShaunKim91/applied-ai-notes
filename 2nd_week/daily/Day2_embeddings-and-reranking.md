# Day 2 — Embedding Models and Reranking

Semantic search usually runs in two stages: a fast, approximate retrieval pass, followed by a slower, more careful reranking pass. Today: why both stages exist, and where each one breaks.

## Bi-encoders: fast but coarse

A **bi-encoder** embeds the query and every candidate document *independently*, each through the same encoder, producing one fixed-size vector per text. Similarity is then just a vector comparison (cosine similarity or dot product), which is what lets you precompute embeddings for millions of documents offline and search them in milliseconds with a vector index. The cost: the encoder never sees the query and a specific document together, so it can miss interactions that only show up when both are read side by side.

## Cross-encoders: slow but sharp

A **cross-encoder** instead concatenates the query and one candidate document into a single input (`[CLS] query [SEP] document`) and runs them jointly through a transformer, outputting one relevance score for that pair. Because the model attends across both texts simultaneously, it can catch subtler mismatches a bi-encoder's independent embeddings blur together. The cost: it has to be rerun once per candidate document — no offline precomputation possible — so it doesn't scale to searching millions of documents directly.

## Watching a bi-encoder get it wrong

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "how do I reset my account password"
docs = [
    "To change your account password, go to Settings > Security and click Reset Password.",  # relevant
    "Our password-protected storage units require a 4-digit code set at check-in.",           # irrelevant, shares 'password'
]

q_vec = bi_encoder.encode(query)
d_vecs = bi_encoder.encode(docs)
scores = d_vecs @ q_vec / (np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec))
```

Because both documents share surface-level vocabulary with the query ("password", "reset"-adjacent phrasing), it's common for a bi-encoder to rank them closer together than their actual relevance warrants — sometimes even inverting the order. A cross-encoder, reading the query and each document jointly, typically separates them correctly:

```python
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc) for doc in docs]
rerank_scores = cross_encoder.predict(pairs)
```

## Retrieve-then-rerank

The practical answer is to use both, in sequence: a bi-encoder (or classic keyword search) retrieves a broad candidate set — say, the top 100 — cheaply from a large corpus, then a cross-encoder reranks just those 100 for precision. This gets the scalability of the bi-encoder and most of the accuracy of the cross-encoder, without paying cross-encoder cost on the entire corpus.

## Why bi-encoders fail this way: contrastive training

Bi-encoders are typically trained with **contrastive learning** — pulling a query's embedding close to its known-relevant document and pushing it away from negative examples. The quality of that push depends heavily on the negatives used. Easy negatives (a random unrelated document) teach little; **hard negatives** (documents that share vocabulary or topic but aren't actually relevant, like the storage-unit example above) are what actually teach the model to separate superficially-similar-but-wrong from truly relevant. A bi-encoder trained without enough hard negatives will keep making exactly this kind of mistake.

## Matryoshka embeddings, briefly

Some newer embedding models are trained so that *truncating* the embedding vector — keeping only its first 256 of 1024 dimensions, say — still yields a usable, if slightly less accurate, embedding. These **Matryoshka embeddings** let a single model serve both a cheap, low-dimensional index for coarse retrieval and a full-dimensional embedding for higher-accuracy comparisons, without training or storing two separate models.

**Takeaway:** bi-encoders scale but can be fooled by surface-level word overlap; cross-encoders are accurate but too slow to run on a whole corpus — retrieve-then-rerank gets you both.
