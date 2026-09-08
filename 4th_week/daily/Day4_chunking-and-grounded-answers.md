# Day 4 — Chunking Documents and Answering Only From What You Retrieved

Everything this week builds toward one function: given a user's question,
answer it *using only the retrieved handbook text* — and say so honestly when
nothing relevant was found.

## Why not just feed the whole handbook to the model?

Three reasons this doesn't scale:

- **Cost** — every request re-sends the entire document, even for a
  one-sentence question. That's a lot of wasted tokens, repeatedly.
- **Relevance** — a 60-page handbook stuffed into one prompt buries the one
  relevant paragraph in noise, and models answer worse with irrelevant
  context crowding the real answer.
- **Context window limits** — sooner or later the document plus the
  question plus the desired answer simply won't fit in the model's context
  window at all.

Chunking + retrieval (Days 2-3) exists specifically to avoid this: send only
the few paragraphs that actually matter for this question.

## Chunk size and overlap

Splitting a document into chunks means picking two knobs:

- **Chunk size** — too small and a chunk loses context (a sentence fragment
  with no surrounding meaning); too large and you're back to diluting
  relevance and burning tokens on a single retrieved chunk.
- **Overlap** — a bit of shared text between consecutive chunks (e.g. the
  last 50 words of chunk N repeated at the start of chunk N+1) prevents an
  answer from being cut in half at a chunk boundary.

```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step back so chunks overlap
    return chunks
```

There's no universally "correct" size — it's a trade-off tuned against your
documents and your embedding model's own limits.

## The `answer()` pattern

The core discipline of RAG isn't retrieval — it's refusing to answer beyond
what retrieval actually found:

```python
def answer(question: str, top_k: int = 3, min_relevance: float = 0.2) -> dict:
    hits = search(question, top_k=top_k)              # Day 2/3 retrieval
    relevant = [h for h in hits if h["score"] >= min_relevance]

    if not relevant:
        return {"answer": "I don't know — nothing relevant was found in the documents.", "sources": []}

    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in relevant)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    reply = call_llm(prompt)
    return {"answer": reply, "sources": [h["source"] for h in relevant]}
```

Three things make this trustworthy: it retrieves before generating, it
instructs the model to stay inside the retrieved context, and it reports
sources alongside the answer so a person can verify it — mirroring the
grounded example from Day 1.

## Takeaway

A RAG system is only as honest as its refusal to answer — chunking makes
retrieval possible, but it's the explicit "I don't know" and cited sources
that make the answers trustworthy.
