# Week 4 — Retrieval-Augmented Generation (RAG) Fundamentals

Building a document-grounded Q&A assistant from first principles: why plain
LLMs aren't enough, how embeddings and similarity find relevant text, how
vector databases scale that search, and how to answer only from retrieved
sources.

| Day | Topic | Notes |
|---|---|---|
| Day 1 | [Why your LLM doesn't know your company's handbook](daily/Day1_llm-limits-and-rag-intro.md) | Hallucination, knowledge cutoff, private-data gap; the open-book-exam analogy; the 4-stage RAG flow |
| Day 2 | [Turning text into numbers you can compare](daily/Day2_embeddings-and-cosine-similarity.md) | Embeddings, a from-scratch n-gram toy embedding, cosine similarity ranking, spelling-vs-meaning limitation |
| Day 3 | [Storing embeddings at scale: vector databases](daily/Day3_vector-databases.md) | Why brute force doesn't scale, core collection/add/query operations, distance vs. similarity, pure-Python fallback |
| Day 4 | [Chunking documents and answering only from what you retrieved](daily/Day4_chunking-and-grounded-answers.md) | Why not send whole documents, chunk size/overlap trade-offs, the `answer()` pattern with citations and "I don't know" |

See [`concepts/4th_week_Concepts.ipynb`](concepts/4th_week_Concepts.ipynb) for a
single runnable notebook covering all four days with short, self-contained
examples.
