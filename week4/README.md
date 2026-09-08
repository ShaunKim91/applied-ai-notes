# Week 4 — Retrieval-Augmented Generation (RAG) Fundamentals

Building a document-grounded Q&A assistant from first principles: why plain
LLMs aren't enough, how embeddings and similarity find relevant text, how
vector databases scale that search, and how to answer only from retrieved
sources.

| Day | Topic | Notes |
|---|---|---|
| D1 | [Why your LLM doesn't know your company's handbook](D1/daily/Day1_llm-limits-and-rag-intro.md) | Hallucination, knowledge cutoff, private-data gap; the open-book-exam analogy; the 4-stage RAG flow |
| D2 | [Turning text into numbers you can compare](D2/daily/Day2_embeddings-and-cosine-similarity.md) | Embeddings, a from-scratch n-gram toy embedding, cosine similarity ranking, spelling-vs-meaning limitation |
| D3 | [Storing embeddings at scale: vector databases](D3/daily/Day3_vector-databases.md) | Why brute force doesn't scale, core collection/add/query operations, distance vs. similarity, pure-Python fallback |
| D4 | [Chunking documents and answering only from what you retrieved](D4/daily/Day4_chunking-and-grounded-answers.md) | Why not send whole documents, chunk size/overlap trade-offs, the `answer()` pattern with citations and "I don't know" |

See [`concepts/Week4_Concepts.ipynb`](concepts/Week4_Concepts.ipynb) for a
single runnable notebook covering all four days with short, self-contained
examples.
