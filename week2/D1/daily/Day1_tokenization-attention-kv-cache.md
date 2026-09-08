# Day 1 — Tokenization, Attention, and the KV Cache

Before a transformer can pay attention to anything, text has to become numbers. Today: how that split happens, how attention uses the result, and why long conversations don't require redoing all the work from scratch.

## BPE: building a vocabulary from scratch

**Byte-Pair Encoding (BPE)** starts with individual characters (or, in byte-level BPE, raw bytes — which guarantees every possible input string is representable, emoji included) and iteratively merges the most frequent adjacent pair into a new token, repeating thousands of times. Common words end up as a single token; rare or made-up words get split into subword pieces. That's why "unbelievable" might tokenize as one piece while a made-up word splits into several.

Different languages don't tokenize equally efficiently under the same vocabulary. A tokenizer whose merge rules were learned mostly on English text tends to represent English words in fewer tokens than an equivalent sentence in, say, Korean — because the frequent subword patterns the merges captured are English-shaped.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
en = "The quarterly report is ready for review."
ko = "분기 보고서 검토 준비가 완료되었습니다."

print(len(enc.encode(en)), "tokens for English")
print(len(enc.encode(ko)), "tokens for Korean")
```

Running this typically shows meaningfully more tokens for the Korean sentence despite it conveying roughly the same content — which matters directly for cost and context-window budget when working across languages.

## Embeddings and cosine similarity

Once tokenized, each token maps to a learned vector — its **embedding**. Vectors that end up close together by **cosine similarity** (the cosine of the angle between them, ignoring magnitude and focusing purely on direction) tend to represent related meaning:

```python
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## Self-attention: Query, Key, Value

For each token, self-attention derives three vectors from its embedding: a **Query** (what this token is looking for), a **Key** (what this token offers), and a **Value** (the content it contributes if attended to). A token's new representation is a weighted sum of every other token's Value, where the weights come from comparing its Query against every Key (`softmax(QK^T / sqrt(d))`). **Multi-head attention** runs several of these Q/K/V projections in parallel, each with a smaller dimension, so different heads can specialize — one tracking syntax, another tracking coreference — then concatenates the results.

Because this operation has no inherent sense of order (attention treats the input as a set), a **positional encoding** — a vector encoding each token's position — is added before attention so the model can tell "dog bites man" apart from "man bites dog."

## The KV cache

During generation, a decoder produces one token at a time, and each new token needs to attend back to every previous token's Key and Value. Recomputing K and V for the entire prefix at every single step would be wasteful — they don't change once a token has already been processed. The **KV cache** stores each token's K and V vectors the first time they're computed, so generating token N+1 only requires computing Q/K/V for the new token and reusing the cache for everything before it.

This is what keeps decode-time cost roughly linear rather than quadratic in the number of generated tokens: without caching, attention cost per new token grows with the full prefix length each time (an O(N²) total cost across a generation); with caching, each new token costs O(N) to attend over a growing cache, and that cache itself becomes the thing consuming memory as N grows. That memory footprint — one K and one V vector, per layer, per head, per token — is exactly what **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** target, by sharing a single (MQA) or a small group of (GQA) K/V heads across many query heads, shrinking the cache without discarding the multi-head query mechanism.

**Takeaway:** tokenization decides how "long" your input looks to the model at all, attention decides what each token learns from every other token, and the KV cache is the reason a 2,000-token conversation doesn't get quadratically slower to continue.
