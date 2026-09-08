# Day 3 — Search + LLM Grounded Research

**Grounding** means handing the LLM a stack of retrieved snippets and telling it, explicitly:
answer *only* from this material, name which source backs each claim, and if the material doesn't
cover the question, say so instead of filling the gap from memory. That instruction is what turns
a chatty guesser into a research assistant you can actually trust — it's the main lever for
suppressing hallucination in a search-grounded pattern.

## A four-part prompt template

Every grounded prompt needs these four pieces, in this order:

1. **Role instruction** — "You are a research assistant. Use only the material below."
2. **Retrieved input material** — the numbered, sourced context bundle from Day 1/2.
3. **Required output format** — e.g. "Write 3-5 sentences, then a Sources list."
4. **Verification/citation rules** — "Cite `[n]` after every claim. If the material doesn't
   answer the question, reply 'Not enough information in the provided sources.'"

```python
def build_grounded_prompt(question: str, bundle: str) -> str:
    return f"""You are a research assistant. Answer using ONLY the material below.

MATERIAL:
{bundle}

QUESTION: {question}

Rules:
- Cite the source number [n] after every factual claim.
- If the material does not answer the question, say "Not enough information in the provided sources."
- Do not add outside knowledge.
"""
```

## Chaining search into a research pipeline

`research()` chains three steps: search, bundle, then summarize-with-sources. Each stage stays a
small, separately testable function.

```python
def research(question: str, searcher, ranker, summarizer) -> str:
    raw_hits = searcher(question)                  # step 1: search
    top_hits = ranker(question, raw_hits)           # step 2: narrow with re-ranking (Day 2)
    bundle = build_context_bundle(top_hits)          # step 3: bundle, URLs preserved (Day 1)
    prompt = build_grounded_prompt(question, bundle)
    return summarizer(prompt)                        # step 4: LLM call, grounded
```

`searcher`, `ranker`, and `summarizer` are passed in as arguments on purpose — swap the mock
search stub for a real client, or a stub LLM call for a real one, without touching `research()`
itself.

## What a grounded answer looks like

Given snippets about a library's release notes, a grounded reply reads like:

```
The library added native retry support in v2.3 [1]. Backoff intervals are configurable
via a `backoff_factor` argument [2]. Not enough information in the provided sources to
say whether this is enabled by default.

Sources:
[1] https://example.dev/changelog/v2.3
[2] https://example.dev/docs/retries
```

Notice it stops rather than guesses at the default-behavior question — that's grounding working
as intended.

**Takeaway:** grounding is one explicit instruction — "answer only from this, cite it, or admit you don't know" — wired into a search-to-summary pipeline.
