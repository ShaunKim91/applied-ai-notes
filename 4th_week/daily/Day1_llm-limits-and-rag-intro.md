# Day 1 — Why Your LLM Doesn't Know Your Company's Handbook

**Scenario for the week:** a small company wants an internal assistant that
can answer questions about its own employee handbook and product manuals —
documents no public model has ever seen.

## Three limits of a plain LLM

Ask a general-purpose LLM "how many vacation days do new hires get at
[our company]?" and one of three things happens:

1. **Hallucination** — it confidently invents a plausible-sounding number.
2. **Knowledge cutoff** — even if it once had relevant public info, anything
   written or changed after its training cutoff simply isn't in its weights.
3. **No access to private documents** — your handbook was never public
   training data in the first place, cutoff or not. The model has zero
   knowledge of it, full stop.

None of these are bugs to "prompt around" — they're structural. The model
only knows what was in its training data, and your internal documents never
were.

## RAG: the open-book exam

Retrieval-Augmented Generation (RAG) fixes this the same way an open-book
exam fixes "I don't remember the exact figure": instead of relying on memory,
you hand the model the relevant page *at answer time*.

- **Closed-book (plain LLM):** answer from memory alone → guesses when memory
  is missing.
- **Open-book (RAG):** look up the relevant passage first, then answer using
  it → grounded in an actual source.

## The four-stage flow

```
documents  →  index (search-ready form)  →  retrieve relevant pieces  →  generate grounded answer
```

1. **Documents** — your handbook, manuals, policies: the raw source of truth.
2. **Index** — break documents into searchable pieces (more on this Day 4)
   and prepare them so a query can find the right ones (more on this Day 2-3).
3. **Retrieve** — given a user's question, find the pieces most relevant to it.
4. **Generate** — hand the model the question *plus* those retrieved pieces,
   and instruct it to answer only from what it was given.

## Ungrounded vs. grounded, side by side

**Question:** "How many vacation days does a new hire get?"

| Ungrounded (plain LLM) | RAG (grounded) |
|---|---|
| "Typically, companies offer around 15 days of PTO for new employees." | "According to the handbook (Section 4.2): 'New hires accrue 12 vacation days in their first year.' → **12 days**." |
| Sounds reasonable. Is generic industry guesswork. | Cites the exact source snippet the answer came from. |

The ungrounded answer isn't obviously wrong — that's what makes it dangerous.
The grounded answer is checkable: you can go read Section 4.2 yourself.

## Takeaway

RAG doesn't make a model smarter — it gives a model something true to read
before it answers, and that's usually worth more than a smarter model with
nothing to read.
