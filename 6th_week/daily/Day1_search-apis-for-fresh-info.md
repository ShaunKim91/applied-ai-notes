# Day 1 — Search APIs for Fresh Information

An LLM only knows what was in its training data, frozen at some cutoff date. Pairing it with a
document-retrieval (RAG) system over a folder of PDFs or wiki pages doesn't fix that — those
documents are just as static as the model's weights. If nobody re-indexes them, "today's" answer
is still whatever was true when the files were last saved. To answer questions about *right now*
(a stock price, a library's latest release, this morning's news), an agent needs a way to reach
outside its own frozen knowledge and outside its own document store, out onto the live web.

## The search API abstraction

Strip away the branding of any web search product and the interface is almost always the same
shape: you send a short text **query**, and you get back an ordered list of **results**, each
with a `title`, a short `snippet` of matching text, and a `source_url`. Everything an agent does
with search — re-ranking, summarizing, citing — builds on top of that one simple contract.

```python
from dataclasses import dataclass

@dataclass
class Hit:
    title: str
    snippet: str
    source_url: str

def lookup_web(query: str, max_hits: int = 5) -> list[Hit]:
    """Real implementation would call a search provider's HTTP endpoint."""
    ...
```

## Mock-first design

Before wiring up a paid API key, build a stand-in that returns realistic `Hit` objects from a
small local dictionary. As long as it matches the real function's signature and return type,
swapping the stub for a live client later is a one-line change — nothing else in the agent has to
know the difference.

```python
_FAKE_INDEX = {
    "rust ownership": [
        Hit("Ownership - The Rust Book", "Each value has a single owner...", "https://doc.rust-lang.org/book/ch04-01"),
        Hit("Borrowing rules explained", "References let you use a value...", "https://example-blog.dev/rust-borrow"),
    ],
}

def lookup_web_stub(query: str, max_hits: int = 5) -> list[Hit]:
    key = query.lower().strip()
    return _FAKE_INDEX.get(key, [])[:max_hits]
```

## Specific queries beat vague ones

"rust ownership" returns two focused hits above. A query like "rust" alone would match nothing in
this tiny index, and against a real search engine it would flood the results with tutorials,
crates, job listings, and unrelated forum threads. Treat query construction as part of the
engineering: name the concept, add a qualifier ("explained", "vs", "error"), and prefer the
phrasing an expert would actually type over a restated version of the user's whole question.

## Bundling results without losing provenance

Once you have hits, fold them into one string to hand to the LLM — but every snippet must stay
attached to its URL. Drop the URLs and you can no longer cite a source or let a reader verify a
claim.

```python
def build_context_bundle(hits: list[Hit]) -> str:
    blocks = [f"[{i}] {h.title}\n{h.snippet}\nSource: {h.source_url}" for i, h in enumerate(hits, 1)]
    return "\n\n".join(blocks)
```

**Takeaway:** a search API is just "query in, titled/sourced snippets out" — mock that shape first, keep URLs glued to every snippet, and a real key becomes a drop-in swap later.
