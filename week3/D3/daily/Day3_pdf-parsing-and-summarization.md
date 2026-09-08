# Day 3 — Reading PDFs and Summarizing Them Without Making Things Up

The store keeps hundreds of supplier spec-sheet PDFs. Today: pull the text out
and get an AI-written summary you can actually trust.

## Extracting text with `pypdf`

Most PDFs store an actual text layer alongside the visual layout, and
`pypdf` can read it page by page:

```python
from pypdf import PdfReader

reader = PdfReader("cordless-drill-spec.pdf")
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
print(len(reader.pages), "pages,", len(full_text), "characters extracted")
```

Note the `or ""` — `extract_text()` can return `None` or an empty string when a
page has no recoverable text layer, which brings us to the next problem.

## The scanned-PDF trap

Some spec sheets are just scanned images saved as a PDF — a photo of paper,
not real text. `pypdf` will extract nothing (or garbage) from these, because
there's no text layer to read, only pixels. The fix is to treat the page as an
*image* and hand it to a multimodal model as an OCR substitute: "read the text
on this page and return it verbatim." The same model that described product
photos on Day 1 can transcribe a scanned page — you're just asking a different
question of it.

## Chunking long documents (map-reduce summarization)

A 40-page manual will exceed most models' context window, or at least blow
your budget on a single request. The standard workaround is a map-reduce
pattern:

1. **Map** — split the document into chunks (e.g. by page or by ~2,000-word
   sections), and summarize each chunk independently.
2. **Reduce** — concatenate the chunk summaries and summarize *that* into one
   final summary.

```python
def summarize_long_document(pages: list[str], summarize_fn) -> str:
    chunk_summaries = [summarize_fn(page) for page in pages]       # map
    combined = "\n".join(chunk_summaries)
    return summarize_fn(combined, instruction="Combine these into one summary")  # reduce
```

This keeps every individual request within context limits while still
covering the whole document.

## Trust, but verify: checking the numbers

Summaries that include specific numbers ("max torque: 60 Nm") are exactly
where hallucination is most dangerous — a fabricated spec looks just as
confident as a real one. A cheap safety net: after generating a summary,
programmatically check that any number it cites actually appears somewhere in
the source text.

```python
import re

def numbers_in_summary_are_grounded(summary: str, source_text: str) -> bool:
    cited_numbers = re.findall(r"\d+(?:\.\d+)?", summary)
    return all(num in source_text for num in cited_numbers)
```

This won't catch every kind of hallucination, but it's a fast, mechanical
check on the class of errors that matters most for spec sheets.

## Takeaway

PDF text extraction is easy until it isn't (scans), and long-document
summarization is easy until you check whether the numbers it quoted are real —
build both checks in from the start.
