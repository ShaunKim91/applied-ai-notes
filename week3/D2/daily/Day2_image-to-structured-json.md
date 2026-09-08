# Day 2 — From Photo to Structured JSON

Yesterday we asked a multimodal model to *describe* a photo in free text.
Today's job is different: turn a photo of a receipt into data a program can
actually use — a total, a date, a list of line items.

## Free text vs. structured output

Asking "what's in this receipt?" gets you a paragraph. That's fine for a human,
useless for a database. Instead, prompt the model to return JSON matching a
shape you specify:

```python
PROMPT = """
Extract the following fields from this receipt image and return ONLY JSON,
no commentary:

{
  "store_name": string,
  "purchase_date": string,
  "line_items": [{"name": string, "price": number}],
  "total": number
}
"""
```

Being explicit about the shape (field names, types) dramatically improves how
often the model returns something parseable.

## `safe_json()`: don't trust the output blindly

Even with a good prompt, models occasionally wrap JSON in markdown fences, add
a stray sentence, or emit almost-valid JSON. Never call `json.loads()` directly
on raw model output in production code — wrap it:

```python
import json

def safe_json(raw_text: str, default=None):
    """Try to parse model output as JSON; return `default` on any failure."""
    cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return default

result = safe_json(model_response, default={"error": "could not parse"})
```

This pattern — try the parse, fall back to a safe default, and log the raw
text for debugging — is the difference between a pipeline that occasionally
skips a bad record and one that crashes on it.

## Where object detection (YOLO-style) fits — and doesn't

Classic object detection models (YOLO and similar) draw bounding boxes around
objects and label them — "there is a *bottle* at these pixel coordinates."
That's the right tool when you need pixel-level localization, e.g. counting
items on a shelf. It is the *wrong* tool for reading a receipt: it has no
notion of "total" or "purchase date," only object classes and boxes. Prompted
extraction from a multimodal model, by contrast, understands document
*semantics* — it can reason about which number is the total versus a subtotal
— at the cost of not telling you *where* on the image that text was.

## Handling PII in real documents

Receipts and ID-like documents contain personal data: names, card numbers
(partial), addresses, signatures. Before storing extracted JSON:

- Strip or mask fields you don't need (e.g. keep last 4 digits of a card, drop
  the rest).
- Avoid sending images to third-party APIs if your data-handling policy
  forbids it — consider a locally-hosted VLM for sensitive documents.
- Log extraction failures without logging the raw image or full PII payload.

## Takeaway

Structured extraction is a prompting problem plus a defensive parsing
problem — ask clearly for a shape, then never trust that the model actually
gave it to you.
