# Day 1 — Seeing Without Training: Pretrained Multimodal Models

**Scenario for the week:** a small hardware store wants to automate parts of its back
office — tagging product photos, reading receipts, summarizing spec sheets, and
turning supplier web pages into price reports. Day 1 starts with the photos.

## How a CNN "sees" an image (conceptually)

You don't need to implement one to understand the idea. A convolutional neural
network builds up understanding of an image in layers:

1. **Early layers** respond to edges, corners, and color gradients.
2. **Middle layers** combine those into textures and simple shapes (a curve, a
   grid pattern, a patch of metal).
3. **Late layers** assemble shapes into parts and then whole objects — "this
   cluster of edges and textures is a hammer head on a handle."

Each layer's output feeds the next, so the network never "sees" a wrench as a
wrench directly — it accumulates the evidence across many stacked filters. That's
useful intuition even though we won't train one ourselves this week.

## Why not just train one?

Training a CNN from scratch to recognize "power drill," "box of screws," and
"paint can" would require: thousands of labeled photos per category, a GPU
budget, hours-to-days of training time, and re-training whenever a new product
category shows up. For a small store, that cost never pays for itself.

**The alternative:** a pretrained multimodal vision-language model (VLM) has
already seen enormous amounts of image-and-text data. It can describe, classify,
or answer questions about a photo it has never encountered, with zero training —
you just send the image and a prompt. Instead of *building* a vision system, you
*call* one.

| | Train your own CNN | Call a pretrained multimodal model |
|---|---|---|
| Data needed | Thousands of labeled images | Zero (few examples at most) |
| Setup time | Days to weeks | One API call |
| Cost | GPU hours + labeling labor | Per-request inference cost |
| Flexibility | Fixed to trained categories | Ask it anything about the image |

## A mock-first wrapper pattern

Because a real key/network call costs money and can fail, build a wrapper that
falls back to a canned response so the rest of your pipeline can be developed
and tested without hitting the network every time:

```python
import os

def describe_product_photo(image_path: str, api_key: str | None = None) -> str:
    """Return a short description of a product photo.

    Falls back to a canned response when no API key/network is available,
    so downstream code can be written and tested offline.
    """
    api_key = api_key or os.getenv("VISION_API_KEY")
    if not api_key:
        return "[mock] A cordless power drill, viewed from the side, on a white background."

    # Real call would go here, e.g. a vision-capable chat completion request
    # that sends the image bytes/URL plus a text prompt and returns the model's answer.
    response = call_multimodal_api(image_path, prompt="Describe this product photo.", api_key=api_key)
    return response.text
```

This "mock-first" habit — write the interface and a fake implementation first,
swap in the real call later — keeps development moving and makes tests
deterministic.

## Takeaway

For most real-world image tasks today, reaching for a pretrained multimodal
model beats training a CNN from scratch — you trade a training pipeline for a
single well-crafted prompt.
