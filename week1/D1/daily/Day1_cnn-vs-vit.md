# Day 1 — CNNs vs. Vision Transformers

Today's question: once a photo is just numbers, how does a network learn to "see"? We compare two dominant architectures — convolutional networks and vision transformers — and when to reach for a pretrained model instead of training from scratch.

## An image is a tensor

A color photo is stored as an `H x W x 3` array — one grid of intensities per RGB channel. A 224x224 photo is 150,528 numbers before any model touches it. Everything below is really just a different way of finding structure in that array.

## CNNs: sliding filters, growing receptive fields

A convolutional layer slides a small filter (e.g. 3x3) across the image, computing a weighted sum at each position. Stacking layers lets early filters detect edges and gradients, mid layers combine those into textures and parts, and deep layers assemble parts into whole objects — a hierarchy of features rather than one giant lookup table.

This design bakes in two assumptions, or **inductive biases**:
- **Locality** — nearby pixels are more related than distant ones, so a filter only looks at a small neighborhood at a time.
- **Translation equivariance** — a filter that finds an edge in the top-left corner finds the same edge in the bottom-right, because the same weights are reused everywhere.

Those biases are why CNNs learn well from comparatively small datasets — the architecture already "knows" something true about images before training even starts.

## ViT: an image as a sentence of patches

A Vision Transformer (ViT) throws away convolution entirely. It:
1. Slices the image into fixed-size patches (e.g. 16x16 pixels).
2. Flattens and linearly projects each patch into an embedding vector — a "patch token."
3. Prepends a learnable **CLS token** that will absorb a summary of the whole image.
4. Adds a **position embedding** to every token, since self-attention has no built-in sense of order.
5. Feeds the resulting sequence through standard transformer self-attention layers, where every patch can directly attend to every other patch.

Because attention carries no locality assumption, a ViT can relate a patch in one corner to a patch in the opposite corner in a single layer — something a CNN only achieves after several layers of downsampling. The tradeoff: with less built-in structure to lean on, ViTs typically need more training data (or a strong pretrained checkpoint) to match CNN-level accuracy.

## Why start from a pretrained model

Training either architecture from scratch demands millions of labeled images — almost nobody does this for a new task. Instead, pick a checkpoint someone already trained on a large dataset and fine-tune it, or use it directly. When choosing one, weigh:
- **Size** — a 90M-parameter model may be overkill for a CPU-only demo.
- **Training data** — a model trained on general web photos may need fine-tuning for medical or satellite imagery.
- **License** — some checkpoints restrict commercial use.

## Code pattern: batch classification with a review flag

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch, pandas as pd

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

rows = []
for path in ["plate_01.jpg", "plate_02.jpg", "plate_03.jpg"]:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)[0]
    top_prob, top_idx = probs.max(dim=-1)
    rows.append({
        "file": path,
        "label": model.config.id2label[top_idx.item()],
        "confidence": round(top_prob.item(), 3),
        "needs_review": top_prob.item() < 0.6,
    })

df = pd.DataFrame(rows)
```

Flagging anything under a confidence threshold (here, 60%) turns a black-box prediction into a human-in-the-loop review queue — cheap insurance against silently wrong auto-tags.

**Takeaway:** CNNs bake locality and translation equivariance directly into the architecture; ViTs learn spatial relationships from data via self-attention over patches instead — which is exactly why pretrained checkpoints and transfer learning matter even more once you move away from convolutions.
