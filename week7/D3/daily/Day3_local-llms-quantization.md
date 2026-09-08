# Day 3 — Running Agents on Local LLMs, and Quantization

## Cloud API vs. local model

A cloud-hosted model gives you bigger weights and zero local hardware
burden, at the cost of per-token billing, a network round trip, and your
prompts leaving the machine. A local model flips every one of those: data
never leaves your device, no metered bill, it works offline — but you're
capped by whatever RAM/VRAM you own, and you pay the electricity and
hardware cost up front. Agents handling sensitive data or high volume lean
local; agents needing frontier-level reasoning lean cloud.

## Measuring dtype cost — in a fresh subprocess each time

Loading the same model in `float32`, `float16`, and `bfloat16` produces
noticeably different memory footprints and speeds. The trap: loading all
three back-to-back in one Python process means the first model's allocator
overhead and cached memory pollute the second measurement. Isolate each
load in its own subprocess instead:

```python
import subprocess, sys

def measure_dtype(model_id, dtype_name):
    script = (
        "import time, torch\n"
        "from transformers import AutoModelForCausalLM\n"
        "t0 = time.time()\n"
        f"m = AutoModelForCausalLM.from_pretrained('{model_id}', torch_dtype=torch.{dtype_name})\n"
        "print(f'{time.time()-t0:.2f}s')\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout.strip()

for dtype in ["float32", "float16", "bfloat16"]:
    print(dtype, measure_dtype("local-small-model", dtype))
```

Each `subprocess.run` call gets a clean process, loads one model, reports
its number, and exits — so the OS reclaims everything before the next run.

## Quantization: fewer bits per weight

Quantization stores weights at lower precision — INT8 or INT4 instead of
16/32-bit floats — trading a small accuracy hit for large memory and speed
wins. Two ways to get there: **post-training quantization (PTQ)** trains
normally at full precision, then quantizes the finished weights afterward —
fast and simple, slightly lossier. **Quantization-aware training (QAT)**
simulates the rounding error of low-precision weights *during* training or
fine-tuning so the model learns to compensate — more setup cost, usually
better accuracy at very low bit-widths.

## When it just doesn't work: a CPU-only case study

Some quantization tooling assumes a CUDA GPU is present and throws when it
isn't. Handle that explicitly rather than letting the whole script die:

```python
try:
    import bitsandbytes as bnb
    quantized = bnb.nn.Linear8bitLt(in_features=512, out_features=512, has_fp16_weights=False)
except Exception as e:
    print(f"8-bit quantization unavailable on this hardware ({e}); falling back to float32.")
    quantized = None
```

On CPU-only hardware that's expected, not a bug — plan a fallback instead of
assuming the GPU branch always succeeds.

## A tool-calling convention for local models

Local models are more reliable at a rigid, string-matchable format than at
JSON tool-calling. A convention like `TOOL_CALL: name(args)` parses cleanly
with a regex:

```python
import re

TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\w+)\((.*)\)")

def dispatch(model_output, tools):
    match = TOOL_CALL_RE.search(model_output)
    return tools[match.group(1)](match.group(2)) if match else None
```

**Takeaway:** local models trade capability for privacy and cost control — quantization and careful, isolated measurement are how you find the setting that still fits your hardware.
