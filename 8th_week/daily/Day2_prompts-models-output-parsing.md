# Day 2: Reusable Prompts, Swappable Models, and Parsing Raw Output

A template is only useful if you can reuse it. Today: filling one
`PromptTemplate` with many different inputs, why swapping model providers is
mostly a construction-line change, and why you should learn to parse raw LLM
text by hand before reaching for a framework shortcut.

## One template, many inputs

```python
from langchain_core.prompts import PromptTemplate

review_prompt = PromptTemplate.from_template(
    "Rate the sentiment of this review from 1-5 and give one reason: {review}"
)
sample_reviews = [
    "The battery died after two days and support never replied.",
    "Fast shipping, exactly as described, would buy again.",
    "It's okay, does the job but the app crashes sometimes.",
]
for review in sample_reviews:
    filled = review_prompt.invoke({"review": review})
    print(filled.to_string())
```

Same template object, three different `{review}` values — no copy-pasting the
instruction text three times.

## Swapping the model provider

Chat model classes across providers implement the same base interface, so
the surrounding code — build a prompt, call `.invoke()`, read the result —
barely changes:

```python
# Both need real API keys to actually run:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# model_a = ChatOpenAI(model="gpt-4o-mini")
# model_b = ChatAnthropic(model="claude-3-5-haiku-20241022")
# for model in (model_a, model_b):
#     response = model.invoke("Summarize LCEL in one sentence.")
#     print(response.content)
```

Only the construction line changes (class name, model string). That's the
point of a shared interface: provider-specific plumbing lives inside the
class, not scattered through your call sites.

## Parsing raw output by hand

Before reaching for a framework's built-in output parser, it's worth
understanding the problem it solves. Raw model output is just a string —
you have to check it's well-formed and that the values make sense:

```python
import json

raw_output = '{"label": "positive", "confidence": 0.82, "tags": ["shipping", "praise"]}'

def parse_and_validate(raw: str) -> dict:
    data = json.loads(raw)  # raises on malformed JSON
    missing = {"label", "confidence", "tags"} - data.keys()
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if data["label"] not in {"positive", "neutral", "negative"}:
        raise ValueError(f"unexpected label: {data['label']}")
    conf = data["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence out of range: {conf}")
    if not isinstance(data["tags"], list):
        raise ValueError("tags must be a list")
    return data

print(parse_and_validate(raw_output))
```

Nothing here is a LangChain helper — `json.loads` plus your own key and
range checks. A model can output almost-JSON, wrong types, or a missing
field; validating it explicitly is how you catch that before it breaks
whatever consumes the result.

**Takeaway:** templates are reusable by design, providers are swappable
because of a shared interface, and raw output should be validated by hand
before you trust it.
