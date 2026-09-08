# Day 1: LangChain and the LCEL Pipe

In an earlier lesson we hand-built a mini agent loop from scratch: a prompt string,
a manual call to a model, and some if/else logic to decide what happened next.
That taught us what an agent actually *is*. Today we look at why a framework
like LangChain exists, and how its pipe syntax (`|`) lets you compose steps
into a runnable chain.

## Why reach for a framework at all

A hand-rolled loop works, but every project ends up re-solving the same
problems: templating prompts, swapping models, retrying failed calls,
parsing output. A framework packages those patterns so you compose them
instead of rewriting them. The trade-off is real, though: you take on a
dependency, an extra layer of abstraction between you and the raw API call,
and a learning curve for the framework's own conventions. Framework code is
not automatically "better" code — it's a bet that reuse outweighs the added
indirection.

## Runnables and the `|` operator

LangChain's building blocks (prompt templates, models, parsers) all implement
a common `Runnable` interface with an `.invoke()` method. The `|` operator is
overloaded so that chaining two Runnables together with `a | b` returns a
`RunnableSequence`: calling `.invoke()` on the sequence runs `a.invoke()`
first, then feeds its output straight into `b.invoke()`. Chain three things
together (`prompt | llm | parser`) and you get a three-step pipeline with the
same `.invoke()` interface as any single piece.

## A chain that needs no API key

Mock the model first — it proves the wiring works before you spend a single
API call.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

class ScriptedModel(Runnable):
    """Stands in for a chat model: same .invoke() shape, zero network calls."""
    def invoke(self, prompt_value, config=None, **kwargs):
        text = prompt_value.to_string()
        return f"MOCK-REPLY: read {len(text)} chars starting '{text[:24]}'"

def to_upper(text: str) -> str:
    return text.upper()

prompt = PromptTemplate.from_template("Explain {topic} in one short sentence.")
chain = prompt | ScriptedModel() | to_upper

print(chain.invoke({"topic": "binary search"}))
```

`to_upper` is a plain function, not a hand-built Runnable — LangChain wraps
ordinary callables automatically when you pipe into them, so it slots into
the chain without extra ceremony.

## Swapping in a real model later

```python
# from langchain_openai import ChatOpenAI
# real_model = ChatOpenAI(model="gpt-4o-mini")  # needs a real API key
# chain = prompt | real_model | to_upper
```

This is only a one-line swap *if* the mock's interface truly matches the real
one. It doesn't, quite: a real chat model returns a message object with a
`.content` attribute, not a plain string, so `to_upper` would break until you
also swap in a parser that reads `.content`. The lesson: matching interfaces
is what makes components swappable, not just having the same variable name.

**Takeaway:** LCEL's `|` composes Runnables into a pipeline where each
step's output becomes the next step's input — mock the pieces first, then
swap in real ones once the interface is proven.
