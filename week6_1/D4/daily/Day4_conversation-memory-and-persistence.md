# Day 4: Memory Is Just Re-Sending the Conversation

A language model call is stateless: nothing is remembered between one
`.invoke()` and the next. When a chatbot "remembers" what you said three
messages ago, that's because the app re-sends the earlier turns as part of
every new prompt. "Memory" is a re-injection strategy, not a built-in
capability of the model.

## Unbounded history gets expensive

```python
def build_prompt(history, new_message):
    convo = "\n".join(f"{role}: {text}" for role, text in history)
    return f"{convo}\nuser: {new_message}\nassistant:"

history = []
for turn in range(1, 21):
    user_msg = f"question {turn} about the order"
    prompt = build_prompt(history, user_msg)
    history.append(("user", user_msg))
    history.append(("assistant", f"answer {turn}"))
    if turn % 5 == 0:
        print(f"turn {turn}: prompt is {len(prompt)} chars")
```

Every turn re-sends *all* prior turns, so the prompt (and the cost/latency of
each call) grows roughly linearly with conversation length — left unchecked,
a long chat eventually hits the model's context limit.

## Mitigation 1: windowed memory

Keep only the last N turns; drop everything older.

```python
def windowed_history(history, max_turns=6):
    return history[-max_turns:]
```

Simple and cheap, but older context is gone outright — fine for a support
bot answering one question at a time, worse for a conversation that
circles back to something said much earlier.

## Mitigation 2: summarized memory

Collapse old turns into a running summary instead of dropping them.

```python
def compact_history(history, keep_recent=6, summarize=None):
    if len(history) <= keep_recent:
        return history
    old, recent = history[:-keep_recent], history[-keep_recent:]
    old_text = "\n".join(f"{r}: {t}" for r, t in old)
    summary = summarize(old_text) if summarize else old_text[:200] + "..."
    return [("system", f"earlier conversation summary: {summary}")] + list(recent)
```

`summarize` would typically be another LLM call — you spend one extra call
to keep every later prompt smaller, trading a little cost now for a lot of
cost saved over the rest of the conversation.

## Persisting history across restarts

```python
import json
from pathlib import Path

def save_history(history, path="chat_history.json"):
    Path(path).write_text(json.dumps(history, indent=2))

def load_history(path="chat_history.json"):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []
```

A JSON file works for a single user; a small SQLite table works once you
have many conversations to look up by id. Either way, the app process can
restart without losing the conversation.

## A real security note: PII in persisted history

If a user pastes an email address, phone number, or ID number into the
chat, that text is stored verbatim in whatever file or database holds the
history — nothing about "memory" strips it out automatically. That matters
for data-retention and compliance reasons: persisted logs are exactly the
kind of place PII quietly accumulates. Either redact obvious patterns before
writing to disk, or explicitly decide what fields are safe to persist at all.

**Takeaway:** memory is re-injected history, not model magic — bound it
(window or summary) before it grows unchecked, and treat anything you
persist as data that needs the same care as any other stored record.
