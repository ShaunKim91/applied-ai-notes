# Day 1 — Agents vs. Chatbots, and the ReAct Loop

A chatbot answers one message with one reply. An **agent** is different: it
repeatedly (1) observes the current state of its environment, (2) reasons
about what to do next, and (3) takes an action — cycling through that loop
until it hits a goal or a stop condition. The loop, not the single reply, is
what makes it an agent.

## ReAct: Thought → Action → Observation

ReAct prompting gets a language model to produce a structured text trace
instead of a plain answer: a `Thought` line explaining its reasoning, an
`Action` line naming a tool call, then an `Observation` line where the tool's
real output gets fed back in. The model reads that observation and writes a
new thought, repeating until it emits a final answer.

```
Thought: I need the current value of 18 * 47 before I can answer.
Action: calc(18 * 47)
Observation: 846
Thought: I now have enough to answer.
Final Answer: 846
```

## Hand-rolling the loop

No framework needed — just a Python `while` loop that calls the model,
parses its last line, dispatches a tool if one was requested, and appends
the observation to the running transcript.

```python
def run_agent(model, tools, question, max_steps=6):
    transcript = FEW_SHOT_EXAMPLES + f"\nQuestion: {question}\n"
    for step in range(max_steps):
        chunk = model.generate(transcript, stop=["Observation:"])
        transcript += chunk
        if "Final Answer:" in chunk:
            return chunk.split("Final Answer:")[-1].strip()
        action_line = next(l for l in chunk.splitlines() if l.startswith("Action:"))
        tool_name, arg = parse_action(action_line)
        result = tools[tool_name](arg)
        transcript += f"Observation: {result}\n"
    return "Gave up after max_steps without a final answer."
```

Two details matter more than they look:

- **Few-shot examples.** A small local model rarely emits a clean, parseable
  `Action: tool(arg)` line on its own — it drifts into prose. Priming the
  prompt with 2–3 worked examples of the exact `Thought/Action/Observation`
  format is usually what turns "sometimes parses" into "reliably parses."
- **A step-limit guard.** `max_steps` above is a safety valve. Without a hard
  cap, a model that never emits `Final Answer:` (or that loops between the
  same two actions) runs forever — every hand-rolled agent loop needs a stop
  condition that isn't "trust the model to stop itself."

## Chatbot mode vs. agent mode, same model

Take one small local model and ask it, in plain chatbot mode, "What's
14.7% of 2,830?" It will often answer fast and fluently — and wrong, because
arithmetic on unfamiliar numbers isn't something language models compute
reliably from pattern-matching alone. Run the *same* model through the ReAct
loop above with a `calc` tool available, and the trace shows it recognizing
it needs the tool, calling it, reading back an exact observation, and
reporting the correct number. Nothing about the model's weights changed —
only the loop wrapped around it did.

**Takeaway:** an agent is a chatbot plus a loop plus tools — ReAct just gives that loop a parseable format to reason and act in.
