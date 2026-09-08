# Day 2 — The Agent Framework Landscape, and Building a Sliver of One

Yesterday we hand-rolled a ReAct loop. Real-world agent tooling generally
falls into a few roles that are worth telling apart before picking anything:

1. **Autonomous agent platforms/runtimes** — take a goal, break it into
   sub-tasks, and execute them somewhat independently, managing the loop,
   memory, and tool dispatch for you. Powerful, but you're trusting the
   platform's planner with a lot of the decision-making.
2. **Sandboxing / security wrapper layers** — sit *around* an agent (of any
   kind) and constrain what it can actually touch: filesystem paths, network
   calls, shell commands. They don't plan or reason; they enforce a boundary
   no matter what the agent decides to try.
3. **Framework-agnostic orchestration layers** — sit *above* several
   different underlying agent libraries so an application can swap engines
   (one vendor's runtime today, a different one next quarter) without
   rewriting the calling code.

None of these replace understanding the loop itself — they automate pieces
of what we wrote by hand on Day 1.

## A minimal framework, built to see what's hidden

```python
import time

class MiniAgentFramework:
    def __init__(self):
        self.tools = {}
        self.audit_log = []

    def register_tool(self, name, fn, allowed=True):
        self.tools[name] = {"fn": fn, "allowed": allowed}

    def run_tool(self, name, arg):
        entry = {"tool": name, "arg": arg, "ts": time.time()}
        if name not in self.tools or not self.tools[name]["allowed"]:
            entry["status"] = "denied"
            self.audit_log.append(entry)
            raise PermissionError(f"tool '{name}' is not registered or not allowed")
        result = self.tools[name]["fn"](arg)
        entry["status"] = "ok"
        entry["result"] = result
        self.audit_log.append(entry)
        return result
```

`register_tool` and `run_tool` plus an `audit_log` are the same three things
every larger framework does — they just add planners, retries, and memory
stores on top. Writing the sliver by hand makes the abstraction legible.

## An LCEL-style pipe chain

Several ecosystem frameworks let you compose `prompt | model | parser` with
the `|` operator. It's a thin wrapper around function composition — here's a
schematic version wrapping a local model:

```python
class LocalModelRunnable:
    """Illustrative wrapper — a real integration would subclass an actual
    base class such as LangChain's `LLM` class and implement `_call`."""
    def __init__(self, generate_fn):
        self.generate_fn = generate_fn

    def __or__(self, other):
        return ChainLink(self, other)

    def invoke(self, x):
        return self.generate_fn(x)

class ChainLink:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def invoke(self, x):
        return self.second.invoke(self.first.invoke(x))

summarize_chain = prompt_template | LocalModelRunnable(local_generate) | strip_parser
answer = summarize_chain.invoke({"text": "..."})
```

**Takeaway:** frameworks package the loop, the tool registry, and the audit trail — knowing what's underneath lets you debug them (or skip them) with confidence.
