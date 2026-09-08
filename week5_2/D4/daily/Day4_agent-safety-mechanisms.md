# Day 4 — Agent Safety Mechanisms

An agent that can loop and call tools on its own needs guardrails a plain
chatbot doesn't. Four concrete mechanisms, then one runner that combines
them.

## The four mechanisms

1. **Tool allowlisting, default-deny.** An agent may call only tools
   explicitly marked allowed; anything not on the list is refused, no
   "allow unless blocked."
2. **Step limit.** A hard cap on loop iterations so a confused agent can't
   spin forever — the same guard from earlier, promoted to a formal rule.
3. **Human-in-the-loop approval gate.** High-stakes actions (sending money,
   deleting data, anything hard to undo) pause the loop until a person
   explicitly approves.
4. **Cost cap.** A running total of estimated dollars spent; once it crosses
   a budget, further calls are refused regardless of what else checks out.

## One guarded runner, checked in a deliberate order

```python
class GuardedAgentRunner:
    def __init__(self, allowed_tools, max_steps, budget_usd, approve_fn):
        self.allowed_tools = set(allowed_tools)
        self.max_steps = max_steps
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.step_count = 0
        self.approve_fn = approve_fn
        self.log = []

    def call_tool(self, name, args, est_cost, needs_approval=False):
        record = {"tool": name, "args": args, "cost": est_cost, "status": None}

        if self.step_count >= self.max_steps:          # 1) cheapest: counter check
            record["status"] = "denied_step_limit"
        elif self.spent_usd + est_cost > self.budget_usd:  # 2) also a counter check
            record["status"] = "denied_cost_cap"
        elif name not in self.allowed_tools:            # 3) a set lookup
            record["status"] = "denied_not_allowed"
        elif needs_approval and not self.approve_fn(name, args):  # 4) slowest: a person
            record["status"] = "denied_no_approval"
        else:
            record["status"] = "allowed"
            self.step_count += 1
            self.spent_usd += est_cost

        self.log.append(record)
        return record["status"] == "allowed"
```

The ordering is deliberate: step-limit and cost-cap are O(1) counter
comparisons, so they run first and reject cheaply. The allowlist lookup is
still fast but does slightly more work. Human approval goes last on purpose
— it's the only check that can block on a real person, so it should run
only once every cheaper, instant check has already passed.

## Turning the log into an audit trail

```python
import pandas as pd

audit_df = pd.DataFrame(runner.log)
suspicious = (
    audit_df.groupby(["tool", "args"])
    .size()
    .reset_index(name="attempts")
    .query("attempts >= 5")
)
```

Grouping by `(tool, args)` and flagging combinations attempted an unusually
high number of times — especially ones repeatedly denied — is a simple,
effective heuristic for catching a stuck or misbehaving agent without
needing any real anomaly-detection model.

## Escalating to a bigger model when stuck

A pattern worth knowing: route routine steps to a cheap local model, and
escalate to a larger cloud model only for one best-effort final attempt once
the local model exhausts its step budget without success. Most requests
never need the expensive model, so this can meaningfully cut average
per-request cost and latency, while still giving hard cases a shot at the
stronger model's capability instead of just failing outright.

**Takeaway:** cheap, deterministic checks first; the slow human check last — and log everything so misuse is visible after the fact, not just prevented in the moment.
