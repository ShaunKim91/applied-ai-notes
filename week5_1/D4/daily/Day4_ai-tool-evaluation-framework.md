# Day 4 — A Framework for Evaluating AI Tools

New AI products ship constantly, and any list of named tools goes stale within a few months. What
stays useful is a framework-agnostic way to size up *any* new tool the moment it crosses your
desk — without caring what it's called. Three lenses cover most of what matters: **cost**,
**security**, and **approval-friction**.

## The three lenses

**Cost** — What's the pricing model (seat-based, usage-metered, token-metered)? Is there a free
tier, and what happens the moment you exceed it? Look specifically for hidden usage costs: a
"free" tool that silently bills a connected API key, or a per-seat price that balloons once every
team member is added.

**Security** — Where does your data go once you use this tool? Does it train on your inputs by
default, or is that opt-out (or unavailable to turn off at all)? What's the auth model — SSO
support, API keys stored where, does it request broad account permissions it doesn't need? A tool
that pipes your prompts (which may contain internal data) through an unclear retention policy is
a security question before it's a productivity question.

**Approval-friction** — How hard is this to actually get approved for use at a typical
organization? Does it need procurement sign-off, a compliance review, a security questionnaire, a
data processing agreement? A brilliant tool that takes six weeks of IT review to greenlight may
lose to a mediocre one your team can start using today.

## A 10-minute first-pass checklist

```
[ ] Cost: pricing model identified? free-tier limits and overage cost known?
[ ] Cost: any usage that bills a connected key/account outside the tool's own price?
[ ] Security: data retention / training-on-input policy findable in <2 min?
[ ] Security: auth method (SSO vs. plain API key) and permission scope checked?
[ ] Approval: does org policy require security review before individual use?
[ ] Approval: is there a lighter-weight sandboxed/trial path available first?
[ ] Verdict: green (adopt), yellow (trial only), or red (skip) — with a one-line reason.
```

Run this in ten minutes on any new tool before spending real time integrating it.

## Keeping the framework current, with Day 3's pipeline

The lenses above don't change, but *what* they get applied to does — new tools, new pricing
pages, new security disclosures appear every week. Reuse the `research()` pipeline from Day 3
verbatim for this: point it at a query like "new AI coding assistant pricing changes this month"
or "AI tool data retention policy update," let it search, re-rank, and summarize-with-sources, and
skim the grounded result before your next checklist pass.

```python
landscape_update = research(
    "recent pricing or security policy changes for AI developer tools",
    searcher=lookup_web_stub, ranker=rerank_hits, summarizer=call_llm_stub,
)
```

**Takeaway:** judge any AI tool on cost, security, and approval-friction with a fast checklist — then let your own search-grounded research pipeline keep that judgment from going stale.
