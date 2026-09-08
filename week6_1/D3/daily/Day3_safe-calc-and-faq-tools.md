# Day 3: A Calculator Tool and an FAQ Tool, Wired by Rules

Not every question belongs to the language model. Exact arithmetic and exact
facts are better served by small, deterministic tools than by free-form
generation. Today: a safe calculator tool, a keyword-matched FAQ tool, and a
rule-based dispatcher between them.

## Why not just `eval()`?

`eval()` runs arbitrary Python — a user could pass `__import__('os').system(...)`
and it would execute. Even `ast.literal_eval`, often suggested as the "safe"
alternative, doesn't help here: it only parses literal constants and
containers (numbers, strings, tuples, lists, dicts, sets, booleans, `None`,
limited unary +/-), not an actual operation between two literals —
`ast.literal_eval("12 * 8")` raises `ValueError`, because multiplication
isn't part of its restricted grammar. We need something that understands
arithmetic but nothing else.

## A whitelist AST evaluator

Parse the expression, then walk the tree, allowing only arithmetic nodes:

```python
import ast
import operator

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"disallowed expression: {type(node).__name__}")

def calc(expr: str):
    return _eval(ast.parse(expr, mode="eval"))
```

Names, attribute access, subscripts, and function calls all fall through to
the final `raise` — `calc("__import__('os')")` raises `ValueError` instead
of running anything. `calc("12 * 8")` returns `96`.

## A deterministic FAQ tool

```python
_FAQ = [
    (("refund", "money back"), "Refunds post within 5-7 business days after we receive the return."),
    (("hours", "open"), "Support is staffed 9am-6pm, Monday through Friday."),
    (("shipping", "delivery"), "Standard shipping takes 3-5 business days."),
]

def faq(question: str):
    q = question.lower()
    for keywords, answer in _FAQ:
        if any(kw in q for kw in keywords):
            return answer
    return None  # no model call — same question always gets the same answer, instantly
```

## Routing between them

```python
def handle(user_input: str) -> str:
    has_digit = any(c.isdigit() for c in user_input)
    has_operator = any(op in user_input for op in "+-*/")
    if has_digit and has_operator:
        try:
            return f"= {calc(user_input)}"
        except ValueError as e:
            return f"couldn't evaluate that: {e}"
    answer = faq(user_input)
    return answer if answer else "no matching tool for that yet."
```

## A known limitation

This dispatcher only recognizes digits and operator symbols. "What is twelve
times eight" or "how many is a dozen doubled" has neither — it slides past
the calc check, misses the FAQ keywords too, and falls through to the
generic fallback. Rule-based routing is fast and predictable, but it only
catches the phrasings you anticipated.

**Takeaway:** deterministic tools beat free-form generation for exact
numbers and exact facts — but a rule-based router is only as good as the
patterns you thought to write.
