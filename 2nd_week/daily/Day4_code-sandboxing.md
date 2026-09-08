# Day 4 — Sandboxing Code-Execution Agents

A tool-calling agent that can only invoke a fixed set of pre-approved functions is easy to reason about — every possible action was reviewed in advance. An agent that can write and execute arbitrary code is far more capable, and far more dangerous, because "arbitrary code" means arbitrary: it can do anything the process it runs in can do.

## Why fixed tools aren't enough, and what goes wrong

Fixed-function tools (`search_database(query)`, `send_email(to, body)`) can only do what their author explicitly wrote. A code-execution tool (`run_python(code)`) can do anything Python can do — powerful for open-ended tasks (data wrangling, quick calculations, file transforms) but open to failure modes a fixed tool never could produce:
- **Destructive file operations** — a generated script with `shutil.rmtree(...)` or a wrong relative path wiping out real project files.
- **Infinite loops / resource exhaustion** — a subtly wrong loop condition pinning a CPU core (or all of them) indefinitely.
- **Unwanted network egress** — generated code silently phoning home, downloading something unexpected, or exfiltrating data it had read access to.

None of these require malicious intent from the model — a plausible-looking bug is enough. The fix isn't "trust the code more," it's running the code somewhere a mistake can't reach anything that matters.

## The shape of a cloud sandbox API

Most hosted code-execution sandbox products converge on a similar shape, roughly:

```python
sandbox = Sandbox.create(timeout=60, network_access=False)

result = sandbox.run_code("print(sum(range(100)))")

sandbox.files.write("/tmp/input.csv", csv_bytes)
sandbox.commands.run("pip install pandas")

sandbox.kill()
```

A `create` call spins up an isolated, ephemeral environment (often a microVM or dedicated container); `run_code` / `commands.run` execute inside it; a `files` interface moves data in and out without touching the host filesystem; a network toggle lets you disable egress entirely for anything that shouldn't be calling out; and `kill` tears the environment down when done. Exact method names vary by provider, but this create → execute → tear-down lifecycle, with an explicit network switch, is close to universal across them.

## A local, DIY approximation (and its limits)

When a hosted sandbox isn't available, a rough local approximation combines a few standard-library pieces:

```python
import resource
import subprocess

def limit_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))                    # 5 CPU-seconds
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)   # 256MB memory

result = subprocess.run(
    ["python3", "-c", generated_code],
    timeout=10,
    preexec_fn=limit_resources,
    capture_output=True,
)
```

Running inside a restricted `exec()` — stripping dangerous builtins like `open`, `__import__`, or `eval` from the globals dict before executing — is a similar-looking pattern people reach for. **None of this is a real security boundary.** `subprocess.run(timeout=...)` and `resource.setrlimit` bound CPU time and memory, not filesystem or network access, and a restricted-builtins `exec()` is well known to be escapable by a sufficiently motivated adversary using nothing but pure Python. This pattern is worth knowing as a cheap first line of defense against *accidental* bad code — never as protection against code you don't trust.

## The isolation spectrum

Roughly, from weakest to strongest:
1. **Process separation** (a plain subprocess) — stops a crash from taking down the parent process; stops almost nothing else.
2. **Containers** (Docker-style namespaces + cgroups) — real filesystem, process, and resource isolation; shares the host kernel, so a kernel-level exploit can still escape.
3. **Kernel-level / VM sandboxing** (gVisor-style intercepted syscalls, or a real lightweight microVM like Firecracker) — the strongest practical isolation, at the cost of more setup and often more latency per sandbox.

The right layer depends on what the agent is allowed to do and what data it can touch — a code-execution agent that only runs data transformations on files it already owns is not the same risk as one that can write to shared infrastructure.

**Takeaway:** code-execution capability is qualitatively riskier than fixed tool calls, and a local `subprocess` + `resource` sandbox is a speed bump against bugs, not a security boundary against determined misuse — for anything untrusted, reach for real process/VM-level isolation.
