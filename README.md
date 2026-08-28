# Two rules

## 1. Do not write a test lane you have not read the fixtures for

The suite is not a pile of independent functions. It has a shape, and the
shape is in `tests/conftest.py` and `pytest.ini`. Changing how the suite is
RUN — the worker count, the distribution mode, which files are selected — is
changing that shape. Read it first.

What it costs when you don't:

A recipe was added that ran `pytest <files> -m slow -n 3 --dist loadfile`.
It looked like a convenience wrapper. It was a change to the parallelism of a
suite whose server pool (`tests/conftest.py` `_servers`) is `scope="session"`
— which under xdist means **one pool per worker** — while the count that pool
uses to decide how many spare application servers to boot ahead
(`sg_browser_modules`) is taken from the **full collection**.

So every worker read the whole suite's browser-module backlog as its own and
kept booting litestar subprocesses for modules the scheduler had already
given to somebody else. The pool's own docstring says "at most two servers
are alive at a time". That is true per worker and false per run: at `-n 4`
it is up to eight application subprocesses, each importing forty-five
modules, behind four chromiums.

The run appeared to hang at 36%. It was oversubscription, and the fix is one
line — the pool takes this worker's share, not the run's:

```python
wanted = getattr(request.config, "sg_browser_modules", 0)
workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1") or 1)
wanted = -(-wanted // workers)
```

Same three files, same `-n 3 --dist loadfile`: 4m39s and still going, before.
**75 passed in 18.32s**, after.

The rules that follow from it:

- **Read `tests/conftest.py` and `pytest.ini` before touching how tests run.**
  Both files carry their reasoning in comments. They were written down so
  nobody has to rediscover them.
- **A wrapper is not a fix.** Deleting the recipe that surfaced this changed
  nothing about the defect. Removing the thing that showed you a problem is
  not repairing the problem.
- **Do not name a cause you have not tested.** "Worker contention" was a
  guess stated as a finding, and it sent the search in the wrong direction.
  Reproduce it, or say you do not know yet.
- **Measure one file alone before blaming the combination.** Each of the
  three files here was fast on its own. That fact was available in seconds
  and it was the whole shape of the answer.
- **A session-scoped fixture under xdist is per worker.** Anything it counts
  from the full collection is wrong by a factor of the worker count.

## 2. Padding timeouts

A timeout is not a safety margin. It is the decision about **how long a human
sits in front of a wedged command before anything tells them.**

Pad it by 20x and you have not made the run safer. You have chosen, on that
person's behalf, that they will lose ten minutes to a command that was never
going to finish — and they will spend those minutes watching a cursor,
unable to tell a slow run from a dead one.

Every lane in this repository is measured. Use the measurement.

```
just check          7s          ->  15s
just test           2s          ->  10s
just test-slow      51s         ->  90s
one test module     under 20s   ->  30s
```

**Measured cost plus about half. Never a round order-of-magnitude guess.**

No measurement yet means take one small step and measure it. It does not
mean pick a big number.
