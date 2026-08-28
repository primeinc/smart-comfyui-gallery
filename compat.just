# The consumer compatibility suite.
#
# Answers one question by execution: what is the minimum durable state this
# application must keep after an expensive face observation so that every
# supported downstream consumer can still be served without reopening the
# source file or re-running the model.
#
# Four lanes, and `run` chains them. Each writes machine-generated evidence
# under compat/generated/ and each is red until it earns green.
#
# The interpreter is the project venv, not the system one: the system python
# has no numpy, so `python -m compat.harness.run` there fails with an import
# error rather than a verdict -- which reads like a broken suite instead of a
# missing environment.
#
# Runtimes come from a group `uv sync` does NOT install by default:
#
#   uv sync --group compat
#
# Nothing under db/ or sg_web/ may import any of it, which is why the group is
# separate and why compat/ lives outside the application import graph.

python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Every lane in order: pins, producer, cases, matrices, attack. Stops at the first red
run: pins producer cases matrices attack

# Three independent layers, and none excuses the others: a clone at the right
# commit says nothing about what the interpreter will import, and neither says
# anything about which model files computed the numbers.

# Clones, installed wheels and weight files against what the manifest pins
pins:
    {{ python }} compat/harness/provenance.py

# Runs the producer over real corpus photographs and records every field it
# emits -- including the ones nothing currently reads, which are precisely the
# ones at risk of being dropped because no column names them.

# One real observation pass, inventoried by shape, dtype and byte cost
producer:
    {{ python }} -m compat.producers.inventory

# Red while any declared consumer is unexercised. That is deliberate: a
# consumer which disappears when nothing runs it is how a suite reports
# success it did not earn.

# Baseline, replay and ablation for every consumer that has a runner
cases:
    {{ python }} -m compat.harness.run

# Reads only what the lanes above wrote, so a hand-typed conclusion cannot
# enter. Red while the population is incomplete, for the same reason `cases`
# is: a matrix reporting green over part of a population is the failure this
# lane exists to prevent.

# Generated views over the evidence, never hand-maintained
matrices:
    {{ python }} -m compat.harness.matrices

# A gate that has never gone red is not known to work. This corrupts pins,
# blob digests, weight paths and the evidence itself, and asserts the checks
# notice. An attack the gate does NOT see is a red result: every green ever
# reported for that class was uninformative.

# Attack the suite and prove its gates can fail
attack:
    {{ python }} -m compat.harness.attack

# The licence is cc-by-nc-nd-4.0, so nothing here copies, resizes or rewrites
# a file. A digest rather than a path is what keeps the evidence honest: a
# file edited in place under the same name would otherwise invalidate every
# baseline recorded against it without changing one line of the index.

# Index the corpus by path and sha256, never vendored
corpus:
    {{ python }} -m compat.corpus.index

# Regenerate the deterministic fixtures for the primitive tier
fixtures:
    {{ python }} -m compat.primitives.build

# Project-wide on purpose: `pyrefly check compat/` reports clean on code that
# `pyrefly check` rejects, because a targeted run does not read the project
# configuration. Measured, not assumed -- it happened here.

# Every gate this tree holds compat/ to, run project-wide
check:
    {{ python }} -m ruff format --check compat/
    {{ python }} -m ruff check compat/
    {{ python }} -m pyrefly check
    {{ python }} -m ty check
