# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]
set script-interpreter := ['bash', '-euo', 'pipefail']

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The fast lane: every test that does not drive a browser, boot a server or
# open a database. tests/conftest.py SLOW_FIXTURES is the criterion, derived
# from each test's own fixture closure.

# The exit code is NOT swallowed: pytest exits 5 when it collected nothing,
# the only signal that the selection stopped matching any test.
test: web::build
    {{ python }} -m pytest tests/ -m "not slow"

# The suite, all of it: real sample libraries, real browsers, real migration
# chains. Minutes rather than seconds, which is why it is in neither
# `just test` nor `just check`.
test-slow: web::build
    {{ python }} -m pytest tests/ -m slow -n 4 --dist loadfile

# Both gates and the whole suite. On green the COMMITTED tree's hash lands
# in .proven-tree, so pushing an already-proven tree re-derives nothing.

# A dirty working tree still runs everything and records nothing: the marker
# claims a committed tree passed, and a dirty run proved something else.
[doc('Both gates + the whole suite; on green, record the proven tree in .proven-tree')]
[script]
prove:
    cd "$(git rev-parse --show-toplevel)"
    # Read BEFORE the run and compared after: a commit landing mid-run must
    # not be stamped with a proof that ran against its parent. The capture
    # has to happen first, so these are inline calls rather than deps.
    tree=$(git rev-parse 'HEAD^{tree}')
    just check-deep
    just test-slow
    if [ -n "$(git status --porcelain)" ] || [ "$(git rev-parse 'HEAD^{tree}')" != "$tree" ]; then
        echo "the tree moved while the proof ran: everything was green, but nothing is recorded"
        exit 0
    fi
    printf '%s\n' "$tree" > .proven-tree
    echo "proven: $tree"

# What pre-push runs (lefthook.yml). An already-proven tree passes in
# milliseconds; anything else runs the AFFECTED slice, selected by
# pytest-testmon from measured per-test file coverage in `.testmondata`.

# `--testmon-forceselect` keeps selection active beside `-m slow`, which
# disables it otherwise (tarpas/pytest-testmon testmon/configure.py:65-76).
# A missing `.testmondata` makes the first run a full one that seeds it.

# `-p pytest-testmon` turns the plugin back on: pytest.ini blocks it for
# every other run, and an explicit `-p` overrides the `-p no:` in addopts.
# Selection is a slice, not the proof; `just prove` is the only writer.

# It runs in a DETACHED WORKTREE AT HEAD. A pre-push gate answers "is the
# commit I am pushing safe" and pytest imports whatever is on disk, so
# uncommitted work can fail a push that lacks it, or make a bad commit pass.

# A worktree rather than a stash: `git worktree add --detach` writes nothing
# into the working tree and takes no lock on it.

# Four things a fresh checkout cannot carry, so they are copied in:

#   sg_web/static/build   gitignored; `web::build` above made it
#   benchmarks/results    gitignored; without it four throughput tests fail

#   .testmondata          copied in AND back out, so the run teaches the next
#   .venv/Scripts on PATH tests/conftest.py boots the app with bare `litestar`

# And one it cannot carry at all: a `.venv` DIRECTORY beside pyproject.toml.
# `sg_web/__main__.py interpreter()` looks at that exact path and nothing
# else, and the two tests below assert it resolves to the running python.

# They are DESELECTED rather than satisfied by a link: `git worktree remove
# --force` follows a link and deletes the environment it points at. A
# pre-push gate must not be able to do that.

# So they run IN THE CHECKOUT at the end of this recipe. A test the gate
# never runs is not gating anything, and both ask about the installation
# rather than the commit, which is the only place that can answer.

[doc('Pre-push gate: skip a proven tree, else run the affected slice of the suite')]
[script]
prove-push: web::build
    cd "$(git rev-parse --show-toplevel)"
    root=$(pwd)
    tree=$(git rev-parse 'HEAD^{tree}')
    if [ -f .proven-tree ] && [ "$(cat .proven-tree)" = "$tree" ]; then
        echo "tree already proven by 'just prove'; nothing to re-derive"
        exit 0
    fi
    pushed=$(mktemp -d)/pushed
    git worktree add --detach --quiet "$pushed" HEAD
    trap 'git -C "$root" worktree remove --force "$pushed" >/dev/null 2>&1; rm -rf "$(dirname "$pushed")" || echo "warning: the pushed worktree at $pushed was left behind" >&2; git -C "$root" worktree prune >/dev/null 2>&1' EXIT
    # `<dir>/.` into an existing `<dir>/`, never `cp -r <dir> <dir>`: the
    # second nests when the destination exists, and both do in a checkout.
    # Nesting is silent and reads as the gate failing on a green commit.
    mkdir -p "$pushed/sg_web/static/build" "$pushed/benchmarks/results"
    cp -r sg_web/static/build/. "$pushed/sg_web/static/build/"
    if [ -d benchmarks/results ]; then cp -r benchmarks/results/. "$pushed/benchmarks/results/"; fi
    if [ -f .testmondata ]; then cp .testmondata "$pushed/.testmondata"; fi
    cd "$pushed"
    launch=tests/test_the_documented_launch_serves_a_whole_application.py
    # `tests/test_the_corpus_spans_the_shape.py` resolves `sg-corpus` beside
    # the repository from the test file, so a worktree under /tmp looks
    # beside /tmp and six coverage tests report an empty corpus.

    # `SG_CORPUS` is the seam that exists for this, so the gate measures the
    # same corpus a developer does.
    if [ -d "$root/../sg-corpus" ]; then export SG_CORPUS="$root/../sg-corpus"; fi
    # And the RUN that scanned it: three of those six ask a library that read
    # the corpus, not the corpus. `tests/needs.py _served_db` resolves it at
    # `sg-run` the same way, wrong in a worktree for the same reason.

    # Without it every rung reports UNKNOWN_NOT_MEASURED, which those tests
    # correctly refuse to read as reached.
    if [ -d "$root/../sg-run" ]; then export SG_HOME="$root/../sg-run"; fi
    PATH="$root/.venv/Scripts:$root/.venv/bin:$PATH" PYTHONPATH=. \
      "$root/{{ python }}" -m pytest tests/ -m slow -n 4 --dist loadfile \
      --deselect "$launch::test_an_interpreter_without_a_server_is_handed_to_the_one_that_has_it" \
      --deselect "$launch::test_the_environment_this_suite_runs_in_is_the_one_the_handover_targets" \
      -p pytest-testmon --testmon --testmon-forceselect
    settled=$?
    if [ -f "$pushed/.testmondata" ]; then cp "$pushed/.testmondata" "$root/.testmondata"; fi
    # The two deselected above, run HERE. Deselecting them from the worktree
    # pass is not the same as not running them.

    # The checkout is the only place they mean anything: both ask whether
    # THIS machine has an environment beside the source, which no commit
    # changes and no temporary copy of the source can answer.
    cd "$root"
    "$root/{{ python }}" -m pytest       "$launch::test_an_interpreter_without_a_server_is_handed_to_the_one_that_has_it"       "$launch::test_the_environment_this_suite_runs_in_is_the_one_the_handover_targets"       -q --no-header
    handover=$?
    if [ "$settled" -ne 0 ]; then exit $settled; fi
    exit $handover

# The PWA's rasters, drawn from the mark: icons, the iOS splash set, and the
# install-sheet screenshots photographed off the real app over a generated
# library (sg_web/branding.py). Commit what it writes.
[doc('Regenerate the PWA icons, iOS splash set, and install screenshots')]
pwa-assets:
    {{ python }} -m sg_web.branding all

# Every test alone, one process each: a test that leans on a sibling's writes
# passes in module order and fails here, which is how the affected-test
# selector will eventually run it.

# Slow by construction, an import and a world per test. An audit, not a gate.
[doc('Run every test in its own process; order-dependent tests fail here first')]
[script]
audit-isolation:
    cd "$(git rev-parse --show-toplevel)"
    # The audit serves the LIVE working tree to real browsers. An edit or a
    # rebuild underneath it makes every later measurement a picture of a torn
    # tree, so the tree is fingerprinted first and a moved verdict is refused.
    opened=$(git rev-parse 'HEAD^{tree}'; git status --porcelain)
    # -q -q -q: pytest.ini pins -vv and verbosity is a counter; net -1
    # is the one level where --collect-only prints one id per line.
    listed=$({{ python }} -m pytest tests/ --collect-only -q -q -q --no-header)
    tests=$(printf '%s\n' "$listed" | sed -n 's/^\(tests\/[^ ]*::[^ ]*\)$/\1/p')
    total=$(printf '%s\n' "$tests" | sed '/^$/d' | wc -l)
    if [ "$total" -eq 0 ]; then
        echo "collection produced no test ids; refusing to report an empty audit as clean"
        printf '%s\n' "$listed"
        exit 2
    fi
    kept=$(mktemp)
    broke=0
    while IFS= read -r test; do
        [ -n "$test" ] || continue
        if ! one=$({{ python }} -m pytest "$test" -q --no-header -p no:cacheprovider 2>&1); then
            broke=$((broke + 1))
            echo "ALONE-FAILS: $test"
            printf '=== %s ===\n%s\n' "$test" "$one" >> "$kept"
        fi
    done <<< "$tests"
    if [ "$(git rev-parse 'HEAD^{tree}'; git status --porcelain)" != "$opened" ]; then
        echo "INVALID: the tree changed while the audit ran; nothing above is a verdict"
        exit 3
    fi
    echo "order-dependence audit: $total tests, $broke fail alone"
    if [ "$broke" -gt 0 ]; then
        echo "full output of each failure: $kept"
        exit 1
    fi

# Ruff over the Python, Biome over the browser source (biome.json: the
# first-party JS and CSS, never the vendored htmx), and this repository's own
# structural rules (sglint).

# Three separate programs reading three separate things, so they run
# together and the lane costs whatever the slowest one costs.
[parallel]
lint: lint-python lint-web lint-structure

[private]
lint-python:
    {{ python }} -m ruff check .

[private]
lint-web:
    npm run --silent lint

[private]
lint-structure:
    {{ python }} -m sglint

# Ruff and Biome format in report-only mode; never rewrites
[parallel]
fmt-check: fmt-python fmt-web

[private]
fmt-python:
    {{ python }} -m ruff format --check .

[private]
fmt-web:
    npm run --silent format-check

# The cross-module inference neither ruff nor esbuild can do. Both Python
# checkers fit the ten-second gate, which is what keeps them here rather than
# in `check-deep`: a type check nobody waits for is one nobody reads.

# BOTH, because they do not agree and neither is a superset. pyrefly reports
# `unnecessary-type-conversion` and `non-convergent-recursion`, which ty has
# no rule for; ty reports `redundant-cast` where pyrefly is silent.

# Both halves always run. As a dependency with a body, the body was skipped
# whenever the dependency failed, so a red browser source meant no Python was
# type checked at all.
[parallel]
types: web::types types-python

[parallel]
[private]
types-python: types-ty types-pyrefly

[private]
types-ty:
    {{ python }} -m ty check

[private]
types-pyrefly:
    {{ python }} -m pyrefly check

# Repository hygiene: the git index, line endings, the requirements
# file, the test command, the evidence stamp -- sglint's SG8xx, which ask
# git; no test ever does
repo-check:
    {{ python }} -m sglint --repo

# THE GATE, held to ten seconds. `just budget` proves it, and what does not
# fit moves to `check-deep` and says so: a gate nobody waits for is a gate
# people commit around.

# `web::fresh` runs FIRST and alone: it deletes and rebuilds
# sg_web/static/build, which biome would otherwise be walking at the same
# moment. Everything after it is independent and runs together.
[doc('The gate. Held to ten seconds; `just budget` proves it')]
check: web::fresh gates

[parallel]
[private]
gates: api::check lint fmt-check web::types types-python web::unit db-check prose

# Comment and docstring prose, against CONTRIBUTING.md. Vale reads Python
# through tree-sitter, so it sees comments and docstrings, never string
# literals or code.

# `--no-global` is load-bearing: Vale loads a per-user config last and lets it
# override, so without it a developer's own file changes what this reports.

# Exit 1 is a finding and 2 is Vale itself failing, reported apart so a broken
# config cannot read as a clean tree.
[private]
[script]
prose:
    if ! command -v vale >/dev/null 2>&1; then
      echo "vale is not installed; the prose gate cannot run (scoop install vale)" >&2
      exit 1
    fi
    # Banded, on the same grounds as [tool.ruff].required-version: a linter
    # that gains rules on an upgrade rewrites the tree. The floor is where
    # code-aware Python linting and per-style severity landed.
    have=$(vale --version | rg -o '[0-9]+\.[0-9]+\.[0-9]+')
    major=${have%%.*}
    rest=${have#*.}
    minor=${rest%%.*}
    if [ "$major" -ne 3 ] || [ "$minor" -lt 19 ]; then
      echo "vale $have is outside the supported band >=3.19,<4" >&2
      exit 1
    fi
    vale --no-global --minAlertLevel=error db vision sg_web metaparse sglint story_renderers tests benchmarks
    rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "vale exited 2: its own configuration or runtime failed, not the tree" >&2
    fi
    exit "$rc"

# What could not be made to fit ten seconds, and not less important:
# repo-check keeps a clone honest, and `types-elsewhere` is the only thing
# that reads the platforms this machine is not.

# repo-check does two full `git checkout-index -a` into temporary trees plus
# a scratch repository, on a platform where each git spawn is expensive.
[doc('What cannot fit ten seconds: repo hygiene, and the platforms this machine is not')]
[parallel]
check-deep: repo-check types-elsewhere

# The same two checkers, at the platforms nobody here is running. `check`
# runs them at each tool's default, so platform-conditional code is read for
# the developer's machine alone.

# `python-platform = "all"` is NOT this: "all" takes the UNION of the
# platform stubs, so a win32-only name resolves under it and the diagnostic
# disappears. A pass per platform is the only form that checks a platform.
[parallel]
[private]
types-elsewhere: types-linux types-darwin

[private]
types-linux:
    {{ python }} -m ty check --python-platform linux
    {{ python }} -m pyrefly check --python-platform linux

[private]
types-darwin:
    {{ python }} -m ty check --python-platform darwin
    {{ python }} -m pyrefly check --python-platform darwin

# Everything: the gate, the deep gate, the suite, and the real run walked.
[doc('Everything: both gates, the suite, and the real run walked')]
check-all: check check-deep test-slow smoke

# The budget, enforced rather than promised. This recipe re-derives the
# timings, so they are never copied into prose where they would go stale
# unread.

# PER LANE, because the lanes are not the same size. The fast lane has a flat
# duration distribution and no tail to cut, so holding it to `check`'s clock
# could only mean dropping tests on the criterion "whatever fits".

# The lanes run in parallel, so the clock is the slowest one: sglint. That,
# not the newer gates, is what to make faster if the budget must come down.
[doc('Prove `just check` and `just test` each stay inside their measured budget')]
[script]
budget:
    over=0
    for lane in check test; do
      case "$lane" in
        check) allowed=20000 ;;
        test)  allowed=30000 ;;
      esac
      start=$(date +%s%N)
      just "$lane"
      spent=$(( ($(date +%s%N) - start) / 1000000 ))
      if [ "$spent" -gt "$allowed" ]; then
        echo "just $lane took ${spent}ms -- OVER ITS ${allowed}ms BUDGET"
        over=1
      else
        echo "just $lane took ${spent}ms -- ok (budget ${allowed}ms)"
      fi
    done
    exit "$over"

# The home directory's database against this build: every version between
# the file's and USER_VERSION must have a migration step; a newer file
# is refused. Reads the version only -- nothing is migrated here.
[doc('Refuse a build that cannot open ~/.smartgallery (no migration step from its version)')]
db-check:
    {{ python }} -c "from sg_web import home; from db import migrate; p = home.db_path(home.home(None)); print(p, 'pending', migrate.pending(p) if p.exists() else 'no database yet')"

# Every surface and five real pictures over the database in the home
# directory -- the check a lane over fresh databases cannot make.
# `just smoke --home D:/runs/two` for another run.
[doc('Walk the real run: every surface and real pictures, over ~/.smartgallery')]
smoke *ARGS: web::build
    {{ python }} -m sg_web.smoke "$@"

# The README's own two commands, on the tree AS COMMITTED. `git
# checkout-index` writes what a CLONE would get: no .venv, no node_modules,
# no build output that is not committed.

# What this catches and nothing else does: every other lane builds the
# bundles first, so one rebuilt and never committed is invisible everywhere
# but here. Every asset a rendered page names is fetched below.

# Outside the suite deliberately: it installs, so it costs a network and
# minutes, and pytest is not where a lane like that belongs.
[doc('The README bootstrap on a cold checkout: install, serve, fetch every asset')]
[script]
acceptance-cold:
    set -eu
    tree=$(mktemp -d)
    home=$(mktemp -d)
    port=8791
    trap 'rm -rf "$tree" "$home"' EXIT
    echo "cold checkout -> $tree"
    git checkout-index -a -f --prefix="$tree/"
    test ! -e "$tree/.venv" || { echo "the export carried a .venv; it is not a cold checkout"; exit 1; }
    cd "$tree"
    uv sync
    uv run python -m sg_web --home "$home" --port "$port" &
    served=$!
    trap 'kill "$served" 2>&1 \n      || echo "warning: the served process $served was left running" >&2; rm -rf "$tree" "$home"' EXIT
    for _ in $(seq 1 60); do
      if curl -fsS "http://127.0.0.1:$port/g" -o /dev/null 2>&1; then break; fi
      sleep 1
    done
    page=$(curl -fsS "http://127.0.0.1:$port/g")
    echo "$page" | grep -q '<nav class="shell"' || { echo "the gallery did not render the shell"; exit 1; }
    missing=0
    for asset in $(echo "$page" | grep -o '\(src\|href\)="/static/[^"]*"' | sed 's/.*="//;s/"//' | sort -u); do
      code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$asset")
      echo "  $code $asset"
      [ "$code" = "200" ] || missing=1
    done
    [ "$missing" = "0" ] || { echo "an asset the page asks for is not in the committed tree"; exit 1; }
    echo "cold checkout served every asset it asked for"

# The repo-wide structural gates, on their own and in seconds: sglint's
# rules (discovered scope: a package created tomorrow is swept the day it
# is born) and the tooling checks. `just test` runs the tests too.
[doc('Structural gates alone: sglint code rules, sglint --repo hygiene, and the linter self-tests')]
audit: repo-check
    {{ python }} -m sglint
    {{ python }} -m pytest -q tests/test_sglint_has_teeth.py

# The application, served. Flags pass through -- `--port`, `--home`,
# `--public` to let other machines reach it. Everything else is a settings
# row changed in the running app.
[doc('Serve the gallery (python -m sg_web)')]
serve *ARGS: web::build
    {{ python }} -m sg_web "$@"

# Benchmarks through the production pipeline (bench.just)
mod bench

# The browser source: type check, bundle, watch (web.just)
mod web

# The Python/browser JSON contract: OpenAPI out, TypeScript in (api.just)
mod api

# Authentic historical db/schema.sql, vendored for migration fixtures (schema.just)
mod schema

# Its runtimes live in the `compat` dependency group, which `uv sync` does
# not install by default. Nothing under db/ or sg_web/ may import any.

# What must be durably stored after an expensive face observation
mod compat

# A decades-wide sample library on disk: mixed modalities, several
# generator dialects, and the holes a real folder has (corpus.just)
mod corpus

# Which faiss the app selects at runtime: the vendored GPU build on
# Windows+NVIDIA, else the installed faiss-cpu. The faiss_gpu setting row
# forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from vision.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
