# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The ten-second lane, and it is EMPTY.
#
# `just test` and `just check` are each held to ten seconds. Measured,
# the suite is 51s and nothing in it is cheap: the least expensive test
# opens a database, and the rest serve the application or drive a
# browser (tests/conftest.py marks the whole collection slow). So this
# lane collects nothing rather than collecting the few that happen to
# fit and reporting green about a suite it did not run.
#
# pytest exits 5 when it collects nothing, which is the expected
# outcome here and not a failure. The suite is `just test-slow`.
# No xdist: sixteen workers importing sixty test modules to collect
# nothing cost 13.4s of the lane's 18.8s. One process collects the same
# nothing in a third of that.
test: web::build
    {{ python }} -m pytest tests/ -m "not slow" || [ $? -eq 5 ]

# The suite. All of it: real sample libraries, real browsers, real
# migration chains. Minutes, not seconds -- which is why it is not in
# `just test` and not in `just check`.
test-slow: web::build
    {{ python }} -m pytest tests/ -m slow -n 4 --dist loadfile

# The recorded full proof. Both gates and the whole suite, run when
# waiting is cheap (backgroundable); on green, the COMMITTED tree's hash
# lands in .proven-tree, so pushing an already-proven tree costs nothing
# instead of re-deriving six minutes of certainty. A dirty working tree
# still runs everything but records nothing -- the marker claims "this
# committed tree passed", and a dirty run proved something else.
[doc('Both gates + the whole suite; on green, record the proven tree in .proven-tree')]
[script]
prove:
    cd "$(git rev-parse --show-toplevel)"
    # The tree is read BEFORE the run and compared after: a commit
    # landing mid-run must not be stamped with a proof that ran against
    # its parent. Dependencies became inline calls for the same reason
    # -- the capture has to happen first.
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
# milliseconds; anything else runs the AFFECTED slice of the suite --
# pytest-testmon selects by measured per-test file coverage
# (.testmondata), and --testmon-forceselect keeps selection active
# beside `-m slow`, which would otherwise disable it
# (tarpas/pytest-testmon testmon/configure.py:65-76). A missing
# .testmondata makes the first run a full one that seeds the database.
# Selection is a slice, not the proof: this never writes .proven-tree;
# `just prove` is the only writer.
[doc('Pre-push gate: skip a proven tree, else run the affected slice of the suite')]
[script]
prove-push: web::build
    cd "$(git rev-parse --show-toplevel)"
    tree=$(git rev-parse 'HEAD^{tree}')
    if [ -f .proven-tree ] && [ "$(cat .proven-tree)" = "$tree" ]; then
        echo "tree already proven by 'just prove'; nothing to re-derive"
        exit 0
    fi
    {{ python }} -m pytest tests/ -m slow -n 4 --dist loadfile --testmon --testmon-forceselect

# Every test alone, one process each: a test that leans on a sibling's
# writes passes in module order and fails here -- which is exactly how
# the affected-test selector (prove-push) will eventually run it. Slow
# by construction (an import and a world per test); an audit, not a
# gate. Failures are listed at the end, and the exit code says so.
[doc('Run every test in its own process; order-dependent tests fail here first')]
[script]
audit-isolation:
    cd "$(git rev-parse --show-toplevel)"
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
    echo "order-dependence audit: $total tests, $broke fail alone"
    if [ "$broke" -gt 0 ]; then
        echo "full output of each failure: $kept"
        exit 1
    fi

# Ruff over the Python, Biome over the browser source (biome.json: the
# first-party JS and CSS, never the vendored htmx), and this repository's
# own structural rules (sglint).
#
# Three separate programs reading three separate things, so they run
# together: serially they were 0.2 + 1.5 + 4.1 = 5.8s of the gate's ten,
# and the gate went over its budget the day the vocabulary work landed.
# In parallel the lane costs whatever sglint costs.
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

# The cross-module inference neither ruff nor esbuild can do.
#
# SPLIT BY COST, not by importance. tsc over the browser source is ~2s and
# stays in the ten-second gate; pyright over the Python is 137s and cannot
# be in it.
#
# Measured, whole tree, 170 files: pyright 137.5s. `--threads` makes it
# WORSE (181s). The cost is one import: vision/semantic/openclip.py,
# vision/semantic/qwen_vl.py and vision/captions.py each take ~90s ALONE,
# and what they share is `torch`. torch and transformers both ship
# py.typed, so pyright reads their inline annotations from source and
# `useLibraryCodeForTypes = false` does not skip them. There is no setting
# that keeps torch's types and avoids parsing torch.
#
# Both halves always run where they run. As a dependency with a body, the
# body was skipped whenever the dependency failed, so a red browser source
# meant no Python was type checked at all -- which during a migration is
# every single run.
[parallel]
types: web::types types-python

[private]
types-python:
    {{ python }} -m pyright

# Repository hygiene: the git index, line endings, the requirements
# file, the test command, the evidence stamp -- sglint's SG8xx, which ask
# git; no test ever does
repo-check:
    {{ python }} -m sglint --repo

# THE GATE, AND IT IS HELD TO TEN SECONDS.
#
# Ten seconds is the budget, `just budget` is what proves it, and what
# does not fit is not quietly kept -- it moves to `check-deep` and says so.
# A gate nobody waits for is a gate people commit around.
#
# `web::fresh` runs FIRST and alone: it deletes and rebuilds
# sg_web/static/build, which biome would otherwise be walking at the same
# moment. Everything after it is independent and runs together.
[doc('The gate. Held to ten seconds; `just budget` proves it')]
check: web::fresh gates

[parallel]
[private]
gates: api::check lint fmt-check web::types web::unit db-check

# What could not be made to fit ten seconds. Not less important -- pyright
# is the only cross-module inference this project has over its Python, and
# repo-check is what keeps a clone honest. Measured: pyright 137s (torch;
# see `types` above), repo-check 9.3s (two full `git checkout-index -a`
# into temporary trees plus a scratch repository, on a platform where each
# git spawn costs about 200ms).
[doc('What cannot fit ten seconds: pyright over the Python, repo hygiene')]
[parallel]
check-deep: types-python repo-check

# Everything: the gate, the deep gate, the suite, and the real run walked.
[doc('Everything: both gates, the suite, and the real run walked')]
check-all: check check-deep test-slow smoke

# The budget, enforced rather than promised.
#
# `just check` and `just test` are each allowed ten seconds. This runs
# them and fails on the clock, so the day something slow is added to
# either lane is the day this says so, by name, with the number.
[doc('Prove `just check` and `just test` each stay inside ten seconds')]
[script]
budget:
    over=0
    for lane in check test; do
      start=$(date +%s%N)
      just "$lane"
      spent=$(( ($(date +%s%N) - start) / 1000000 ))
      if [ "$spent" -gt 10000 ]; then
        echo "just $lane took ${spent}ms -- OVER THE TEN SECONDS"
        over=1
      else
        echo "just $lane took ${spent}ms -- ok"
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

# The README's own two commands, on the tree AS COMMITTED.
#
# `git checkout-index` writes what a CLONE would get: no .venv, no
# node_modules, no build output that is not committed. Then the two
# lines under "Run" in README.md are the only thing that happens, and a
# real process is asked for a real page over a real socket.
#
# What this catches and nothing else does: the README says "No Node, no
# npm -- the browser bundles are committed", and every other lane in
# this repo builds them first (`web::build` is a dependency of test,
# check and smoke). So a bundle that was rebuilt and never committed is
# invisible everywhere except here, and the symptom in a clone is a page
# that renders with scripts that 404 -- the pictures arrive and nothing
# about them works. Every asset a rendered page names is fetched below
# for exactly that reason.
#
# Outside the suite deliberately: it installs, so it costs a network and
# minutes, and pytest is not where a lane like that belongs.
# tests/test_the_documented_launch_serves_a_whole_application.py says so
# in its own docstring, and this is the lane it points at.
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
    trap 'kill "$served" 2>&1 || true; rm -rf "$tree" "$home"' EXIT
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

# The application, served. Flags pass through: `just serve --port 9000`,
# `just serve --home D:/runs/two`, `just serve --public` to let other
# machines on the network reach it -- everything else is a settings row
# changed in the running app.
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

# A decades-wide sample library on disk: mixed modalities, several
# generator dialects, and the holes a real folder has (corpus.just)
mod corpus

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. The faiss_gpu setting
# row forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from vision.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
