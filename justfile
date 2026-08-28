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
#
# `-n 0` on the command line, because pytest.ini's `-n` is global and
# every worker it starts is an interpreter importing sixty test modules
# to collect nothing. Measured on this lane: 4.7s with the workers, 2.4s
# without -- two seconds of a ten-second budget, spent to parallelise an
# empty set. (The cost scales with the count, which is how it was found:
# at eight it was 7.4s.)
test: web::build
    {{ python }} -m pytest tests/ -m "not slow" -n 0 || [ $? -eq 5 ]

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
#
# `-p pytest-testmon` is what turns the plugin back on: pytest.ini blocks
# it for every other run, because loading it costs ~99 ms of the ~950 ms
# a single-module run spends before its first test, and this is the ONE
# lane that uses it. An explicit `-p` overrides the `-p no:` in addopts
# -- verified rather than assumed: with both flags the run prints the
# same `testmon: changed files: ..., unchanged files: 3` header it
# prints with the plugin loaded normally, so selection is really
# running and not a flag being parsed and ignored.
#
# It runs in a DETACHED WORKTREE AT HEAD, not in the working tree, because
# the question a pre-push gate answers is "is the commit I am pushing
# safe" and pytest imports whatever is on disk. Testing the working tree
# answers a different question and answers it wrongly in both directions:
# uncommitted work fails a push that does not contain it (one WIP template
# attribute broke eleven tests here, and the gate reported it as the
# pushed commit), and uncommitted work can equally make a broken commit
# pass.
#
# The worktree, not a stash: a gate that moves the developer's files to
# inspect them can lose them. `git worktree add --detach` writes nothing
# into the working tree and takes no lock on it.
#
# Four things a fresh checkout cannot carry, each measured by watching
# this run fail without it:
#   sg_web/static/build   gitignored; `web::build` above made it here
#   benchmarks/results    gitignored; without it the four throughput
#                         tests fail on a green commit -- and
#                         test_a_library_with_no_benchmarks_shows_no_panel
#                         says a fresh checkout legitimately has none
#   .testmondata          the selection cache; copied in AND back out, so
#                         a run in the worktree still teaches the next one
#   .venv/Scripts on PATH tests/conftest.py boots the app with a bare
#                         `litestar`, which is FileNotFoundError WinError 2
#                         from a cwd that is not the repo root
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
    trap 'git -C "$root" worktree remove --force "$pushed" >/dev/null 2>&1 || true' EXIT
    # `<dir>/.` into an existing `<dir>/`, never `cp -r <dir> <dir>`: the
    # second nests when the destination already exists, and both of these
    # DO exist in a checkout (benchmarks/results is tracked; only the
    # newer results are untracked). Nesting is silent and reads as the
    # gate failing on a green commit.
    mkdir -p "$pushed/sg_web/static/build" "$pushed/benchmarks/results"
    cp -r sg_web/static/build/. "$pushed/sg_web/static/build/"
    if [ -d benchmarks/results ]; then cp -r benchmarks/results/. "$pushed/benchmarks/results/"; fi
    if [ -f .testmondata ]; then cp .testmondata "$pushed/.testmondata"; fi
    cd "$pushed"
    PATH="$root/.venv/Scripts:$root/.venv/bin:$PATH" PYTHONPATH=. \
      "$root/{{ python }}" -m pytest tests/ -m slow -n 4 --dist loadfile \
      -p pytest-testmon --testmon --testmon-forceselect
    settled=$?
    if [ -f "$pushed/.testmondata" ]; then cp "$pushed/.testmondata" "$root/.testmondata"; fi
    exit $settled

# The PWA's rasters, drawn from the mark: icons, the iOS splash set,
# and the install-sheet screenshots photographed off the real app over
# a generated library (sg_web/branding.py). Run after changing the
# palette, the mark, or the gallery's look; commit what it writes.
[doc('Regenerate the PWA icons, iOS splash set, and install screenshots')]
pwa-assets:
    {{ python }} -m sg_web.branding all

# Every test alone, one process each: a test that leans on a sibling's
# writes passes in module order and fails here -- which is exactly how
# the affected-test selector (prove-push) will eventually run it. Slow
# by construction (an import and a world per test); an audit, not a
# gate. Failures are listed at the end, and the exit code says so.
[doc('Run every test in its own process; order-dependent tests fail here first')]
[script]
audit-isolation:
    cd "$(git rev-parse --show-toplevel)"
    # The audit serves the LIVE working tree to real browsers for an
    # hour. An edit or a rebuild underneath it (build-web clears
    # static/build before rewriting) makes every later measurement a
    # picture of a torn tree -- ten phantom failures, all one 404. So
    # the tree is fingerprinted first and every verdict is refused if
    # it moved: an invalid audit must say so, not report ghosts.
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
# TWO Python checkers, and both fit the ten-second gate. Measured on this
# tree with hyperfine, 3 runs after warmup, the venv's own binaries: ty
# 2.865s +/- 0.049, pyrefly 1.772s +/- 0.058, tsc over the browser source
# ~2s. That is what keeps them here rather than in `check-deep`, which no
# hook runs: a Python type check nobody waits for is one nobody reads.
#
# Both, not one, because they do not agree and neither is a superset. On
# the tree as it stands each finds things the other does not: pyrefly
# reports `unnecessary-type-conversion` and `non-convergent-recursion`,
# which ty has no rule for; ty reports `redundant-cast` where pyrefly is
# silent, and its `possibly-unresolved-reference` is the possibly-unbound
# check pyproject.toml turns on by name. At four and a half seconds for
# the pair, the question is not which one.
#
# Both halves always run where they run. As a dependency with a body, the
# body was skipped whenever the dependency failed, so a red browser source
# meant no Python was type checked at all -- which during a migration is
# every single run.
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
gates: api::check lint fmt-check web::types types-python web::unit db-check

# What could not be made to fit ten seconds. Not less important --
# repo-check is what keeps a clone honest, and `types-elsewhere` is the
# only thing that reads the platforms this machine is not. Measured:
# repo-check 9.3s (two full `git checkout-index -a` into temporary trees
# plus a scratch repository, on a platform where each git spawn costs
# about 200ms).
#
# Python type checking is NOT here any more; at four and a half seconds
# for both checkers it moved into `check`, and so into the pre-commit
# hook, which is where it can catch anything.
[doc('What cannot fit ten seconds: repo hygiene, and the platforms this machine is not')]
[parallel]
check-deep: repo-check types-elsewhere

# The same two checkers, at the platforms nobody here is running.
#
# `check` runs them at each tool's default, which is the developer's own
# machine, and platform-conditional code is then read for that platform
# alone. This project ships a Linux container and declares a darwin branch
# of its dependencies, so two thirds of what it targets went unread.
#
# `python-platform = "all"` is NOT this, measured: "all" takes the UNION
# of the platform stubs, so a win32-only name RESOLVES under it and the
# diagnostic disappears. A pass per platform is the only form that checks
# a platform.
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

# Its runtimes live in the `compat` dependency group, which `uv sync` does
# not install by default: `uv sync --group compat`. Nothing under db/ or
# sg_web/ may import any of them.
#
# What must be durably stored after an expensive face observation
mod compat

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
