# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The fast lane: every test that does not drive a browser, boot a server
# or open a database. tests/conftest.py SLOW_FIXTURES is the criterion and
# derives it from each test's own fixture closure.
#
# The exit code is NOT swallowed. pytest exits 5 when it collected nothing,
# and that is the only signal that the selection has stopped matching any
# test, so it must fail the recipe rather than be converted to success.
#
# Workers come from pytest.ini.
test: web::build
    {{ python }} -m pytest tests/ -m "not slow"

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
#
# And one thing it cannot carry at all: a `.venv` DIRECTORY beside
# pyproject.toml. `sg_web/__main__.py interpreter()` is a lookup at that
# exact path and nothing else -- deliberately, so the handover cannot
# pick up a stranger's python -- and the two tests below assert that the
# path it finds resolves to the interpreter running them. A worktree has
# no .venv, so both fail here and only here, on every commit, for ever.
#
# They are DESELECTED rather than satisfied. Satisfying them means a
# junction or symlink from the worktree to the real environment, and that
# was tried: `git worktree remove --force` walks the tree, follows the
# link, and deletes the environment it points at. Measured, the hard way
# -- .venv lost its pyvenv.cfg and its site-packages, and the repository
# needed `uv sync` to stand up again. A pre-push gate must not be able to
# do that, however carefully the cleanup is ordered, because the ordering
# only holds when the cleanup runs at all.
#
# So they are deselected from the worktree pass and RUN IN THE CHECKOUT
# at the end of this recipe. The gate still runs them -- a test the gate
# never runs is not gating anything -- and it runs them in the only place
# they can answer: both ask whether this machine has an environment
# beside the source and whether the handover would find it, which is a
# question about the installation and not about the commit.
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
    launch=tests/test_the_documented_launch_serves_a_whole_application.py
    # The corpus, said out loud. `tests/test_the_corpus_spans_the_shape.py`
    # looks for `sg-corpus` BESIDE the repository, resolved from the test
    # file -- so in a worktree under /tmp it looks beside /tmp, finds
    # nothing, and six coverage tests report that the corpus produced no
    # dialects, no media kinds and no dating rungs. Not a skip: a wall of
    # red saying the corpus is empty, about a corpus sitting where it
    # always was. `SG_CORPUS` is the seam that exists for this, so the
    # gate measures the same corpus a developer does.
    if [ -d "$root/../sg-corpus" ]; then export SG_CORPUS="$root/../sg-corpus"; fi
    # And the RUN that scanned it. Three of those six tests do not ask the
    # corpus what it holds, they ask a library that read it -- which kinds
    # were served, which dating rung each file landed on, what precision
    # came out. `tests/needs.py _served_db` finds that library at `sg-run`
    # beside the repository, resolved the same way and wrong in a worktree
    # for the same reason. Without it every rung reports
    # UNKNOWN_NOT_MEASURED, which these tests correctly refuse to read as
    # "reached" -- so the gate said the corpus covers no media kind at all.
    if [ -d "$root/../sg-run" ]; then export SG_HOME="$root/../sg-run"; fi
    PATH="$root/.venv/Scripts:$root/.venv/bin:$PATH" PYTHONPATH=. \
      "$root/{{ python }}" -m pytest tests/ -m slow -n 4 --dist loadfile \
      --deselect "$launch::test_an_interpreter_without_a_server_is_handed_to_the_one_that_has_it" \
      --deselect "$launch::test_the_environment_this_suite_runs_in_is_the_one_the_handover_targets" \
      -p pytest-testmon --testmon --testmon-forceselect
    settled=$?
    if [ -f "$pushed/.testmondata" ]; then cp "$pushed/.testmondata" "$root/.testmondata"; fi
    # The two deselected above, run HERE, in the checkout. Deselecting
    # them from the worktree pass is not the same as not running them: a
    # test the gate never runs is not gating anything.
    #
    # The checkout is also the only place they mean anything. Both ask
    # whether THIS machine has an environment beside the source and
    # whether the handover would find it -- a question about the
    # installation, which no commit can change and no temporary copy of
    # the source can answer.
    cd "$root"
    "$root/{{ python }}" -m pytest       "$launch::test_an_interpreter_without_a_server_is_handed_to_the_one_that_has_it"       "$launch::test_the_environment_this_suite_runs_in_is_the_one_the_handover_targets"       -q --no-header
    handover=$?
    if [ "$settled" -ne 0 ]; then exit $settled; fi
    exit $handover

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
gates: api::check lint fmt-check web::types types-python web::unit db-check prose

# Comment and docstring prose, against CONTRIBUTING.md. Vale reads Python
# through tree-sitter, so it sees comments and docstrings and not string
# literals or code.
#
# `--no-global` is load-bearing. Vale always loads a per-user config last and
# lets it override, so without this a developer's own file changes what this
# gate reports.
#
# Exit 1 is a finding and 2 is Vale itself failing; they are reported apart so
# a broken config cannot read as a clean tree. A missing binary fails here
# rather than passing silently.
[private]
[script]
prose:
    if ! command -v vale >/dev/null 2>&1; then
      echo "vale is not installed; the prose gate cannot run (scoop install vale)" >&2
      exit 1
    fi
    # Banded, on the same grounds as [tool.ruff].required-version in
    # pyproject.toml: a linter that gains rules on an upgrade rewrites the
    # tree. The floor is where the features this gate needs landed --
    # code-aware Python linting and per-style severity are both 3.17.0 -- and
    # the ceiling keeps a major release from arriving unread.
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

# The budget, enforced rather than promised. Runs each lane and fails on
# the clock, so the day something slow is added is the day this says so,
# by name, with the number. This recipe is what re-derives the timings;
# they are not copied into prose, where they would go stale unread.
#
# The budgets are PER LANE because the lanes are not the same size, and
# `test` is deliberately not held to `check`'s ten seconds. The fast lane
# is several hundred tests with a flat duration distribution -- no tail to
# cut -- so a lane that fit ten seconds could only be built by dropping
# half of them on the criterion "whatever fits", which is a lane whose
# green says nothing. `--durations` on the lane is what shows the shape.
[doc('Prove `just check` and `just test` each stay inside their measured budget')]
[script]
budget:
    over=0
    for lane in check test; do
      case "$lane" in
        check) allowed=10000 ;;
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
