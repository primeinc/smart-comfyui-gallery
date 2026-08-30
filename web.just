# The authored browser source: TypeScript under frontend/src, bundled by
# esbuild into sg_web/static/build, one output per <script> a template
# renders.

# esbuild strips types without reading them (esbuild.github.io/content-types
# #typescript), so bundling green proves nothing about types and neither
# implies the other. `check` runs both and needs both.

set script-interpreter := ['bash', '-euo', 'pipefail']

# The shortest check that is still valid for the browser source: it type
# checks, it bundles, and it runs the unit tests that need no browser.
[default]
[parallel]
check: types build unit

# The browser modules whose invariant a browser cannot witness: what the
# keyboard registry holds after a REFUSED registration is invisible from a
# page, because a refused registration is a page that did not finish loading.

# node runs the TypeScript directly, no test framework and no transpile step.
unit:
    npm run --silent test-web

# Type check only; esbuild never will. Three projects, one per host: browser
# source has the DOM and no node, build.ts has node and no DOM, a *.test.ts
# is run BY node over browser code and needs both.

# Parallel rather than chained: `&&` stops checking the others at the first
# error, which during a migration is always.
[parallel]
types: types-source types-build types-test

[private]
types-source:
    npm run --silent types-source

[private]
types-build:
    npm run --silent types-build

[private]
types-test:
    npm run --silent types-test

# The committed bundles against the source they were made from. They ship
# with the repository so running the app needs no Node, which is only safe if
# a stale one cannot survive this gate.
[script]
fresh: build
    # Two questions. `git diff` reads the INDEX, so it sees a rebuilt bundle
    # nobody staged; the index says nothing about a file git was never told
    # exists, which is the one `ls-files --others` asks.
    stale="$(git diff --name-only -- sg_web/static/build)"
    never="$(git ls-files --others --exclude-standard -- sg_web/static/build)"
    if [ -n "$stale" ] || [ -n "$never" ]; then
      echo 'the committed bundles do not match the built ones: run `just web build` and commit the result' >&2
      [ -n "$stale" ] && printf 'rebuilt, not staged:\n%s\n' "$stale" >&2
      [ -n "$never" ] && printf 'never added to git:\n%s\n' "$never" >&2
      exit 1
    fi

# Bundle every surface. The clean lives inside `frontend/build.ts`, so this
# recipe and the bare `npm run build-web` the README documents are one
# contract, safe against a renamed surface leaving its old bundle behind.
build:
    npm run --silent build-web

# Rebuild on change, for a second terminal. Deliberately not type checking:
# keep `just web types` where you can see it.
watch:
    npm run --silent watch-web
