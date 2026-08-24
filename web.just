# The authored browser source: TypeScript under frontend/src, bundled by
# esbuild into sg_web/static/build -- one output per <script> a template
# renders.
#
# The two halves are independent by design. esbuild strips types without
# reading them ("esbuild does not do any type checking so you will still need
# to run `tsc -noEmit` in parallel", esbuild.github.io/content-types/
# #typescript), so bundling green proves nothing about types and vice versa.
# Neither implies the other, so `check` runs both and needs both.
#
# Measured here: bundle 61ms, type check ~1s. Cheap enough that everything
# which serves or tests the app depends on the bundle rather than hoping it
# is current.

set script-interpreter := ['bash', '-euo', 'pipefail']

# The shortest check that is still valid for the browser source: it type
# checks, it bundles, and it runs the unit tests that need no browser.
[default]
[parallel]
check: types build unit

# The browser modules whose invariant a browser cannot witness. The keyboard
# registry is the one so far: what its map holds after a registration was
# REFUSED is invisible from a page, because a refused registration is a page
# that did not finish loading. node runs the TypeScript directly (no test
# framework, no transpile step -- package.json pins the node floor for the
# same reason build.ts does), so this costs about a tenth of a second.
unit:
    npm run --silent test-web

# Type check only -- esbuild never will. Three projects, one per host: the
# browser source has the DOM and no node, build.ts has node and no DOM, and
# a *.test.ts is run BY node over code written for a browser, so it needs
# both. Chaining them with `&&` would stop checking the others the moment
# one had an error, which during a migration is always.
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

# Bundle every surface. The output directory is removed first because esbuild
# does not clear it ("it will not be cleared if it already contains some
# files", esbuild.github.io/api/#outdir): a renamed or deleted surface would
# otherwise leave a stale bundle behind that a template still loads.
build:
    rm -rf sg_web/static/build
    npm run --silent build-web

# Rebuild on change, for a second terminal. Deliberately not type checking:
# keep `just web types` where you can see it.
watch:
    npm run --silent watch-web
