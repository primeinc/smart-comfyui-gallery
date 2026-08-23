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
# checks AND it bundles.
[default]
[parallel]
check: types build

# Type check only -- esbuild never will. Two projects, because the browser
# source has the DOM and no node, and build.ts has node and no DOM. Chaining
# them with `&&` would stop checking the build script the moment the browser
# source had one error, which during a migration is always.
[parallel]
types: types-source types-build

[private]
types-source:
    npm run --silent types-source

[private]
types-build:
    npm run --silent types-build

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
