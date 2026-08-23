// Bundles the authored browser source into sg_web/static/build, one output
// per script set a template renders.
//
// Run by node directly -- node strips the types (v23.6+) and the tsconfig's
// erasableSyntaxOnly keeps this file strippable, so there is no build step
// for the build. esbuild never type checks; `tsc` is the other half of
// `just web`.
//
// The options object is checked against esbuild's own BuildOptions, whose
// SameShape helper resolves an unknown key to `never`: a typo here is a type
// error at `just web types`, not a silently ignored flag.
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));

// One entry per <script> a template renders. Two surfaces that load different
// sets get different entries even where they share most modules: the
// duplication is a few kB, and code splitting is ESM-only with a known
// import-ordering bug (esbuild.github.io/api/#splitting).
const options = {
  absWorkingDir: here,
  entryPoints: {
    gallery: "src/entries/gallery.ts",
    people: "src/entries/people.ts",
    person: "src/entries/person.ts",
    media: "src/entries/media.ts",
    collection: "src/entries/collection.ts",
    timeline: "src/entries/timeline.ts",
    evolution: "src/entries/evolution.ts",
    operations: "src/entries/operations.ts",
    story: "src/entries/story.ts",
  },
  outdir: resolve(here, "../sg_web/static/build"),
  bundle: true,
  platform: "browser",
  // Already the default for a bundled browser build; said out loud because it
  // is what keeps the bundle's own names out of the global scope.
  format: "iife",
  target: "es2022",
  sourcemap: true,
  minify: false,
  logLevel: "info",
} satisfies esbuild.BuildOptions;

if (process.argv.includes("--watch")) {
  const context = await esbuild.context(options);
  // watch() resolves as soon as watching begins and does NOT wait for a first
  // build ("the JavaScript and Go watch APIs complete as soon as watch mode
  // has started", esbuild.github.io/api/#watch). Without this rebuild the
  // server would serve the previous run's output until the first edit.
  await context.rebuild();
  await context.watch();
} else {
  await esbuild.build(options);
}
