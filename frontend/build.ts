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
import { rm } from "node:fs/promises";
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
    dupes: "src/entries/dupes.ts",
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

// The output directory is cleared HERE, not by whatever ran this.
// esbuild has no option for it -- `BuildOptions` carries `outdir` and
// nothing that empties it (lib/shared/types.ts) -- and it does not clear
// a directory that already has files in it, so a renamed or deleted
// surface leaves a stale bundle behind that a template goes on loading.
//
// It used to live in `just web build` alone, which meant the command the
// README hands people (`npm run build-web`) and the command the gate runs
// were two different contracts, and only one of them was safe. One
// command owns the clean now, and every caller inherits it.
//
// Once, before watching too: rebuilds inside a watch are incremental and
// clearing between them would delete output nothing is about to rewrite.
await rm(options.outdir, { recursive: true, force: true });

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
