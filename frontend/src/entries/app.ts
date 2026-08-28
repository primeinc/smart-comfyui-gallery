// The whole browser application, in one bundle.
//
// There used to be twelve, one per script set a template rendered. That
// arrangement made every module that says "there is one of me" a lie: each
// bundle carried its own copy, so /g loaded shell.js and gallery.js and got
// TWO keyboard registries and two keydown listeners. `c` and `?` each fired
// twice, and keys.ts could not refuse a collision it could not see, because
// the two claims were in different copies of the map.
//
// One bundle is what makes the singletons real -- one keys.ts registry, one
// workspace, one compare tray -- and it is also less to download across a
// visit: the twelve overlapped almost entirely, so a person walking from the
// grid to a person to the timeline used to fetch the same modules three
// times under three names.
//
// Every module below mounts by LOOKING for its own markup and returning when
// it is not there, so loading all of them on every page is how this works
// rather than something it survives. Anything added here must do the same.
// A module that assumes its page throws at import and takes the rest of the
// bundle down with it.

// The surfaces.
import "../collection";
import "../dupes";
import "../evolution";
import "../gallery";
import "../keywords";
import "../media";
import "../operations";
import "../people";
import "../story";
import "../timeline";

// What the surfaces are made of.
import "../authored";
import "../selection";

// Kept everywhere on purpose: the compare tray until it is dismissed, a
// thumbnail that says what it is instead of breaking, and a `<time
// data-epoch>` that is a number until something spells it.
import "../compare-mount";
import "../pictures-mount";
import "../spelling-mount";

import { mountField } from "../field";
import { mountInstall, mountServiceWorker } from "../install";
import { mountPanes } from "../panes";
import { mountShortcuts } from "../shortcuts";

// Last, after every import above has claimed its keys: the panel is built
// FROM the registry, so it can only list what has already registered.
mountInstall();
mountServiceWorker();
mountShortcuts(document);
// The canvas over the grid. It reveals itself only where the grid it
// draws is present, so every other surface is untouched.
mountField(document.body);
// The older surfaces, brought to the canvas rather than navigated to.
mountPanes(document.body);
