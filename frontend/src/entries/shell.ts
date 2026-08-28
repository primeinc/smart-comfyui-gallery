// base.html: the shell every page renders. Install affordances and the
// service worker's registration belong to the shell, not to any one page.
import { mountInstall, mountServiceWorker } from "../install";
import { mountShortcuts } from "../shortcuts";

mountInstall();
mountServiceWorker();
// What the keyboard does. Here rather than in each entry: the control
// lives in the shell, so every surface that renders the shell has it.
mountShortcuts(document);
