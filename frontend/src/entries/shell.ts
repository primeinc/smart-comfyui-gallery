// base.html: the shell every page renders. Install affordances and the
// service worker's registration belong to the shell, not to any one page.
import { mountInstall, mountServiceWorker } from "../install";

mountInstall();
mountServiceWorker();
