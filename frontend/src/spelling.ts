// A `<time data-epoch>` is a number until something spells it.
//
// The application renders epochs on nine surfaces and only three of them
// carried a speller -- media.ts, operations.ts and people.ts, each with
// its own copy. Albums, folders, places and stories loaded entries that
// had none, so those pages showed the reader `1787276354.1022315` where
// a date goes. It is invisible on an empty library, which is the only
// state those pages had ever been looked at in.
//
// One speller, exported once, mounted by every entry that renders one.
import { everyElement, requireData } from "./dom";

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * Spell every epoch not yet spelled.
 *
 * Only nodes without `data-spelled`: a caller watching the document for
 * changes fires on this very write, and spelling an already-spelled node
 * again would loop forever.
 *
 * UTC, deliberately: the epoch on these cards is a day the library
 * computed, and re-reading it in the reader's zone would move a
 * photograph across midnight depending on where it is being looked at.
 *
 * `data-domain` says what KIND of moment it is, and the three answers
 * want different sentences:
 *
 *   day       a span the library computed -- a date, and no clock
 *   wall      the time on the clock where the photograph was taken
 *   instant   a moment on the world's clock, so it is marked Z
 *
 * media.ts used to carry a second copy of this that knew about the
 * domain, and this one did not. Once every module shared one bundle both
 * ran on the media page: media's wrote the moment, this one saw a node
 * with no `data-spelled` and overwrote it with the date, and `wall` and
 * `instant` came out identical on the one surface built to tell them
 * apart. There is one speller, and it is this one.
 */
export function spellDays(root: ParentNode): void {
  for (const node of everyElement(root, "time[data-epoch]:not([data-spelled])", HTMLTimeElement)) {
    const d = new Date(Number(requireData(node, "epoch")) * 1000);
    const day = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
    const domain = node.dataset.domain;
    if (domain === "instant" || domain === "wall") {
      const clock = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
      node.textContent = domain === "instant" ? `${day} ${clock}Z` : `${day} ${clock} wall`;
    } else {
      node.textContent = day;
    }
    node.dataset.spelled = "";
  }
}
