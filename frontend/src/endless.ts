/**
 * Browsing: the next page arrives because you kept going.
 *
 * The gallery had exactly two ways forward -- `next` and `previous` --
 * which is a fine way to READ a result set and a poor way to browse one.
 * Looking for a picture you would recognise means going through a few
 * hundred, and clicking `next` every sixty is a decision you have to make
 * five times a minute about something you are not thinking about.
 *
 * What this is NOT: a second paging engine. The server's ResultSet still
 * decides membership, order and what page 7 contains; this asks for page
 * 7 through the same `/g/grid` fragment the pager's own link asks for,
 * and appends what comes back. There is no client-side ordering here and
 * no page arithmetic beyond "the one after the last one I have".
 *
 * Three things it has to get right.
 *
 * THE DOM IS BOUNDED. Appending for ever is how a browser tab dies on a
 * real library: 80,000 cells is 80,000 image elements. Past a window of
 * pages the oldest are dropped and the space they took is held open as
 * padding, so the scroll position does not jump and coming back up
 * restores them.
 *
 * THE URL FOLLOWS. `page` is replaced -- never pushed -- with whichever
 * page fills the top of the viewport, so reload lands where you were
 * reading and Back still means the query before this one rather than
 * one press per sixty pictures.
 *
 * THE PAGER STAYS. It is the sentinel that triggers the next fetch, it
 * is the way to jump, and it is what a browser with no JavaScript uses.
 * Nothing here removes it.
 */

import { findElement, requireData } from "./dom";

/**
 * How many pages of cells stay in the document.
 *
 * Six pages is 360 cells, which is several screens in every direction at
 * any sane cell size -- enough that the drop is never visible, small
 * enough that it is a bound.
 */
const WINDOW = 6;

/** How far below the fold the end has to be before nothing more is asked for. */
const REACH = 600;

/** What one page of cells left behind when it was dropped. */
interface Dropped {
  height: number;
}

export function mountEndless(root: HTMLElement): void {
  const grid = findElement(root, "[data-grid]", HTMLElement);
  if (!grid) return;
  const cells = findElement(grid, "[data-cells]", HTMLElement);
  const pager = findElement(grid, "[data-pager]", HTMLElement);
  if (!cells || !pager) return;

  const pages = Number(requireData(grid, "pages"));
  const first = Number(requireData(grid, "page"));
  const qbase = grid.dataset.qbase ?? "";
  if (!Number.isFinite(pages) || !Number.isFinite(first)) return;

  // The cells the server rendered belong to the page it rendered, and
  // every appended cell says which page it came from -- that is what
  // makes dropping a page and restoring it a matter of reading the DOM
  // rather than of counting.
  for (const cell of cells.children) {
    if (cell instanceof HTMLElement) cell.dataset.page = String(first);
  }

  let lowest = first;
  let highest = first;
  let busy = false;
  const dropped = new Map<number, Dropped>();

  const cellsOf = (page: number): HTMLElement[] =>
    [...cells.children].filter(
      (one): one is HTMLElement => one instanceof HTMLElement && one.dataset.page === String(page),
    );

  /** The height a run of cells occupies, including the gap under it. */
  const spanOf = (held: HTMLElement[]): number => {
    if (!held.length) return 0;
    const top = Math.min(...held.map((one) => one.getBoundingClientRect().top));
    const bottom = Math.max(...held.map((one) => one.getBoundingClientRect().bottom));
    return bottom - top;
  };

  const padding = (): number => Number.parseFloat(cells.style.paddingTop || "0") || 0;

  /**
   * Drop the oldest page and hold its space open.
   *
   * Measured BEFORE removal and applied as padding, so everything below
   * stays exactly where it is on screen: a drop that reflowed the page
   * under the reader's eyes would be worse than the unbounded DOM it is
   * there to prevent.
   */
  const dropOldest = () => {
    const held = cellsOf(lowest);
    if (!held.length) return;
    const height = spanOf(held);
    for (const one of held) one.remove();
    dropped.set(lowest, { height });
    cells.style.paddingTop = `${padding() + height}px`;
    lowest += 1;
  };

  const fetchPage = async (page: number): Promise<HTMLElement[] | null> => {
    const answered = await fetch(`/g/grid?${qbase}page=${page}`, { headers: { accept: "text/html" } });
    if (!answered.ok) return null;
    const parsed = new DOMParser().parseFromString(await answered.text(), "text/html");
    const fresh = parsed.querySelector("[data-cells]");
    if (!fresh) return null;
    const made: HTMLElement[] = [];
    for (const one of [...fresh.children]) {
      if (!(one instanceof HTMLElement)) continue;
      one.dataset.page = String(page);
      made.push(one);
    }
    return made;
  };

  const extend = async () => {
    if (highest >= pages) return;
    grid.dataset.loading = "true";
    try {
      const made = await fetchPage(highest + 1);
      if (made) {
        cells.append(...made);
        highest += 1;
        while (highest - lowest + 1 > WINDOW) dropOldest();
      }
    } catch {
      // The next scroll tries again. A failed append is a page that did
      // not arrive, not a gallery that has ended -- and saying "no more
      // results" because of a dropped connection is the lie this catch
      // exists to avoid.
    } finally {
      delete grid.dataset.loading;
    }
  };

  /**
   * Keep extending while the end is still within reach.
   *
   * An IntersectionObserver fires when intersection CHANGES, not while it
   * persists. A sentinel that is already on screen when its page arrives
   * -- which is every time, because a page of sixty cells rarely fills
   * the trigger margin on a wide window -- never changes, so nothing
   * fires again and the gallery stops dead after one append. The observer
   * is the wake-up; this loop is what actually fills the screen.
   */
  const pump = async () => {
    if (busy) return;
    busy = true;
    try {
      while (highest < pages) {
        if (pager.getBoundingClientRect().top > window.innerHeight + REACH) break;
        const before = highest;
        await extend();
        // a page that did not arrive stops the loop rather than spinning
        if (highest === before) break;
      }
    } finally {
      busy = false;
    }
  };

  /** Bring back a page dropped off the top, and give back its padding. */
  const restore = async () => {
    if (busy || lowest <= 1 || !dropped.has(lowest - 1)) return;
    busy = true;
    try {
      const page = lowest - 1;
      const made = await fetchPage(page);
      if (made) {
        cells.prepend(...made);
        const held = dropped.get(page);
        dropped.delete(page);
        cells.style.paddingTop = `${Math.max(0, padding() - (held?.height ?? 0))}px`;
        lowest = page;
        while (highest - lowest + 1 > WINDOW) {
          // dropping from the BOTTOM now: the reader is going up
          const last = cellsOf(highest);
          if (!last.length) break;
          for (const one of last) one.remove();
          highest -= 1;
        }
      }
    } catch {
      // as above: the next scroll upward tries again
    } finally {
      busy = false;
    }
  };

  // The pager IS the sentinel. It already sits after the last cell, it is
  // already the way to jump, and it is what a browser with no JavaScript
  // uses -- so nothing new is added to the document and nothing is taken
  // away from anyone.
  const watchDown = new IntersectionObserver(
    (entries) => {
      if (entries.some((one) => one.isIntersecting)) void pump();
    },
    { rootMargin: `${REACH}px 0px` },
  );
  watchDown.observe(pager);

  // --- the URL follows ------------------------------------------------------
  //
  // REPLACE, never push. Pushing would make Back mean "up one screen",
  // and then leaving a gallery somebody scrolled through would be twenty
  // presses. Back stays what it was: the query before this one.
  let shown = first;
  let waiting = 0;
  const follow = () => {
    waiting = 0;
    // Going back up is decided by where the window IS, not by an observer
    // on the cells container -- that container is always intersecting,
    // which is a trigger that fires once and then never again.
    if (window.scrollY < REACH) void restore();
    const top = [...cells.children].find(
      (one) => one instanceof HTMLElement && one.getBoundingClientRect().bottom > 0,
    ) as HTMLElement | undefined;
    const page = Number(top?.dataset.page);
    if (!Number.isFinite(page) || page === shown) return;
    shown = page;
    grid.dataset.page = String(page);
    const held = new URLSearchParams(window.location.search);
    if (page > 1) held.set("page", String(page));
    else held.delete("page");
    const spelled = held.toString();
    window.history.replaceState(
      window.history.state,
      "",
      spelled ? `${window.location.pathname}?${spelled}` : window.location.pathname,
    );
  };
  window.addEventListener(
    "scroll",
    () => {
      if (!waiting) waiting = window.requestAnimationFrame(follow);
    },
    { passive: true },
  );
}
