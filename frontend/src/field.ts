/**
 * The field: the answer as a place you move through, drawn on a canvas.
 *
 * A grid of `<img>` can show sixty pictures. It cannot show four
 * thousand, and it cannot put a picture anywhere except the next slot in
 * a row -- so the only thing the layout ever says is "these came back in
 * this order". Everything the library knows about WHEN a picture
 * happened is thrown away by the container it is drawn in.
 *
 * This draws the same answer to one `<canvas>` and gives every picture a
 * position that means something:
 *
 *   TIME     x is the moment it was taken; pictures that will not fit
 *            beside each other stack. A busy afternoon is a tower, a
 *            quiet month is a gap you can see across.
 *   RANKED   the answer's own order, justified into rows -- what the
 *            grid draws, so switching between them is the same pictures
 *            moving rather than a different page.
 *
 * Moving between the two is an animation of the same nodes, because they
 * ARE the same nodes: nothing is added, removed or re-fetched, and the
 * picture you were looking at is the picture you are still looking at.
 *
 * ZOOM IS THE NAVIGATION. There is no "open" step. Push into a picture
 * until it fills the frame and you are on that picture -- its details
 * arrive beside it, the full-size file replaces the thumbnail, and the
 * arrows walk the answer. Pull back out and you are in the field again,
 * where you left it. One camera, one continuous space, no page loads.
 *
 * The DOM grid is still rendered underneath and is still the truth: it
 * is what a reader without this bundle gets, what the keyboard walks,
 * and what selection reads. This mounts OVER it and takes its nodes FROM
 * it, so nothing here can disagree with what the server said.
 */

import { closestFrom, findElement } from "./dom";
import { register } from "./keys";
import { pin as keep, type Pin, board as pinned, unpin } from "./workspace";

/** Below this on-screen width a picture is drawn as its own average
 *  colour: the thumbnail would be a smear, and ten thousand smears cost
 *  more than ten thousand rectangles. */
const TINY = 26;

/** How much of the viewport a picture must cover before it IS the page. */
const PAGE_COVER = 0.62;

/** How many thumbnails may be in flight at once. A page of sixty asked
 *  for at once starves whichever ones the reader is actually looking at. */
const IN_FLIGHT = 6;

/** Uniform picture height in the time layout, world units. Every picture
 *  the same height is what makes a tower legible as a count. */
const TIME_H = 132;

/** The shortest a time axis may be, world units. Below this a handful of
 *  pictures would be drawn on top of each other. */
const TIME_W0 = 3200;

/** How much room the floating controls need at the top of the field, in
 *  screen pixels, so fitting does not park the first row underneath them. */
const TOP_INSET = 62;

/** Row height for the ranked layout, world units. The row WIDTH is not a
 *  constant: it is derived per layout from the frame's proportions, so
 *  the block ends up roughly the shape of the box it will be fitted
 *  into. A fixed width lays a long answer out as a tall narrow column
 *  and leaves two thirds of a wide field empty either side of it. */
const ROW_H = 260;

const GAP = 8;

/** How much width an empty stretch gets, however long it lasted. Fixed,
 *  because the point of drawing it is that it is EMPTY and how long for
 *  -- and a month of nothing rendered to scale is just distance. */
const VOID_W = 190;

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One unbroken stretch of picture-taking, and the emptiness after it.
 *  The time axis is a chain of these rather than one straight line. */
interface Run {
  t0: number;
  t1: number;
  x0: number;
  x1: number;
  /** How many pictures it holds. The run's width is proportional to it,
   *  and it is what spreads a burst across the room it earned. */
  count: number;
  /** Seconds of nothing between this run and the next; 0 on the last. */
  gapAfter: number;
  /** The shape's sample moments that fall inside this run, ascending.
   *
   *  These are what a picture's position is measured against, and the
   *  reason is stability: they come from the whole answer and do not
   *  change when a different window is loaded. Spreading pictures by
   *  their rank among the LOADED ones instead meant every refetch moved
   *  the world under a stationary camera -- zoom in, and the pictures
   *  you were looking at were suddenly somewhere else. */
  at: number[];
}

interface Node {
  key: string;
  slug: string;
  name: string;
  kind: string;
  thumb: string | null;
  ar: number;
  moment: number | null;
  /** Whether `moment` is the interpretation, or only when the file
   *  landed. A copy date presented as a capture date is a lie the axis
   *  would tell silently, so the field marks these. */
  dated: boolean;
  copies: number;
  rank: Box;
  time: Box;
  box: Box;
  from: Box;
  tint: string;
  img: HTMLImageElement | null;
  full: HTMLImageElement | null;
  state: "cold" | "loading" | "warm" | "failed";
}

/** How big a card on the board is, in board units. */
const CARD_W = 300;
const CARD_H = 186;

/**
 * One thing kept on the board, ready to draw.
 *
 * The `Pin` is what persists; everything else here is what this drawing
 * of it needs and is thrown away on reload. A card knows how to answer
 * "what is in there" only by asking the same endpoints the field asks,
 * so a pinned question is re-answered every time it is looked at rather
 * than remembering a count that slowly stops being true.
 */
/** What two questions have in common, as the server counted it. */
interface Against {
  left: number;
  right: number;
  both: number;
  only_left: number;
  only_right: number;
  shared: Array<{ thumb?: string | null }>;
  left_only: Array<{ thumb?: string | null }>;
  right_only: Array<{ thumb?: string | null }>;
}

interface Card {
  pin: Pin;
  /** A compare card only: the arithmetic, once it has answered. */
  against?: Against;
  box: Box;
  /** Up to four covers, so a card shows what is inside it. */
  covers: HTMLImageElement[];
  /** How many the pinned question holds; null until it has answered. */
  held: number | null;
  state: "cold" | "loading" | "warm" | "failed";
}

interface Camera {
  /** The world point at the centre of the viewport. */
  x: number;
  y: number;
  /** Screen pixels per world unit. */
  k: number;
}

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

const easeOut = (t: number): number => 1 - (1 - t) ** 3;

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

const lerpBox = (a: Box, b: Box, t: number): Box => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  w: lerp(a.w, b.w, t),
  h: lerp(a.h, b.h, t),
});

/** A colour to stand in for a picture that has not loaded. Derived from
 *  the slug so it is stable across reloads and two pictures beside each
 *  other rarely land on the same one -- a field of one grey is a field
 *  with no shape in it. */
function seeded(slug: string): string {
  let h = 0;
  for (let i = 0; i < slug.length; i++) h = (h * 31 + slug.charCodeAt(i)) | 0;
  return `hsl(${Math.abs(h) % 360} 14% 22%)`;
}

/** What a canvas can read out of a stylesheet: the tokens, resolved. */
function token(root: HTMLElement, name: string): string {
  return getComputedStyle(root).getPropertyValue(name).trim() || "#888";
}

/** The average colour of a loaded picture, for the far-out view.
 *
 *  One pixel: the browser's own downscale is the average, and it is
 *  done on the GPU. Thumbnails are served by this application, so the
 *  canvas is never tainted and the read is allowed. */
function averaged(img: HTMLImageElement): string | null {
  try {
    const board = document.createElement("canvas");
    board.width = 1;
    board.height = 1;
    const hand = board.getContext("2d", { willReadFrequently: true });
    if (!hand) return null;
    hand.drawImage(img, 0, 0, 1, 1);
    const [r, g, b] = hand.getImageData(0, 0, 1, 1).data;
    return `rgb(${r ?? 0} ${g ?? 0} ${b ?? 0})`;
  } catch {
    // A picture from somewhere else would throw here rather than answer
    // wrongly. The seeded colour is already on the node.
    return null;
  }
}

/**
 * A wheel event's vertical delta, in pixels, whatever unit it arrived in.
 *
 * `deltaY` is NOT pixels. It is expressed in whatever `deltaMode` says --
 * pixels, lines or pages -- and the spec's own words are "You must check
 * the deltaMode property to determine the unit... Do not assume that
 * those values are specified in pixels"
 * (../refs/mdn/content/files/en-us/web/api/wheelevent/deltamode/index.md:17-21).
 * A notch is about 100 in pixel mode and about 3 in line mode, so
 * treating them alike makes the same gesture zoom thirty times slower on
 * whichever browser reports lines.
 *
 * The same paragraph carries a stranger rule: some browsers change which
 * unit they report DEPENDING ON WHETHER `deltaMode` HAS BEEN READ. So
 * this reads it every time rather than only when it looks unusual --
 * touching the property is part of getting the right answer.
 */
function pixels(event: WheelEvent): number {
  if (event.deltaMode === 1) return event.deltaY * 16; // a line
  if (event.deltaMode === 2) return event.deltaY * window.innerHeight; // a page
  return event.deltaY;
}

/** Spell a moment the way a label under an axis should: as much as the
 *  span needs and no more. */
function spelled(at: number, span: number): string {
  const d = new Date(at * 1000);
  if (span > 86400 * 365 * 4) return String(d.getFullYear());
  if (span > 86400 * 120) return d.toLocaleDateString(undefined, { month: "short", year: "numeric" });
  if (span > 86400 * 3) return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export function mountField(root: ParentNode): void {
  const found = findElement(root, "[data-field]", HTMLElement);
  const surface = findElement(root, "[data-field-canvas]", HTMLCanvasElement);
  const held = findElement(root, "[data-cells]", HTMLElement);
  // The grid is OPTIONAL. On `/g` the field mounts over the server's
  // justified rows and reads them so something is on screen before any
  // fetch returns; on `/field` there are no rows at all, because that
  // page is the canvas and nothing else. Requiring them meant the
  // canvas-only surface silently refused to mount.
  if (!found || !surface) return;
  const context = surface.getContext("2d");
  if (!context) return;
  // Re-bound after the guard: every function below is a declaration, and
  // a declaration is hoisted, so TypeScript will not carry the narrowing
  // from an `if` above it into a body that could have been called first.
  const stage = found;
  const board = surface;
  const cells = held;
  const hand = context;

  const chip = findElement(stage, "[data-field-chip]", HTMLElement);
  const sheet = findElement(stage, "[data-field-sheet]", HTMLElement);
  const sheetName = findElement(stage, "[data-field-name]", HTMLElement);
  const sheetWhen = findElement(stage, "[data-field-when]", HTMLElement);
  const sheetOpen = findElement(stage, "[data-field-open]", HTMLAnchorElement);
  const clock = findElement(stage, "[data-field-clock]", HTMLElement);
  const count = findElement(stage, "[data-field-count]", HTMLElement);

  let nodes: Node[] = [];
  // Three arrangements, and BOARD is a different kind of thing from the
  // other two. Order and Time arrange the pictures of one answer. The
  // board arranges ANSWERS -- a question, a person, an album, a single
  // photograph -- so it is the surface you keep, and the other two are
  // what you see once you have gone into one.
  let mode: "time" | "rank" | "board" = "rank";
  let cards: Card[] = [];
  let cam: Camera = { x: 0, y: 0, k: 0.2 };
  let bounds: Box = { x: 0, y: 0, w: 1000, h: 1000 };
  let hovering: Node | null = null;
  let page: Node | null = null;
  let width = 0;
  let height = 0;
  /** The axis in pieces: runs of picture-taking with compressed, marked
   *  emptiness between them. Both the placing and the rules read it, so
   *  a label can never sit somewhere its pictures are not. */
  let runs: Run[] = [];

  /** A layout change in progress: nodes lerp from `from` to the mode's box. */
  let morphAt = 0;
  const MORPH = 620;

  /** A camera move in progress. */
  let flight: { from: Camera; to: Camera; at: number; ms: number } | null = null;

  let drawing = false;
  let loading = 0;
  /** Whether the WHOLE answer is in hand, rather than the page of it the
   *  server rendered. Once it is, the page's own cells must not be read
   *  again -- they are one page of sixty-three, and re-reading them
   *  would throw the other sixty-two away. */
  let whole = false;
  /** How many members the whole answer holds -- from the shape, so it is
   *  the answer's own count and never how many happen to be drawn. */
  let total = 0;
  /** The first and last moment the answer covers, from the shape. */
  let span: [number, number] = [0, 0];
  /** The shape: every nth moment of the WHOLE answer, ascending, and how
   *  many members each one stands for. The axis is built from these, so
   *  it describes the answer rather than describing whichever pictures
   *  happen to be loaded. */
  let samples: number[] = [];
  let stride = 1;
  /** The stretch of time the loaded pictures came from, and whether that
   *  stretch was cut short. */
  let covering: [number, number] | null = null;
  let cut = false;
  /** A window fetch in flight, so a drag does not queue fifty of them. */
  let asking = false;
  let asked = 0;
  /** Whether a window has ever landed. The first one frames the view;
   *  every later one arrives under a camera the reader is holding. */
  let settled = false;

  // ── where the pictures come from ──────────────────────────────────

  /** One node, with every drawing field at rest. */
  function made(
    key: string,
    slug: string,
    name: string,
    kind: string,
    thumb: string | null,
    ar: number,
    moment: number | null,
    dated: boolean,
    copies: number,
  ): Node {
    return {
      key,
      slug,
      name,
      kind,
      thumb,
      ar,
      moment,
      dated,
      copies,
      rank: { x: 0, y: 0, w: 0, h: 0 },
      time: { x: 0, y: 0, w: 0, h: 0 },
      box: { x: 0, y: 0, w: 0, h: 0 },
      from: { x: 0, y: 0, w: 0, h: 0 },
      tint: seeded(slug),
      img: null,
      full: null,
      state: "cold",
    };
  }

  /**
   * The whole answer, in one fetch.
   *
   * The grid is a page because a page is what fits on a screen as rows.
   * The field has no rows: it puts every picture where its moment says,
   * and the shape that makes -- the bursts, and the stretches of nothing
   * between them -- does not exist in sixty of three thousand. Drawing a
   * page of it is not a smaller version of the picture, it is a
   * different and wrong one.
   */
  /** The question this page is asking, as the field asks it. */
  function question(): URLSearchParams {
    const asked = new URLSearchParams(window.location.search);
    // Not part of the question the field asks: it has no pages, no page
    // size, and `view` is which surface is drawing, not what is in it.
    for (const drop of ["page", "size", "view"]) asked.delete(drop);
    return asked;
  }

  async function fetchAnswer(): Promise<void> {
    const asked = question();
    try {
      // LEVEL ONE: the shape. Where this answer's pictures fall in time,
      // for the WHOLE answer, at a size that does not depend on how big
      // the answer is -- a few kilobytes for four hundred pictures or
      // four hundred thousand. This is what the axis is built from, so
      // the axis is true even where the pictures are not loaded.
      const outline = await fetch(`/g/field/shape?${asked}`, { headers: { accept: "application/json" } });
      if (!outline.ok) return;
      const shape = (await outline.json()) as { total?: number; samples?: number[]; stride?: number };
      const stamps = shape.samples;
      if (!Array.isArray(stamps) || stamps.length < 1) return;
      span = [stamps[0] ?? 0, stamps[stamps.length - 1] ?? 0];
      total = shape.total ?? 0;
      samples = stamps;
      stride = Math.max(1, shape.stride ?? 1);
      await fetchWindow(span[0], span[1]);
    } catch {
      // The page's own cells are already drawn and are a true answer,
      // just a shorter one. A field of one page beats an empty field.
    }
  }

  /**
   * LEVEL TWO: the pictures, for one stretch of time.
   *
   * Bounded by the screen rather than by the library, which is why there
   * is no ceiling on how large an answer this surface draws: zooming in
   * NARROWS the window, so looking closer costs less than looking wide.
   * This is what makes "zoom in" a working instruction rather than
   * advice the field cannot honour.
   */
  async function fetchWindow(after: number, before: number): Promise<void> {
    if (asking) return;
    asking = true;
    const mine = ++asked;
    try {
      const wanted = new URLSearchParams(question());
      wanted.set("after", String(after));
      wanted.set("before", String(before));
      const answer = await fetch(`/g/field/window?${wanted}`, { headers: { accept: "application/json" } });
      if (!answer.ok || mine !== asked) return;
      const told: unknown = await answer.json();
      if (!told || typeof told !== "object") return;
      const held = told as { held?: number; more?: number; items?: unknown };
      if (!Array.isArray(held.items) || !held.items.length) return;
      nodes = held.items.map((raw): Node => {
        const one = raw as {
          slug?: string;
          name?: string;
          thumb?: string | null;
          ar?: number;
          moment?: number | null;
          dated?: boolean;
          copies?: number | null;
        };
        const slug = one.slug ?? "";
        return made(
          slug,
          slug,
          one.name ?? slug,
          "image",
          one.thumb ?? null,
          one.ar && Number.isFinite(one.ar) ? one.ar : 1,
          typeof one.moment === "number" && Number.isFinite(one.moment) ? one.moment : null,
          one.dated !== false,
          one.copies ?? 1,
        );
      });
      whole = true;
      covering = [after, before];
      cut = (held.more ?? 0) > 0;
      // What the field is showing, out loud. The window says how many
      // fell inside it and how many of those it did not name, and a
      // canvas that drew nine hundred of four thousand without saying so
      // would be lying about how dense that stretch of time is.
      if (count) {
        count.textContent = cut
          ? `${nodes.length.toLocaleString()} of the ${(held.held ?? 0).toLocaleString()} here — zoom in for the rest`
          : `${nodes.length.toLocaleString()} of ${total.toLocaleString()}`;
        count.hidden = false;
      }
      const first = !settled;
      settled = true;
      layout();
      for (const n of nodes) {
        n.box = mode === "time" ? { ...n.time } : { ...n.rank };
        n.from = { ...n.box };
      }
      // Only the FIRST window frames the view. A refetch is the reader
      // arriving somewhere and asking for what is there -- yanking the
      // camera back to fit would undo the move that asked for it.
      if (first) {
        resize();
        fit(false);
      } else draw();
    } catch {
      // The page's own cells are already drawn and are a true answer,
      // just a shorter one. A field of one page beats an empty field.
    } finally {
      asking = false;
    }
  }

  /** The pictures the page itself rendered: what the field draws until
   *  the whole answer arrives, and all it ever draws if that fetch
   *  fails. */
  function ingest(): void {
    // Once the whole answer is in hand the page's own page-worth of
    // cells is a subset of it, and re-reading them would throw away the
    // other sixty-two pages. And `/field` has no cells at all.
    if (whole || !cells) return;
    const seen = new Map(nodes.map((n) => [n.key, n]));
    const held: Node[] = [];
    for (const shell of cells.querySelectorAll("[data-selection-key]")) {
      if (!(shell instanceof HTMLElement)) continue;
      const key = shell.dataset.selectionKey;
      if (!key) continue;
      const kept = seen.get(key);
      if (kept) {
        held.push(kept);
        continue;
      }
      const link = shell.querySelector("[data-slug]");
      if (!(link instanceof HTMLElement)) continue;
      const slug = link.dataset.slug ?? "";
      const picture = shell.querySelector("img");
      const raw = shell.dataset.moment;
      const moment = raw ? Number(raw) : null;
      held.push(
        made(
          key,
          slug,
          shell.querySelector(".cell-name")?.textContent?.trim() ?? slug,
          link.dataset.kind ?? "image",
          picture instanceof HTMLImageElement ? picture.src : null,
          // `--ar` is what the server already computed for the justified
          // grid: the picture's own proportion, or 1 for a file nothing
          // has measured. Reading it keeps one answer to that question.
          Number(getComputedStyle(shell).getPropertyValue("--ar")) || 1,
          moment !== null && Number.isFinite(moment) ? moment : null,
          shell.dataset.dated !== "file",
          Number(shell.querySelector(".cell-copies")?.textContent ?? "1") || 1,
        ),
      );
    }
    const grew = held.length !== nodes.length;
    nodes = held;
    layout();
    if (grew) {
      for (const n of nodes) {
        n.box = mode === "time" ? { ...n.time } : { ...n.rank };
        n.from = { ...n.box };
      }
    }
  }

  // ── where each picture goes ───────────────────────────────────────

  /** The answer's own order, justified into rows of one height. What the
   *  DOM grid draws, in world units. */
  function layoutRanked(): void {
    // How wide to make the block, so the block ends up roughly the shape
    // of the frame it will be fitted into. A fixed width lays a long
    // answer out as a tall narrow column, and fitting THAT to a wide
    // frame leaves two thirds of the field empty either side.
    const area = nodes.reduce((sum, n) => sum + ROW_H * ROW_H * n.ar, 0);
    const shape = Math.max(0.5, width / Math.max(1, height - TOP_INSET));
    const wide = clamp(Math.sqrt(area * shape), ROW_H * 3, ROW_H * 40);
    let x = 0;
    let y = 0;
    let row: Node[] = [];
    const flush = (last: boolean): void => {
      if (!row.length) return;
      // Every row but the last is stretched to the full width, which is
      // what makes rows read as rows instead of as a ragged edge.
      const used = row.reduce((sum, n) => sum + ROW_H * n.ar, 0);
      const gaps = GAP * (row.length - 1);
      const scale = last ? 1 : (wide - gaps) / used;
      const h = ROW_H * scale;
      let at = 0;
      for (const n of row) {
        const w = h * n.ar;
        n.rank = { x: at, y, w, h };
        at += w + GAP;
      }
      y += h + GAP;
      row = [];
      x = 0;
    };
    for (const n of nodes) {
      const w = ROW_H * n.ar;
      if (x > 0 && x + w > wide) flush(false);
      row.push(n);
      x += w + GAP;
    }
    flush(true);
  }

  /**
   * Time, as a place.
   *
   * x is the moment. Pictures are all one height, so width is the only
   * thing that varies and a column of them counts. Where two would
   * overlap the later one drops a lane -- which means a burst grows
   * UPWARD as a tower and a quiet stretch is empty space you can see
   * across. Nothing is binned: the gaps are real gaps.
   *
   * There is no undated band, because there is no undated picture. A
   * file the context job has not reached still arrived on a day, and the
   * server sends that day rather than a null -- so everything is placed,
   * and the ones standing on their mtime are MARKED instead of being
   * swept into a corner or drawn at the epoch.
   */
  function layoutTime(): void {
    const placed = nodes.filter((n) => n.moment !== null);
    if (!placed.length) {
      // Nothing carries a moment at all, which means the markup did not
      // render one. Time cannot say anything, so it says the same thing
      // order does rather than drawing one meaningless column.
      for (const n of nodes) n.time = { ...n.rank };
      return;
    }
    const first = Math.min(...placed.map((n) => n.moment ?? 0));
    const sorted = [...nodes].sort((a, b) => (a.moment ?? first) - (b.moment ?? first));
    // The axis comes from the SHAPE when there is one -- every nth
    // moment of the whole answer -- so it describes the answer and not
    // whichever window is loaded. Without it the axis would redraw
    // itself every time the camera moved, because the pictures it was
    // measuring would have changed.
    runs =
      samples.length > 1
        ? axisOf(samples, stride)
        : axisOf(
            sorted.map((n) => n.moment ?? first),
            1,
          );

    // The right edge of the last picture in each lane, so a picture takes
    // the lowest lane it does not collide in. Lane 0 is the axis and
    // towers grow upward, which is why y is negative.
    // The runs are in time order and so is `sorted`, so one pointer
    // walks both -- no search.
    const lanes: number[] = [];
    let r = 0;
    sorted.forEach((n) => {
      const t = n.moment ?? first;
      while (r < runs.length - 1 && t > (runs[r]?.t1 ?? 0)) r += 1;
      const run = runs[r];
      const w = TIME_H * n.ar;
      const x = run ? placed_at(run, t) : 0;
      let lane = lanes.findIndex((edge) => x >= edge);
      if (lane === -1) {
        lane = lanes.length;
        lanes.push(0);
      }
      lanes[lane] = x + w + GAP;
      n.time = { x, y: -lane * (TIME_H + GAP), w, h: TIME_H };
    });
  }

  /**
   * Where one picture sits inside its run.
   *
   * Mostly by POSITION IN THE RUN, a little by the clock. Placing purely
   * by the clock is the pure idea and it collapses: three hundred frames
   * shot in ninety seconds all land on one x, the lane packing stacks
   * them into a column three hundred high, and the width the run earned
   * goes unused. Placing purely by order is a strip that has forgotten
   * time. The blend spreads a burst across its room while keeping the
   * rhythm inside it, so a pause in the middle of a shoot still opens a
   * space.
   */
  function placed_at(run: Run, t: number): number {
    const width = run.x1 - run.x0;
    const marks = run.at;
    if (marks.length < 2) return run.x0 + width / 2;
    // Where this moment falls among the run's sample moments. Because
    // the samples are evenly spaced BY RANK across the whole answer,
    // this is the picture's approximate position in the answer's own
    // order -- which is what spreads a burst across the room its run
    // earned, instead of piling every frame of one afternoon onto a
    // single x and building a tower three hundred high.
    let lo = 0;
    let hi = marks.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((marks[mid] ?? 0) < t) lo = mid + 1;
      else hi = mid;
    }
    // Between two samples, by the clock: two pictures inside one
    // sampling interval still land in the order they were taken.
    const before = marks[Math.max(0, lo - 1)] ?? t;
    const after = marks[lo] ?? t;
    const inner = after > before ? clamp((t - before) / (after - before), 0, 1) : 0;
    return run.x0 + ((Math.max(0, lo - 1) + inner) / (marks.length - 1)) * width;
  }

  /**
   * The axis, in pieces: a run of pictures, then a gap, then a run.
   *
   * A straight line from the first moment to the last is the honest axis
   * and the useless one. Pictures are not spread evenly over the years
   * they cover -- they are in bursts with nothing between them -- so a
   * proportional axis spends most of its width on emptiness and squeezes
   * every afternoon that mattered into a smear a few pixels wide.
   *
   * Empty stretches are COMPRESSED to a fixed width and MARKED, which is
   * what the timeline surface already does with its skipped bands: a gap
   * you can see and name is information, a gap drawn to scale is only
   * distance. Each run then gets width in proportion to how many
   * pictures it holds, so a busy afternoon has room to be looked at and
   * a single frame does not take an inch to itself.
   */
  function axisOf(times: number[], weight: number): Run[] {
    if (times.length < 2) return [];
    const steps: number[] = [];
    for (let i = 1; i < times.length; i++) steps.push((times[i] ?? 0) - (times[i - 1] ?? 0));
    // A gap is a stretch far longer than the typical step between two
    // pictures, floored at an hour so one burst does not shatter into
    // hundreds of runs. Derived from THIS answer rather than fixed: a
    // library of one wedding and a library of fifteen years have very
    // different ideas of how long "a while" is.
    const ordered = [...steps].sort((a, b) => a - b);
    const median = ordered[Math.floor(ordered.length / 2)] ?? 0;
    const wide = Math.max(3600, median * 60);

    const cuts: Array<{ t0: number; t1: number; count: number; at: number[] }> = [];
    let start = times[0] ?? 0;
    let at: number[] = [start];
    for (let i = 1; i < times.length; i++) {
      const t = times[i] ?? 0;
      if ((steps[i - 1] ?? 0) > wide) {
        cuts.push({ t0: start, t1: times[i - 1] ?? start, count: at.length, at });
        start = t;
        at = [t];
      } else at.push(t);
    }
    cuts.push({ t0: start, t1: times[times.length - 1] ?? start, count: at.length, at });

    // Width by PICTURE COUNT, so room follows what there is to look at.
    // `weight` is how many members each entry in `times` stands for: 1
    // when these are the pictures themselves, the shape's stride when
    // they are samples of them. It is what makes a run's width mean "how
    // much was shot here" rather than "how many samples landed here".
    const total = cuts.reduce((sum, c) => sum + c.count, 0);
    const content = Math.max(TIME_W0, total * weight * TIME_H * 0.42);
    const built: Run[] = [];
    let x = 0;
    cuts.forEach((c, i) => {
      const w = (c.count / total) * content;
      built.push({ t0: c.t0, t1: c.t1, x0: x, x1: x + w, count: c.count * weight, gapAfter: 0, at: c.at });
      x += w;
      const next = cuts[i + 1];
      if (next) {
        const held = built[built.length - 1];
        if (held) held.gapAfter = next.t0 - c.t1;
        x += VOID_W;
      }
    });
    return built;
  }

  /** What moment a place on the axis stands for -- the inverse of the
   *  placement, including inside a compressed gap, so panning into one
   *  asks for the right stretch of time rather than for its edges. */
  function timeAt(x: number): number {
    const opening = runs[0];
    if (!opening) return 0;
    if (x <= opening.x0) return opening.t0;
    for (const run of runs) {
      if (x <= run.x1) {
        const width = run.x1 - run.x0;
        return width <= 0 ? run.t0 : run.t0 + ((x - run.x0) / width) * (run.t1 - run.t0);
      }
      if (run.gapAfter && x <= run.x1 + VOID_W) {
        return run.t1 + ((x - run.x1) / VOID_W) * run.gapAfter;
      }
    }
    return runs[runs.length - 1]?.t1 ?? 0;
  }

  /**
   * Ask for the pictures where the camera now is.
   *
   * This is what makes "zoom in for the rest" TRUE rather than advice
   * the field cannot honour. The loaded window covers some stretch of
   * time and may have been cut short; when the view moves outside it, or
   * narrows enough that the same budget would now cover the stretch
   * whole, the field asks again. Zooming in therefore delivers more.
   */
  let refocusing = 0;
  function refocus(): void {
    if (mode !== "time" || !samples.length || !covering) return;
    window.clearTimeout(refocusing);
    // After the movement stops, not during it: a drag would otherwise
    // ask for a hundred windows on the way to the one that matters.
    refocusing = window.setTimeout(() => {
      if (!covering) return;
      const t0 = timeAt(cam.x - width / 2 / cam.k);
      const t1 = timeAt(cam.x + width / 2 / cam.k);
      const [c0, c1] = covering;
      const outside = t0 < c0 || t1 > c1;
      // Narrowed enough that the same budget would now cover this
      // stretch whole. Only worth asking when the last answer was cut.
      const closer = cut && t1 - t0 < (c1 - c0) * 0.6;
      if (outside || closer) void fetchWindow(t0, t1);
    }, 280);
  }

  function layout(): void {
    layoutRanked();
    layoutTime();
    layoutBoard();
    measure();
  }

  /** Cards sit where they were put. The board is the one arrangement
   *  this application does not compute -- somebody placed these. */
  function layoutBoard(): void {
    for (const card of cards) {
      card.box = { x: card.pin.x, y: card.pin.y, w: CARD_W, h: CARD_H };
    }
  }

  /**
   * Read the board, and ask each card what it holds.
   *
   * One request per card, to the same endpoint the field itself uses --
   * so a pinned question answers with today's library and a card can
   * never quietly show a count from the day it was pinned.
   */
  function loadBoard(): void {
    const held = pinned();
    const seen = new Map(cards.map((c) => [c.pin.id, c]));
    cards = held.map(
      (one) =>
        seen.get(one.id) ?? {
          pin: one,
          box: { x: one.x, y: one.y, w: CARD_W, h: CARD_H },
          covers: [],
          held: null,
          state: "cold" as const,
        },
    );
    // The stored pin is the truth about WHERE, even for a card that was
    // already drawn: dragging one writes the pin, and this is what makes
    // the drawing agree with it again.
    for (const card of cards) {
      const now = held.find((one) => one.id === card.pin.id);
      if (now) card.pin = now;
    }
    layoutBoard();
    for (const card of cards) void fillCard(card);
  }

  /** What is inside one card: a few covers and a count. */
  async function fillCard(card: Card): Promise<void> {
    if (card.state !== "cold") return;
    card.state = "loading";
    const show = (src: string): void => {
      const img = new Image();
      img.decoding = "async";
      img.addEventListener("load", () => {
        card.covers.push(img);
        draw();
      });
      img.src = src;
    };
    if (card.pin.kind === "compare") {
      const [one, two] = card.pin.against ?? ["", ""];
      const held = pinned();
      const left = held.find((p) => p.id === one);
      const right = held.find((p) => p.id === two);
      if (!left || !right) {
        // One of the two was taken off the board. The comparison is
        // still a card, and it says what happened rather than showing
        // three zeros as though the answer were empty.
        card.state = "failed";
        card.held = 0;
        draw();
        return;
      }
      try {
        const asked = new URLSearchParams({ a: left.at, b: right.at });
        const answer = await fetch(`/g/field/against?${asked}`, { headers: { accept: "application/json" } });
        if (!answer.ok) throw new Error(String(answer.status));
        card.against = (await answer.json()) as Against;
        card.held = card.against.both;
        card.state = "warm";
        // The DIFFERENCE, where there is one. Two questions that overlap
        // almost entirely are interesting for the handful that fell
        // outside, and a card showing four pictures both sides hold says
        // nothing a count did not already say. Only when nothing
        // differs does the overlap become the thing worth showing --
        // there, "these are the same pictures" IS the finding.
        const telling = [...card.against.left_only, ...card.against.right_only];
        for (const one of (telling.length ? telling : card.against.shared).slice(0, 4)) {
          if (one.thumb) show(one.thumb);
        }
        draw();
      } catch {
        card.state = "failed";
        draw();
      }
      return;
    }
    if (card.pin.kind === "picture") {
      card.held = 1;
      card.state = "warm";
      show(`/preview/${card.pin.at}`);
      return;
    }
    try {
      const asked = new URLSearchParams(card.pin.at);
      asked.set("after", "0");
      // Far past anything a photograph carries, so the window is "all of
      // it" without the caller having to know the answer's own span.
      asked.set("before", "99999999999");
      asked.set("most", "4");
      const answer = await fetch(`/g/field/window?${asked}`, { headers: { accept: "application/json" } });
      if (!answer.ok) throw new Error(String(answer.status));
      const told = (await answer.json()) as { held?: number; items?: Array<{ thumb?: string | null }> };
      card.held = told.held ?? 0;
      card.state = "warm";
      for (const one of told.items ?? []) if (one.thumb) show(one.thumb);
      draw();
    } catch {
      card.state = "failed";
      draw();
    }
  }

  function measure(): void {
    if (mode === "board") {
      // An empty board still needs an extent, or fitting it divides by
      // nothing and the camera lands at an undefined scale.
      if (!cards.length) {
        bounds = { x: -CARD_W, y: -CARD_H, w: CARD_W * 2, h: CARD_H * 2 };
        return;
      }
      let bx0 = Infinity;
      let by0 = Infinity;
      let bx1 = -Infinity;
      let by1 = -Infinity;
      for (const card of cards) {
        bx0 = Math.min(bx0, card.box.x);
        by0 = Math.min(by0, card.box.y);
        bx1 = Math.max(bx1, card.box.x + card.box.w);
        by1 = Math.max(by1, card.box.y + card.box.h);
      }
      bounds = { x: bx0, y: by0, w: bx1 - bx0, h: by1 - by0 };
      return;
    }
    if (!nodes.length) return;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const n of nodes) {
      const b = mode === "time" ? n.time : n.rank;
      x0 = Math.min(x0, b.x);
      y0 = Math.min(y0, b.y);
      x1 = Math.max(x1, b.x + b.w);
      y1 = Math.max(y1, b.y + b.h);
    }
    bounds = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  // ── the camera ────────────────────────────────────────────────────

  const fitScale = (): number =>
    Math.min(width / Math.max(1, bounds.w), (height - TOP_INSET) / Math.max(1, bounds.h)) * 0.94;

  /**
   * Where "show me this" lands, and it is not the same question in the
   * two arrangements.
   *
   * ORDER is a block: fit all of it, both ways, and you see the answer.
   * TIME is a ribbon -- a few lanes tall and very long -- and fitting all
   * of it means a hairline of pictures across an empty box. So time fits
   * the HEIGHT, fills the frame, and pans sideways, which is what a
   * timeline is; and it opens at the RIGHT end, on the most recent
   * pictures, because that is the end somebody came to look at.
   */
  function fit(animate = true): void {
    const whole = fitScale();
    const tall = ((height - TOP_INSET) / Math.max(1, bounds.h)) * 0.9;
    const k = mode === "time" ? clamp(tall, whole, 1.4) : whole;
    const to: Camera = {
      x: mode === "time" ? bounds.x + bounds.w - width / 2 / k + 40 : bounds.x + bounds.w / 2,
      // Centred in the band BELOW the floating controls rather than in
      // the whole box, so fitting never parks the first row under them.
      y: bounds.y + bounds.h / 2 - TOP_INSET / 2 / k,
      k,
    };
    if (animate) flyTo(to, 520);
    else {
      cam = to;
      draw();
    }
  }

  function flyTo(to: Camera, ms = 460): void {
    flight = { from: { ...cam }, to, at: performance.now(), ms };
    tick();
  }

  /** Push into one picture until it is the page. */
  function enter(n: Node): void {
    const b = n.box;
    const k = Math.min(width / b.w, height / b.h) * 0.86;
    flyTo({ x: b.x + b.w / 2, y: b.y + b.h / 2, k }, 480);
  }

  /**
   * Keep the camera over the work.
   *
   * An unbounded plane is a plane you can lose yourself on: two flicks
   * and the pictures are somewhere behind you with nothing on screen to
   * say which way. So panning is CLAMPED -- the camera may travel until
   * the far edge of the content reaches the middle of the frame, and no
   * further, which allows a picture to be looked at against empty space
   * on any side while never letting the content leave entirely.
   *
   * The margin is half a viewport rather than a fixed number of world
   * units, because how far is "too far" depends on how far out you are:
   * at a wide zoom half a screen is most of the library, and pushed in
   * close it is one photograph.
   */
  function anchor(): void {
    const halfW = width / 2 / cam.k;
    const halfH = height / 2 / cam.k;
    cam.x = clamp(cam.x, bounds.x - halfW / 2, bounds.x + bounds.w + halfW / 2);
    cam.y = clamp(cam.y, bounds.y - halfH / 2, bounds.y + bounds.h + halfH / 2);
  }

  const toWorld = (sx: number, sy: number): { x: number; y: number } => ({
    x: (sx - width / 2) / cam.k + cam.x,
    y: (sy - height / 2) / cam.k + cam.y,
  });

  function at(sx: number, sy: number): Node | null {
    const p = toWorld(sx, sy);
    // Backwards: later nodes are drawn over earlier ones, so the last
    // one that contains the point is the one being pointed at.
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      if (!n) continue;
      const b = n.box;
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return n;
    }
    return null;
  }

  // ── loading pictures, nearest first ───────────────────────────────

  function want(n: Node): void {
    if (n.state !== "cold" || !n.thumb || loading >= IN_FLIGHT) return;
    n.state = "loading";
    loading += 1;
    const img = new Image();
    img.decoding = "async";
    img.addEventListener("load", () => {
      loading -= 1;
      n.img = img;
      n.state = "warm";
      const mean = averaged(img);
      if (mean) n.tint = mean;
      draw();
    });
    img.addEventListener("error", () => {
      loading -= 1;
      n.state = "failed";
      draw();
    });
    img.src = n.thumb;
  }

  /** The full file, only for the picture that has become the page. A
   *  thumbnail blown up to fill a screen is the one place the small
   *  version is visibly the wrong file. */
  function wantFull(n: Node): void {
    if (n.full || !n.slug) return;
    const img = new Image();
    img.decoding = "async";
    img.addEventListener("load", () => {
      n.full = img;
      draw();
    });
    img.src = `/preview/${n.slug}`;
  }

  // ── drawing ───────────────────────────────────────────────────────

  function resize(): void {
    // How many real pixels one CSS pixel is worth. NOT clamped to 2: a
    // modern phone screen is often more than that
    // (../refs/mdn/content/files/en-us/web/api/window/devicepixelratio/index.md:28-29),
    // and clamping there is what makes a canvas photo viewer look soft
    // on exactly the devices whose screens are best. The ceiling of 3 is
    // a memory bound on a backing store that can be most of a screen --
    // stated, because it IS a tradeoff and not the platform's answer.
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const rect = board.getBoundingClientRect();
    width = Math.max(1, Math.round(rect.width));
    height = Math.max(1, Math.round(rect.height));
    board.width = Math.round(width * dpr);
    board.height = Math.round(height * dpr);
    // Writing width or height RESETS the whole context -- transform,
    // styles and the smoothing settings below -- so everything that must
    // hold for the life of the canvas is re-stated here and nowhere
    // else.
    hand.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Every picture in this field is drawn scaled, and the default
    // smoothing quality is "low"
    // (../refs/mdn/content/files/en-us/web/api/canvasrenderingcontext2d/imagesmoothingquality/index.md:31)
    // -- which on a downscaled photograph is visible aliasing along
    // every edge. For a surface whose whole job is showing photographs
    // that is the wrong default, and it is one line to correct.
    //
    // Firefox does not implement the quality hint at all (mdn
    // browser-compat-data: api.CanvasRenderingContext2D
    // .imageSmoothingQuality, firefox: false) and uses its own
    // resampling; the assignment is simply ignored there. `enabled` is
    // supported everywhere and is what the hint depends on.
    hand.imageSmoothingEnabled = true;
    hand.imageSmoothingQuality = "high";
    draw();
  }

  function draw(): void {
    if (drawing) return;
    drawing = true;
    requestAnimationFrame(() => {
      drawing = false;
      paint();
    });
  }

  function paint(): void {
    const ground = token(stage, "--sunken");
    const line = token(stage, "--line");
    const brand = token(stage, "--brand");
    const accent = token(stage, "--accent");
    const faint = token(stage, "--ink-faint");
    const panel = token(stage, "--panel");
    const ink = token(stage, "--ink");

    hand.save();
    hand.fillStyle = ground;
    hand.fillRect(0, 0, width, height);

    const k = cam.k;
    const left = cam.x - width / 2 / k;
    const right = cam.x + width / 2 / k;
    const top = cam.y - height / 2 / k;
    const bottom = cam.y + height / 2 / k;

    if (mode === "board") {
      paintBoard(panel, line, ink, faint, brand, accent);
      paintMinimap(accent, line, panel, faint);
      hand.restore();
      return;
    }

    if (mode === "time") paintTimeRules(left, right, line, faint, ground);

    // Nearest the middle first, so what a reader is looking at is what
    // gets the six slots in flight.
    const near = [...nodes].sort(
      (a, b) =>
        Math.abs(a.box.x - cam.x) + Math.abs(a.box.y - cam.y) - (Math.abs(b.box.x - cam.x) + Math.abs(b.box.y - cam.y)),
    );
    for (const n of near) {
      const b = n.box;
      if (b.x + b.w < left || b.x > right || b.y + b.h < top || b.y > bottom) continue;
      want(n);
    }

    for (const n of nodes) {
      const b = n.box;
      const sx = (b.x - cam.x) * k + width / 2;
      const sy = (b.y - cam.y) * k + height / 2;
      const sw = b.w * k;
      const sh = b.h * k;
      if (sx + sw < -40 || sx > width + 40 || sy + sh < -40 || sy > height + 40) continue;

      // One of several near-identical copies: the edges of the ones
      // underneath, so a stack reads as a stack at any zoom.
      if (n.copies > 1 && sw > TINY) {
        hand.fillStyle = n.tint;
        hand.globalAlpha = 0.5;
        for (let i = Math.min(n.copies - 1, 3); i > 0; i--) {
          hand.fillRect(sx + i * 3, sy - i * 3, sw, sh);
        }
        hand.globalAlpha = 1;
      }

      const picture = n === page && n.full ? n.full : n.img;
      if (sw < TINY || !picture) {
        hand.fillStyle = n.tint;
        hand.fillRect(sx, sy, sw, sh);
      } else {
        hand.drawImage(picture, sx, sy, sw, sh);
      }

      // Standing on its mtime rather than on an interpreted date. A bar
      // along the bottom edge, so a tower says which of its pictures are
      // only there because that is when the FILE landed. Marked, never
      // moved: the position is the best the library currently knows.
      if (!n.dated && sw > TINY) {
        hand.fillStyle = accent;
        hand.fillRect(sx, sy + sh - 3, sw, 3);
      }

      if (n === hovering && n !== page) {
        hand.strokeStyle = brand;
        hand.lineWidth = 2;
        hand.strokeRect(sx - 1, sy - 1, sw + 2, sh + 2);
      }
      // A name only where there is room for one; below that the picture
      // is the label.
      if (sw > 150 && n !== page) {
        // The page's own surfaces, not a hardcoded dark bar: this canvas
        // is drawn on whichever theme the reader is in, and white text
        // on near-black was invisible on the light one.
        hand.fillStyle = panel;
        hand.fillRect(sx, sy + sh - 22, sw, 22);
        hand.fillStyle = ink;
        hand.font = "500 12px system-ui, sans-serif";
        hand.textBaseline = "middle";
        const room = sw - 16;
        let text = n.name;
        while (hand.measureText(text).width > room && text.length > 4) text = `${text.slice(0, -5)}…`;
        hand.fillText(text, sx + 8, sy + sh - 11);
      }
    }

    // Where the whole answer is, and where you are inside it. A field
    // you can move a long way in needs to say when you are lost.
    paintMinimap(accent, line, panel, faint);
    hand.restore();
  }

  /**
   * The axis, under the pictures standing on it.
   *
   * Each run is labelled with when it began. Each gap is drawn as a
   * hatched band saying how long nothing happened -- which is a fact
   * about the library, and the one thing a proportional axis cannot say
   * because it renders the same fact as blank space that could equally
   * mean "nothing here" or "the drawing broke".
   */
  function paintTimeRules(left: number, right: number, line: string, faint: string, ground: string): void {
    if (runs.length < 1) return;
    const first = runs[0]?.t0 ?? 0;
    const span = Math.max(1, (runs[runs.length - 1]?.t1 ?? first) - first);
    const sxOf = (wx: number): number => (wx - cam.x) * cam.k + width / 2;
    /** A label with the ground behind it, so it is readable over the
     *  pictures standing on the axis rather than tangled in them. */
    const chipped = (text: string, x: number, y: number, centred = false): void => {
      const w = hand.measureText(text).width + 12;
      const cx = centred ? x - w / 2 : x;
      hand.fillStyle = ground;
      hand.globalAlpha = 0.88;
      hand.beginPath();
      hand.roundRect(cx, y - 3, w, 18, 4);
      hand.fill();
      hand.globalAlpha = 1;
      hand.fillStyle = faint;
      hand.fillText(text, cx + 6, y);
    };
    hand.save();
    hand.font = "500 11px system-ui, sans-serif";
    hand.textBaseline = "top";

    for (const run of runs) {
      if (run.x1 < left || run.x0 > right) continue;
      const x0 = sxOf(run.x0);
      // Where the run begins, and when. A rule rather than a tick: the
      // pictures stand on it, so it reads as ground.
      hand.strokeStyle = line;
      hand.lineWidth = 1;
      hand.beginPath();
      hand.moveTo(Math.round(x0) + 0.5, 0);
      hand.lineTo(Math.round(x0) + 0.5, height);
      hand.stroke();
      if (sxOf(run.x1) - x0 > 62) chipped(spelled(run.t0, span), x0 + 4, height - 24);

      if (!run.gapAfter) continue;
      // The emptiness after it: hatched, bounded and NAMED.
      const gx = sxOf(run.x1);
      const gw = VOID_W * cam.k;
      if (gx + gw < 0 || gx > width) continue;
      hand.save();
      hand.beginPath();
      hand.rect(gx, 0, gw, height);
      hand.clip();
      hand.strokeStyle = line;
      hand.lineWidth = 1;
      for (let d = -height; d < gw + height; d += 9) {
        hand.beginPath();
        hand.moveTo(gx + d, height);
        hand.lineTo(gx + d + height, 0);
        hand.stroke();
      }
      hand.restore();
      hand.strokeStyle = line;
      hand.setLineDash([4, 4]);
      hand.beginPath();
      hand.moveTo(Math.round(gx) + 0.5, 0);
      hand.lineTo(Math.round(gx) + 0.5, height);
      hand.moveTo(Math.round(gx + gw) + 0.5, 0);
      hand.lineTo(Math.round(gx + gw) + 0.5, height);
      hand.stroke();
      hand.setLineDash([]);
      if (gw > 74) chipped(`${lasted(run.gapAfter)}, nothing`, gx + gw / 2, height / 2, true);
    }
    hand.restore();
  }

  /** How long a stretch lasted, in the coarsest unit that is still true. */
  function lasted(seconds: number): string {
    const days = seconds / 86400;
    if (days >= 730) return `${Math.round(days / 365)} years`;
    if (days >= 60) return `${Math.round(days / 30)} months`;
    if (days >= 13) return `${Math.round(days / 7)} weeks`;
    if (days >= 1.5) return `${Math.round(days)} days`;
    const hours = seconds / 3600;
    return hours >= 1.5 ? `${Math.round(hours)} hours` : `${Math.max(1, Math.round(seconds / 60))} minutes`;
  }

  /**
   * The board: what this person keeps, drawn as cards.
   *
   * A card shows what is INSIDE it rather than naming it and hoping --
   * up to four covers from the question it stands for, its name, and how
   * many it holds today. That is the difference between a bookmark and a
   * thing you recognise at a glance, and it is why the count is fetched
   * rather than remembered.
   */
  function paintBoard(panel: string, line: string, ink: string, faint: string, brand: string, accent: string): void {
    if (!cards.length) {
      hand.fillStyle = faint;
      hand.font = "500 15px system-ui, sans-serif";
      hand.textAlign = "center";
      hand.textBaseline = "middle";
      hand.fillText("Nothing on the board yet — open a question and press Pin", width / 2, height / 2);
      hand.textAlign = "left";
      return;
    }
    const k = cam.k;
    for (const card of cards) {
      const b = card.box;
      const sx = (b.x - cam.x) * k + width / 2;
      const sy = (b.y - cam.y) * k + height / 2;
      const sw = b.w * k;
      const sh = b.h * k;
      if (sx + sw < -40 || sx > width + 40 || sy + sh < -40 || sy > height + 40) continue;

      const r = Math.min(14 * k, sh / 2);
      hand.save();
      hand.beginPath();
      hand.roundRect(sx, sy, sw, sh, Math.max(1, r));
      hand.fillStyle = panel;
      hand.fill();
      hand.strokeStyle = card === holding?.card ? brand : line;
      hand.lineWidth = card === holding?.card ? 2 : 1;
      hand.stroke();
      hand.clip();

      // The covers: a strip across the top, so the card is a window into
      // the question rather than a label about it.
      const coverH = sh * 0.7;
      if (card.covers.length) {
        const each = sw / card.covers.length;
        card.covers.forEach((img, i) => {
          hand.drawImage(img, sx + i * each, sy, each, coverH);
        });
      } else {
        hand.fillStyle = token(stage, "--sunken");
        hand.fillRect(sx, sy, sw, coverH);
      }

      // The words. Below a certain size they are unreadable and drawing
      // them is just noise on the card.
      if (sw > 120) {
        hand.fillStyle = ink;
        hand.font = `600 ${Math.min(15, Math.max(10, 15 * k))}px system-ui, sans-serif`;
        hand.textBaseline = "top";
        let name = card.pin.name;
        const room = sw - 22;
        while (hand.measureText(name).width > room && name.length > 4) name = `${name.slice(0, -5)}…`;
        hand.fillText(name, sx + 11, sy + coverH + 9 * k);

        hand.fillStyle = faint;
        hand.font = `400 ${Math.min(12.5, Math.max(9, 12.5 * k))}px system-ui, sans-serif`;
        // A comparison says what it FOUND, which is three numbers and
        // not a count: how many both hold, and how many each has that
        // the other does not. "1,842 pictures" would be true of the
        // overlap and would say nothing about the question asked.
        const said =
          card.state === "failed"
            ? card.pin.kind === "compare"
              ? "one of the two is gone"
              : "could not answer"
            : card.held === null
              ? "counting…"
              : card.pin.kind === "compare" && card.against
                ? card.against.only_left + card.against.only_right === 0
                  ? `the same ${card.against.both.toLocaleString()} pictures`
                  : `${card.against.both.toLocaleString()} in both · showing the ` +
                    `${(card.against.only_left + card.against.only_right).toLocaleString()} that differ`
                : card.pin.kind === "picture"
                  ? "one picture"
                  : `${card.held.toLocaleString()} pictures`;
        hand.fillText(said, sx + 11, sy + coverH + 28 * k);

        // What kind of thing it is, in the colour that means "where you
        // are" -- a pin is a place, never an action.
        hand.fillStyle = brand;
        hand.font = `600 ${Math.min(10.5, Math.max(8, 10.5 * k))}px system-ui, sans-serif`;
        hand.fillText(
          card.pin.kind.toUpperCase(),
          sx + sw - 11 - hand.measureText(card.pin.kind.toUpperCase()).width,
          sy + coverH + 10 * k,
        );
      }
      hand.restore();

      if (card === hoverCard && card !== holding?.card) {
        hand.strokeStyle = accent;
        hand.lineWidth = 2;
        hand.beginPath();
        hand.roundRect(sx - 1, sy - 1, sw + 2, sh + 2, Math.max(1, r));
        hand.stroke();
      }
    }
  }

  /** The whole answer, and where in it you are looking. A field you can
   *  travel a long way in has to say when you are lost. */
  function paintMinimap(accent: string, line: string, panel: string, faint: string): void {
    if (!nodes.length) return;
    const mw = 150;
    const mh = 34;
    const mx = width - mw - 14;
    const my = height - mh - 14;
    hand.save();
    hand.globalAlpha = 0.94;
    hand.fillStyle = panel;
    hand.beginPath();
    hand.roundRect(mx, my, mw, mh, 6);
    hand.fill();
    hand.strokeStyle = line;
    hand.lineWidth = 1;
    hand.stroke();
    hand.clip();
    const s = Math.min(mw / Math.max(1, bounds.w), mh / Math.max(1, bounds.h)) * 0.84;
    const ox = mx + mw / 2 - (bounds.x + bounds.w / 2) * s;
    const oy = my + mh / 2 - (bounds.y + bounds.h / 2) * s;
    hand.fillStyle = faint;
    for (const n of nodes) {
      const b = n.box;
      hand.fillRect(ox + b.x * s, oy + b.y * s, Math.max(1, b.w * s), Math.max(1, b.h * s));
    }
    hand.strokeStyle = accent;
    hand.lineWidth = 1.5;
    hand.strokeRect(
      ox + (cam.x - width / 2 / cam.k) * s,
      oy + (cam.y - height / 2 / cam.k) * s,
      (width / cam.k) * s,
      (height / cam.k) * s,
    );
    hand.restore();
  }

  // ── the animation clock ───────────────────────────────────────────

  function tick(): void {
    const now = performance.now();
    let more = false;

    if (morphAt) {
      const t = clamp((now - morphAt) / MORPH, 0, 1);
      const e = easeOut(t);
      for (const n of nodes) n.box = lerpBox(n.from, mode === "time" ? n.time : n.rank, e);
      if (t >= 1) morphAt = 0;
      else more = true;
    }

    if (flight) {
      const t = clamp((now - flight.at) / flight.ms, 0, 1);
      const e = easeOut(t);
      // Scale interpolates geometrically: a linear walk from 0.2 to 8
      // spends most of its time already arrived.
      cam = {
        x: lerp(flight.from.x, flight.to.x, e),
        y: lerp(flight.from.y, flight.to.y, e),
        k: flight.from.k * (flight.to.k / flight.from.k) ** e,
      };
      if (t >= 1) flight = null;
      else more = true;
    }

    settle();
    paint();
    if (more) requestAnimationFrame(tick);
  }

  /** Whether one picture has become the page, and the chrome that says so. */
  function settle(): void {
    let found: Node | null = null;
    for (const n of nodes) {
      if (n.box.h * cam.k >= height * PAGE_COVER && n.box.w * cam.k >= width * 0.4) {
        found = n;
        break;
      }
    }
    if (found === page) return;
    page = found;
    if (sheet) sheet.hidden = page === null;
    stage.dataset.fieldPage = page ? "" : "none";
    if (!page) return;
    wantFull(page);
    if (sheetName) sheetName.textContent = page.name;
    if (sheetOpen) sheetOpen.href = `/i/${page.slug}`;
    if (sheetWhen) {
      const when =
        page.moment === null
          ? null
          : new Date(page.moment * 1000).toLocaleString(undefined, {
              day: "numeric",
              month: "long",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            });
      // Which time this is, said out loud. A file's mtime shown as if it
      // were a capture date is the one claim this surface must never
      // make silently.
      sheetWhen.textContent =
        when === null
          ? "no time recorded"
          : page.dated
            ? when
            : `${when} — when the file landed. Nothing has read this one's own date yet.`;
      sheetWhen.dataset.dated = page.dated ? "read" : "file";
    }
  }

  // ── input ─────────────────────────────────────────────────────────

  const peekImage = findElement(stage, "[data-field-peek-image]", HTMLImageElement);
  const peekName = findElement(stage, "[data-field-peek-name]", HTMLElement);
  const peekWhen = findElement(stage, "[data-field-peek-when]", HTMLElement);

  /**
   * Show what is under the pointer, as the picture rather than as its
   * name.
   *
   * At the zooms this field is for, a picture can be twenty pixels
   * across, and "666A0271.CR2" is not something anybody recognises a
   * photograph by. The peek is the thumbnail the field already has in
   * hand, so it costs no fetch.
   *
   * It is kept INSIDE the field: a card that hangs off the right edge is
   * a card half off the screen on the last column, so it flips to the
   * other side of the pointer when there is not room.
   */
  function peek(over: Node | null, x: number, y: number): void {
    if (!chip) return;
    if (!over || over === page) {
      chip.hidden = true;
      return;
    }
    chip.hidden = false;
    if (peekImage && over.thumb && peekImage.getAttribute("src") !== over.thumb) {
      peekImage.src = over.thumb;
    }
    if (peekName) peekName.textContent = over.name;
    if (peekWhen) {
      peekWhen.textContent =
        over.moment === null
          ? "no time recorded"
          : new Date(over.moment * 1000).toLocaleString(undefined, {
              day: "numeric",
              month: "short",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            });
      peekWhen.dataset.dated = over.dated ? "read" : "file";
    }
    const box = chip.getBoundingClientRect();
    const left = x + 18 + box.width > width ? x - 18 - box.width : x + 18;
    const top = y + 18 + box.height > height ? y - 18 - box.height : y + 18;
    chip.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
  }

  let dragging = false;
  let moved = 0;
  let lastX = 0;
  let lastY = 0;
  /** The card under the pointer, and the one being moved. A card is
   *  dragged with the same gesture that pans the board, so which of the
   *  two is happening is decided by what was under the finger when it
   *  went down. */
  let hoverCard: Card | null = null;
  let holding: { card: Card; dx: number; dy: number } | null = null;

  /** The card at a point on screen, topmost first. */
  /** The topmost card at a point, ignoring one of them.
   *
   *  Used while dragging, where the card in hand is always under the
   *  pointer and is never what the pointer is over. */
  function cardUnder(sx: number, sy: number, except: Card): Card | null {
    const p = toWorld(sx, sy);
    for (let i = cards.length - 1; i >= 0; i--) {
      const card = cards[i];
      if (!card || card === except) continue;
      const b = card.box;
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return card;
    }
    return null;
  }

  function cardAt(sx: number, sy: number): Card | null {
    const p = toWorld(sx, sy);
    for (let i = cards.length - 1; i >= 0; i--) {
      const card = cards[i];
      if (!card) continue;
      const b = card.box;
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return card;
    }
    return null;
  }

  /**
   * Go into what a card stands for.
   *
   * A pinned question opens as the field for that question -- the same
   * canvas, a different answer in it. A pinned photograph opens as its
   * own page, because there is nothing to arrange about one picture.
   */
  function openCard(card: Card): void {
    // Every kind but a picture is a QUESTION, and the field is what
    // answers a question -- so a person, an album and a folder all open
    // the same way, with their own clause in the address.
    window.location.href =
      card.pin.kind === "picture" ? `/i/${card.pin.at}` : `/field${card.pin.at ? `?${card.pin.at}` : ""}`;
  }

  board.addEventListener("pointerdown", (event) => {
    dragging = true;
    moved = 0;
    lastX = event.clientX;
    lastY = event.clientY;
    board.setPointerCapture(event.pointerId);
    board.style.cursor = "grabbing";
    if (mode === "board") {
      const rect = board.getBoundingClientRect();
      const under = cardAt(event.clientX - rect.left, event.clientY - rect.top);
      if (under) {
        const p = toWorld(event.clientX - rect.left, event.clientY - rect.top);
        holding = { card: under, dx: p.x - under.box.x, dy: p.y - under.box.y };
        // The card comes to the top: whatever you are moving should be
        // the thing you can see.
        cards = [...cards.filter((c) => c !== under), under];
      }
    }
    // A finger never hovers, so it would never see the peek. Pressing
    // shows it; letting go without moving is still the tap that opens
    // the picture, so looking costs nothing and commits to nothing.
    if (event.pointerType !== "mouse") {
      const rect = board.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      hovering = at(x, y);
      peek(hovering, x, y);
      draw();
    }
  });

  board.addEventListener("pointermove", (event) => {
    const rect = board.getBoundingClientRect();
    if (holding && dragging) {
      const p = toWorld(event.clientX - rect.left, event.clientY - rect.top);
      moved += Math.abs(event.clientX - lastX) + Math.abs(event.clientY - lastY);
      lastX = event.clientX;
      lastY = event.clientY;
      holding.card.box.x = p.x - holding.dx;
      holding.card.box.y = p.y - holding.dy;
      // What dropping here would do, shown before it is done: the card
      // underneath lights up, and letting go holds the two against each
      // other.
      hoverCard = cardUnder(event.clientX - rect.left, event.clientY - rect.top, holding.card);
      draw();
      return;
    }
    if (mode === "board" && !dragging) {
      const over = cardAt(event.clientX - rect.left, event.clientY - rect.top);
      if (over !== hoverCard) {
        hoverCard = over;
        board.style.cursor = over ? "pointer" : "grab";
        draw();
      }
      return;
    }
    if (dragging) {
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      cam.x -= dx / cam.k;
      cam.y -= dy / cam.k;
      anchor();
      lastX = event.clientX;
      lastY = event.clientY;
      flight = null;
      settle();
      refocus();
      draw();
      return;
    }
    const over = at(event.clientX - rect.left, event.clientY - rect.top);
    if (over !== hovering) {
      hovering = over;
      board.style.cursor = over ? "pointer" : "grab";
      draw();
    }
    peek(hovering, event.clientX - rect.left, event.clientY - rect.top);
  });

  const release = (event: PointerEvent): void => {
    if (!dragging) return;
    dragging = false;
    board.style.cursor = hovering ? "pointer" : "grab";
    // The peek belongs to the press on a touch screen; a pointer that
    // hovers will put it back on the next move.
    if (event.pointerType !== "mouse") {
      hovering = null;
      peek(null, 0, 0);
      draw();
    }
    if (holding) {
      const moving = holding;
      holding = null;
      if (moved > 6) {
        // DROPPED ON another card: hold the two questions against each
        // other. A gesture rather than a mode, because what is being
        // asked -- "what do these two have in common" -- is about two
        // objects, and putting one on the other is how that is said.
        const where = board.getBoundingClientRect();
        const onto = cardUnder(event.clientX - where.left, event.clientY - where.top, moving.card);
        if (onto && onto.pin.kind !== "compare" && moving.card.pin.kind !== "compare") {
          keep({
            id: `pin-${Date.now().toString(36)}`,
            kind: "compare",
            name: `${moving.card.pin.name} vs ${onto.pin.name}`,
            at: "",
            against: [moving.card.pin.id, onto.pin.id],
            x: Math.round((moving.card.pin.x + onto.pin.x) / 2),
            y: Math.round(Math.max(moving.card.pin.y, onto.pin.y) + CARD_H + 40),
          });
          // The dragged card goes back where it was: the drag WAS the
          // gesture, not a move, and leaving it on top of the other one
          // would hide the card it was asked about.
          moving.card.box.x = moving.card.pin.x;
          moving.card.box.y = moving.card.pin.y;
          hoverCard = null;
          loadBoard();
          measure();
          // Brought into view. A comparison that appears below the fold
          // looks like a gesture that did nothing, which is the one way
          // this could read as broken while working perfectly.
          fit();
          tally();
          return;
        }
        // Where it was put is what persists. Written on release rather
        // than on every frame of the drag: one arrangement, one write.
        keep({ ...moving.card.pin, x: Math.round(moving.card.box.x), y: Math.round(moving.card.box.y) });
        moving.card.pin = { ...moving.card.pin, x: moving.card.box.x, y: moving.card.box.y };
        measure();
        draw();
        return;
      }
      openCard(moving.card);
      return;
    }
    if (moved > 6) return;
    const rect = board.getBoundingClientRect();
    const hit = at(event.clientX - rect.left, event.clientY - rect.top);
    // A click is a step INWARD, never a navigation: pushing into a
    // picture is what opening it means here, and clicking the ground
    // steps back out to the whole answer.
    if (hit) enter(hit);
    else if (page) fit();
  };
  board.addEventListener("pointerup", release);
  // A drag can end without a pointerup: the browser cancels the pointer
  // when the gesture becomes a viewport pan or pinch, when the screen
  // rotates, when palm rejection fires, or when too many fingers are
  // down -- and it may cancel every pointer WHILE THE FINGER IS STILL ON
  // THE GLASS
  // (../refs/mdn/content/files/en-us/web/api/element/pointercancel_event/index.md:11-19).
  // So this restores everything `release` would have, minus the click:
  // a cancelled gesture is not a choice, and leaving the grabbing cursor
  // behind is how a dead drag looks like a stuck page.
  board.addEventListener("pointercancel", () => {
    dragging = false;
    moved = 0;
    board.style.cursor = hovering ? "pointer" : "grab";
  });

  board.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = board.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const before = toWorld(mx, my);
      // ctrl+wheel IS the pinch gesture on a trackpad -- the platform
      // reports a pinch as a wheel event with ctrlKey set
      // (../refs/mdn/content/files/en-us/web/api/element/wheel_event/index.md:17)
      // -- so it zooms harder per unit. A plain wheel zooms too, because
      // this surface has nothing else for a wheel to do.
      const by = Math.exp(-pixels(event) * (event.ctrlKey ? 0.01 : 0.0022));
      // The stops. Out is FIT -- there is nothing beyond the whole answer
      // worth showing, and letting it shrink past that just puts the
      // library in a corner. In is one picture at a comfortable size;
      // past 14 a thumbnail is only its own compression.
      cam.k = clamp(cam.k * by, fitScale(), 14);
      const after = toWorld(mx, my);
      cam.x += before.x - after.x;
      cam.y += before.y - after.y;
      anchor();
      flight = null;
      settle();
      refocus();
      draw();
    },
    { passive: false },
  );

  // The layout switch and the way back out.
  /** Which of the three arrangements is on screen. The attribute lives
   *  on the grid root because the stylesheet has to hide the OTHER two
   *  from there -- the canvas and the rows are siblings. */
  const shell = stage.parentElement;

  function arrange(wanted: "rank" | "time" | "grid" | "board", button: Element | null): void {
    for (const other of stage.querySelectorAll("[data-field-mode]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    if (shell) shell.dataset.arrangement = wanted;
    // `grid` means "show the rows instead", which only exists where
    // there are rows. On the canvas-only page there is nothing to
    // show instead, and the button is not rendered.
    if (wanted === "grid") return;
    if (wanted !== mode) {
      for (const n of nodes) n.from = { ...n.box };
      mode = wanted;
      // The board is a different set of things, not a rearrangement of
      // the same ones, so it is read rather than morphed into.
      if (mode === "board") loadBoard();
      // The chip describes what is on screen, and on the board that is
      // the board -- not the window of pictures behind it, which is a
      // count of something the reader is not looking at.
      if (count) {
        if (mode === "board") {
          const held = pinned().length;
          count.textContent = held ? `${held} on the board` : "nothing on the board yet";
          count.hidden = false;
        } else if (covering) {
          count.textContent = cut
            ? `${nodes.length.toLocaleString()} of the pictures here — zoom in for the rest`
            : `${nodes.length.toLocaleString()} of ${total.toLocaleString()}`;
        }
      }
      measure();
      morphAt = mode === "board" ? 0 : performance.now();
      stage.dataset.fieldMode = mode;
    }
    // A canvas sized while it was display:none measured zero, so the
    // first paint after coming back has to re-read the box.
    resize();
    fit();
    tick();
  }

  stage.addEventListener("click", (event) => {
    const button = closestFrom(event.target, "[data-field-mode]", HTMLButtonElement);
    if (button) {
      const said = button.dataset.fieldMode;
      arrange(said === "time" ? "time" : said === "grid" ? "grid" : said === "board" ? "board" : "rank", button);
      return;
    }
    if (closestFrom(event.target, "[data-field-fit]", HTMLButtonElement)) fit();

    // Keep this question. The name is what the person typed where they
    // typed one, and the address is the question itself -- so the card
    // re-answers rather than remembering an answer.
    if (closestFrom(event.target, "[data-field-pin]", HTMLButtonElement)) {
      const asked = question().toString();
      const said = new URLSearchParams(asked).get("q");
      const already = pinned().find((one) => one.at === asked);
      if (already) {
        unpin(already.id);
        if (mode === "board") {
          loadBoard();
          measure();
          draw();
          tally();
        } else say("taken off the board");
        return;
      }
      // Somewhere free: down the diagonal, so a new card never lands
      // exactly on one already there.
      const held = pinned();
      keep({
        id: `pin-${Date.now().toString(36)}`,
        kind: "query",
        name: said || "The whole library",
        at: asked,
        x: (held.length % 4) * (CARD_W + 40),
        y: Math.floor(held.length / 4) * (CARD_H + 40),
      });
      if (mode === "board") {
        loadBoard();
        measure();
        draw();
        tally();
      } else say("on the board");
    }
  });

  /** What is on the board, in the chip. Written whenever the board
   *  changes, rather than through `say`, whose whole job is to put back
   *  what was there before -- which after a change is the old count. */
  function tally(): void {
    if (!count) return;
    const held = pinned().length;
    count.textContent = held ? `${held} on the board` : "nothing on the board yet";
    count.hidden = false;
  }

  /** A word about what just happened, where the controls are. */
  function say(what: string): void {
    if (!count) return;
    const was = count.textContent;
    const hidden = count.hidden;
    count.textContent = what;
    count.hidden = false;
    window.setTimeout(() => {
      count.textContent = was;
      count.hidden = hidden;
    }, 1800);
  }

  // Through the registry, never a listener of its own. One document
  // keydown per module is how this application ended up with two
  // keyboards that could not see each other's claims; `register` refuses
  // a collision at registration, which is the only moment it can be
  // caught, and the shortcuts panel is built from what it holds -- so a
  // key that works is a key a person can find.
  //
  // Escape and Enter are deliberately NOT claimed. They belong to
  // whatever is on top -- a dialog, the viewer overlay -- and a field
  // that took them would be taking them from the surface that needs them
  // more. Clicking the ground already steps back out.
  const zoomed = (by: number): void => flyTo({ ...cam, k: clamp(cam.k * by, fitScale(), 14) }, 240);
  //
  // NOT `0`, which reads as "zoom to 100%" everywhere else and is taken
  // here: the digits are the star ratings (`authored: clear rating`).
  // The registry refused it out loud rather than letting two surfaces
  // answer to one key, which is exactly what it is for -- and is why
  // this is `z` instead.
  register([
    { key: "z", by: "the field: show all of them", run: () => fit() },
    { key: "+", by: "the field: closer", run: () => zoomed(1.6) },
    { key: "=", by: "the field: closer", run: () => zoomed(1.6) },
    { key: "-", by: "the field: further back", run: () => zoomed(1 / 1.6) },
  ]);

  // The grid grows as somebody keeps going; the field grows with it.
  // Only where there IS a grid.
  if (cells) {
    new MutationObserver(() => {
      ingest();
      draw();
    }).observe(cells, { childList: true, subtree: false });
  }

  new ResizeObserver(() => resize()).observe(board);

  // A canvas is sized in REAL pixels, and how many of those one CSS
  // pixel is worth changes without the element changing size at all --
  // the page is zoomed, or the window is dragged to a second monitor
  // with a different density
  // (../refs/mdn/content/files/en-us/web/api/window/devicepixelratio/index.md:16-22).
  // The ResizeObserver above sees neither, so the canvas would keep its
  // old backing store and go soft. This is the platform's own idiom for
  // watching it: a media query on the CURRENT ratio, which stops
  // matching the moment the ratio moves, and is rebuilt each time
  // (same file, lines 122-138).
  let watching: MediaQueryList | null = null;
  const density = (): void => {
    watching?.removeEventListener("change", density);
    watching = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    watching.addEventListener("change", density);
    resize();
  };
  density();

  // Revealed only now: the markup ships it hidden so a reader without
  // this bundle keeps the rows and never sees an empty box where a
  // canvas was going to be.
  stage.hidden = false;
  if (shell) shell.dataset.arrangement = mode;
  ingest();
  stage.dataset.fieldMode = mode;
  resize();
  fit(false);
  // The page's own cells are drawn first and are a true answer, just a
  // short one. This replaces them with the whole answer's shape and the
  // pictures for the stretch being looked at; if it fails, the page of
  // cells stays, which beats an empty field.
  void fetchAnswer();
  if (clock) clock.hidden = true;
}
