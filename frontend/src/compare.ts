/**
 * The compare tray: what I have kept, until I say otherwise.
 *
 * Looking at two pictures side by side is the ordinary thing somebody
 * does with a library of near-identical generations, and the application
 * had no way to do it at all. Two tabs was the workaround.
 *
 * The tray is deliberately NOT a page. It is a strip along the bottom
 * that collapses to a tab and comes back where it was, because the point
 * is to gather things while carrying on browsing -- a picture from the
 * gallery, another from a person's profile, a third from an analysis --
 * and a comparison surface you have to navigate TO cannot be filled that
 * way.
 *
 * Three things this gets right that a naive version does not.
 *
 * IT KEEPS THEM UNTIL DISMISSED. Not until the next page, not until the
 * tab closes: the tray is workspace state, so it survives navigation and
 * reload and empties when somebody empties it. Anything less makes it
 * useless for the case it exists for, which is gathering across surfaces.
 *
 * IT NEEDS NO SERVER. A kept picture is a slug and the name the adding
 * surface already had on screen; `/thumb/<slug>` addresses the picture
 * directly. So the tray costs no round trip, works on every surface that
 * shows media without any of them knowing about it, and cannot get out
 * of step with an endpoint.
 *
 * IT IS ORDERED, AND THE ORDER IS THE COMPARISON. Dragging a thumbnail
 * moves it; the comparison shows them left to right in exactly that
 * order. "Swap these two" is a drag rather than a mode.
 *
 * And it shows them two ways, because they answer different questions.
 * SIDE BY SIDE answers "how do these differ", and you read it by moving
 * your eyes. FLIP answers "did this change", and you read it by NOT
 * moving them: the pictures occupy the same pixels, so a small
 * difference that side-by-side loses in the saccade is the only thing
 * on screen that moves. Two generations of one prompt at different
 * steps is the case the second exists for.
 */

import { everyElement, findElement } from "./dom";
import { register } from "./keys";
import { remember, workspace } from "./workspace";

/** One piece of media, kept. */
export interface Kept {
  slug: string;
  name: string;
  /**
   * Where its picture lives, as the surface that kept it was told.
   *
   * Content-addressed and immutable when the bytes have been hashed
   * (vision/thumbs.py `asset_url`), so the tray costs no application
   * request at all. Absent for a surface that does not carry one yet,
   * where the slug route still answers.
   */
  thumb?: string;
  /**
   * What it is, when the surface that kept it said so.
   *
   * Optional because not every grid in this application spells it, and
   * the fallback is honest: `/preview/<slug>` is a still for EVERY kind
   * -- it is what the viewer uses as a video's poster -- so a kept clip
   * with no stated kind still shows the right picture. Knowing the kind
   * only buys the comparison a playable element instead of a frame.
   */
  kind?: string;
}

/**
 * How many the tray holds.
 *
 * A bound, not a target: past this the thumbnails are too small to
 * choose between, which defeats the tray. Adding to a full tray drops
 * the OLDEST, because the alternative is refusing a click that looks
 * exactly like every other click that worked.
 */
export const MOST = 8;

export function kept(): Kept[] {
  const held = workspace().compare;
  return Array.isArray(held) ? held.filter((one) => one && typeof one.slug === "string") : [];
}

function keep(held: Kept[]): void {
  remember({ compare: held.slice(-MOST) });
}

/**
 * The picture this surface is currently about.
 *
 * In order, because a surface can be several things at once: an open
 * lightbox is what somebody is looking at even though the grid behind it
 * still exists; a viewer page is its own subject; and on a grid it is
 * whatever the pointer or the keyboard is on. Falling through to "the
 * first cell" would keep a picture nobody indicated.
 */
function current(root: HTMLElement): Kept | null {
  const lightbox = findElement(root, "[data-lightbox][data-slug]", HTMLElement);
  if (lightbox) {
    const named = findElement(lightbox, "[data-viewer][data-slug]", HTMLElement) ?? lightbox;
    const slug = named.dataset.slug ?? lightbox.dataset.slug;
    if (slug) return { slug, name: named.dataset.name ?? slug, kind: named.dataset.kind ?? "" };
  }
  const viewer = findElement(root, "[data-viewer][data-slug]", HTMLElement);
  if (viewer?.dataset.slug) {
    return {
      slug: viewer.dataset.slug,
      name: viewer.dataset.name ?? viewer.dataset.slug,
      kind: viewer.dataset.kind ?? "",
    };
  }

  // A grid: whatever is under the pointer, else whatever has focus.
  const under = root.querySelector("a.cell[data-slug]:hover");
  const focused = document.activeElement?.closest?.("a.cell[data-slug]") ?? null;
  const cell = (under ?? focused) as HTMLElement | null;
  if (cell?.dataset.slug) {
    const shown = cell.querySelector("img");
    return {
      slug: cell.dataset.slug,
      name: shown?.getAttribute("alt") || cell.dataset.slug,
      kind: cell.dataset.kind ?? "",
      thumb: shown?.getAttribute("src") ?? "",
    };
  }
  return null;
}

/**
 * The element that shows one kept thing at comparison size.
 *
 * NOT the grid thumbnail: a comparison is a claim about what two pieces
 * of media look like, and 88 pixels of one of them is not that claim.
 * Not the original either, for a still -- `/preview` is the sized one
 * the viewer itself opens with, and pulling four 40-megapixel originals
 * to lay them side by side is a slideshow of loading spinners.
 *
 * A clip is a clip. Comparing two generations of a video by looking at
 * two frozen frames is the failure this whole surface exists to avoid,
 * so a kind that moves gets an element that plays.
 */
function playable(one: Kept): HTMLElement {
  if (one.kind === "video" || one.kind === "animated_image") {
    const clip = document.createElement("video");
    clip.src = `/media/${one.slug}`;
    clip.poster = `/preview/${one.slug}`;
    clip.controls = true;
    clip.loop = true;
    clip.playsInline = true;
    clip.setAttribute("aria-label", one.name);
    return clip;
  }
  if (one.kind === "audio") {
    const sound = document.createElement("audio");
    sound.src = `/media/${one.slug}`;
    sound.controls = true;
    sound.setAttribute("aria-label", one.name);
    return sound;
  }
  const shown = document.createElement("img");
  shown.src = `/preview/${one.slug}`;
  shown.alt = one.name;
  return shown;
}

// --- the comparison itself --------------------------------------------------

/** A, B, C … -- what a kept picture is called while comparing. */
function letter(at: number): string {
  return String.fromCharCode(65 + (at % 26));
}

/**
 * Everything kept, above everything else -- side by side, or one at a
 * time in the same place.
 *
 * `object-fit: contain` per column and nothing else: the whole value of
 * a comparison is that the pictures are shown the same way, so nothing
 * here crops, scales one to another, or picks a "primary".
 *
 * EVERY column is built in both modes, and flipping changes which one
 * is shown -- never the `src` of one element. Swapping a source makes
 * the browser fetch and decode on the flip, and a flip you can watch
 * happen is worse than useless: the delay is the only thing your eye
 * reports, and it is not a fact about the pictures. Built once, they
 * are decoded once and the change is a repaint.
 */
function showComparison(held: Kept[]): void {
  const old = document.querySelector("[data-compare-view]");
  if (old) old.remove();
  if (held.length < 2) return;

  const sheet = document.createElement("div");
  sheet.className = "compare-view";
  sheet.dataset.compareView = "";
  sheet.setAttribute("role", "dialog");
  sheet.setAttribute("aria-label", "comparing");

  const bar = document.createElement("header");
  bar.className = "compare-view-bar";
  const said = document.createElement("span");
  said.className = "compare-view-said";
  const close = document.createElement("button");
  close.type = "button";
  close.className = "compare-view-close";
  close.dataset.compareViewClose = "";
  close.setAttribute("aria-label", "stop comparing");
  close.textContent = "×";
  // How to show them. A choice and not a better default: the two modes
  // answer different questions, and which one somebody wants is a fact
  // about what they are looking for.
  const modes = document.createElement("div");
  modes.className = "compare-modes";
  modes.setAttribute("role", "group");
  modes.setAttribute("aria-label", "how to compare");
  bar.append(said, modes, close);

  const strip = document.createElement("div");
  strip.className = "compare-view-strip";
  for (const [at, one] of held.entries()) {
    const column = document.createElement("figure");
    column.className = "compare-column";
    column.dataset.compareColumn = one.slug;
    column.dataset.at = String(at);
    column.dataset.letter = letter(at);
    // The media sits in a FRAME rather than directly in the column.
    // The frame is what takes the column's share of the height, so
    // every column is the same height and the pictures line up; the
    // media is centred inside its own frame at its own shape, so a
    // clip's controls stay attached to the clip instead of stranded
    // three hundred pixels below it.
    const frame = document.createElement("div");
    frame.className = "compare-frame";
    const shown = playable(one);
    frame.append(shown);
    const label = document.createElement("figcaption");
    // The letter is how a person says which one they mean -- "B is
    // sharper" -- and it is the same letter in both modes, so switching
    // does not renumber what somebody was just talking about.
    const named = document.createElement("b");
    named.className = "compare-letter";
    named.textContent = letter(at);
    const link = document.createElement("a");
    link.href = `/i/${one.slug}`;
    link.textContent = one.name;
    label.append(named, link);
    column.append(frame, label);
    strip.append(column);
  }
  sheet.append(bar, strip);
  document.body.append(sheet);

  // --- the two ways of looking -------------------------------------------
  let mode: "side" | "flip" = workspace().compareMode === "flip" ? "flip" : "side";
  let at = 0;

  const columns = () => everyElement(strip, "[data-compare-column]", HTMLElement);

  const paint = () => {
    sheet.dataset.mode = mode;
    const all = columns();
    at = ((at % all.length) + all.length) % all.length;
    for (const [index, column] of all.entries()) {
      // `hidden`, not display:none in a class: the element stays built
      // and decoded either way, which is the whole reason a flip is
      // instant. A video also keeps playing rather than restarting.
      column.hidden = mode === "flip" && index !== at;
      column.dataset.showing = String(mode === "side" || index === at);
    }
    const one = held[at];
    said.textContent =
      mode === "side" ? `${held.length} side by side` : `${letter(at)} of ${held.length} · ${one ? one.name : ""}`;
    for (const button of everyElement(modes, "[data-compare-mode]", HTMLElement)) {
      button.setAttribute("aria-pressed", String(button.dataset.compareMode === mode));
    }
  };

  for (const [name, words, why] of [
    ["side", "side by side", "every one at once: how do these differ"],
    ["flip", "flip", "one at a time in the same place: did this change"],
  ] as const) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "compare-mode";
    button.dataset.compareMode = name;
    button.title = why;
    button.textContent = words;
    button.addEventListener("click", () => {
      mode = name;
      remember({ compareMode: name });
      paint();
    });
    modes.append(button);
  }

  /** Move to the next kept picture, and switch to flip if it is not on.
   *
   * Stepping IS flipping, so a person who presses the key without
   * having found the mode button gets what they were reaching for.
   */
  const step = (by: number) => {
    at += by;
    if (mode !== "flip") {
      mode = "flip";
      remember({ compareMode: "flip" });
    }
    paint();
  };

  paint();

  const dismiss = () => sheet.remove();
  close.addEventListener("click", dismiss);
  sheet.addEventListener("click", (event) => {
    if (event.target === sheet) dismiss();
  });

  // Escape on the SHEET, not on the document, and not through the key
  // registry either.
  //
  // Not the document, because one keystroke has one meaning here and
  // that is decided in keys.ts (sglint SG503 says so). Not the registry,
  // because Escape is already claimed by whichever surface is underneath
  // -- the overlay, or the media page -- and the registry refuses a
  // second claim by design. So the sheet takes focus when it opens and
  // hears its own keys: while it has focus it IS the outermost thing,
  // and when it closes there is nothing to unregister.
  sheet.tabIndex = -1;
  sheet.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      dismiss();
      return;
    }
    // Space is the flip. It is what a hand rests on, and flipping is
    // the one thing somebody does over and over here -- the value of
    // the mode is in the repetition, and a two-key reach kills it.
    if (event.key === " " || event.key === "f" || event.key === "F") {
      event.preventDefault();
      step(1);
      return;
    }
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      step(1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      step(-1);
    }
  });
  sheet.focus();
}

// --- the tray ---------------------------------------------------------------

function drawTray(tray: HTMLElement): void {
  const held = kept();
  const open = workspace().tray !== "closed";
  tray.hidden = held.length === 0;
  tray.dataset.tray = open ? "open" : "closed";

  const count = findElement(tray, "[data-compare-count]", HTMLElement);
  if (count) count.textContent = String(held.length);

  const compare = findElement(tray, "[data-compare-open]", HTMLButtonElement);
  // Two is the smallest number of things that can be compared, so below
  // it the control is present and inert rather than absent and confusing.
  if (compare) compare.disabled = held.length < 2;

  const list = findElement(tray, "[data-compare-items]", HTMLElement);
  if (!list) return;
  list.replaceChildren();
  for (const [at, one] of held.entries()) {
    const item = document.createElement("li");
    item.className = "tray-item";
    item.draggable = true;
    item.dataset.compareSlug = one.slug;
    item.dataset.at = String(at);

    const shown = document.createElement("img");
    // The surface that kept it knew where its picture lives; keeping
    // the URL rather than the slug means the tray never asks the
    // application to work it out again.
    shown.src = one.thumb ?? `/thumb/${one.slug}`;
    shown.alt = one.name;
    shown.title = one.name;

    const drop = document.createElement("button");
    drop.type = "button";
    drop.className = "tray-drop";
    drop.dataset.compareRemove = one.slug;
    drop.setAttribute("aria-label", `stop keeping ${one.name}`);
    drop.textContent = "×";

    item.append(shown, drop);
    list.append(item);
  }

  for (const drop of everyElement(list, "[data-compare-remove]", HTMLElement)) {
    drop.addEventListener("click", () => {
      keep(kept().filter((one) => one.slug !== drop.dataset.compareRemove));
      drawTray(tray);
    });
  }

  // Dragging is the reorder, and the order IS the comparison, so
  // "swap these two" is a drag rather than a mode with its own buttons.
  let from: number | null = null;
  for (const item of everyElement(list, "[data-compare-slug]", HTMLElement)) {
    item.addEventListener("dragstart", (event) => {
      from = Number(item.dataset.at);
      event.dataTransfer?.setData("text/plain", item.dataset.compareSlug ?? "");
      item.dataset.dragging = "true";
    });
    item.addEventListener("dragend", () => {
      delete item.dataset.dragging;
    });
    item.addEventListener("dragover", (event) => event.preventDefault());
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      const to = Number(item.dataset.at);
      if (from === null || from === to) return;
      const order = kept();
      const [moved] = order.splice(from, 1);
      if (moved) order.splice(to, 0, moved);
      keep(order);
      from = null;
      drawTray(tray);
    });
  }
}

function build(): HTMLElement {
  const tray = document.createElement("aside");
  tray.className = "tray";
  tray.dataset.compareTray = "";
  tray.hidden = true;
  tray.setAttribute("aria-label", "kept to compare");
  tray.innerHTML = [
    '<header class="tray-bar">',
    '<button type="button" class="tray-tab" data-compare-collapse aria-label="show or hide what is kept">',
    "kept <b data-compare-count>0</b>",
    "</button>",
    '<button type="button" class="tray-act" data-compare-open>compare</button>',
    '<button type="button" class="tray-act" data-compare-clear>clear</button>',
    "</header>",
    '<ol class="tray-items" data-compare-items></ol>',
  ].join("");
  return tray;
}

/**
 * Put the tray on this surface.
 *
 * Called from every entry point that shows media. The tray builds its
 * own markup rather than living in a template, because "every surface
 * that shows a picture" is nine templates and a rule nobody would keep.
 */
export function mountCompare(root: HTMLElement): void {
  if (document.querySelector("[data-compare-tray]")) return;
  const tray = build();
  document.body.append(tray);

  const collapse = findElement(tray, "[data-compare-collapse]", HTMLElement);
  if (collapse) {
    collapse.addEventListener("click", () => {
      remember({ tray: workspace().tray === "closed" ? "open" : "closed" });
      drawTray(tray);
    });
  }
  const open = findElement(tray, "[data-compare-open]", HTMLElement);
  if (open) open.addEventListener("click", () => showComparison(kept()));
  const clear = findElement(tray, "[data-compare-clear]", HTMLElement);
  if (clear) {
    clear.addEventListener("click", () => {
      keep([]);
      drawTray(tray);
    });
  }

  const add = () => {
    const one = current(root);
    if (!one) return;
    const held = kept();
    // Pressing it twice on one picture takes it back off, so the key is
    // its own undo and nobody has to find the × for a mistake.
    keep(held.some((each) => each.slug === one.slug) ? held.filter((each) => each.slug !== one.slug) : [...held, one]);
    remember({ tray: "open" });
    drawTray(tray);
  };
  register([{ key: "c", by: "compare: keep this", run: add }]);

  drawTray(tray);
}
