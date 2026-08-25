// The photograph, and the few facts that change what it means.
//
// ONE viewer, mounted two ways. The page and the gallery overlay are
// container adapters -- they differ in what surrounds the stage, never in
// what the stage does -- so there is no second implementation of zoom, pan,
// quality or the inspector to keep synchronously disappointing.
//
// There is no viewer MODE. There are independent axes, and the shapes a
// design sketch would call "minimal", "sidebar", "focus" or "bottom sheet"
// are combinations of them the viewer arrives at by itself:
//
//   chrome      visible -> auto-hidden while the pointer rests, back on the
//               next movement -> put out by L, which a movement does NOT
//               undo, so one dim x stays behind as the way back
//   inspector   closed | open (I). WHERE it sits is geometry's business:
//               CSS docks it beside a wide stage and sheets it under a
//               narrow one, so nothing here knows which
//   zoom/pan    fit | fill | 1:1 | anywhere between, with pan
//   quality     preview -> original, decided by arithmetic, never by a button
//
// The walk recedes while zoomed in, because pixels matter more than
// neighbours once somebody is inspecting them.
//
// Nothing here is persisted. Zoom, pan, which panel is open and whether the
// chrome is hidden are what the person is doing RIGHT NOW; the moment they
// become a setting is the moment somebody has to be asked a question about
// a photograph they were looking at.
import { closestFrom, everyElement, findElement, isPlainClick, requireData } from "./dom";
import type { components } from "./generated/api";
import { register } from "./keys";
import { mountRecipe } from "./recipe";
import { panelState, remember, rememberPanel, workspace } from "./workspace";

type Stage = components["schemas"]["MediaSurface"]["stage"];
type Pixels = components["schemas"]["Pixels"];

/** How the stage is presently framed. `fit` is where every picture opens. */
export type Framing = "fit" | "fill" | "actual" | "free";

/** A size in CSS pixels, as the stylesheet laid it out. */
interface Size {
  width: number;
  height: number;
}

/** One picture's viewing state. Ephemeral by construction. */
interface Look {
  framing: Framing;
  /** multiplier on the fitted size: 1 IS fit, whatever the picture's shape */
  scale: number;
  /** the stage's translation in CSS pixels, applied before the scale */
  x: number;
  y: number;
}

const FIT: Look = { framing: "fit", scale: 1, x: 0, y: 0 };

/** Whether this stage is a still picture -- the one kind that zooms. */
function isStill(stage: Stage): stage is components["schemas"]["ImageStage"] {
  return stage.kind === "image";
}

/**
 * The scale at which one source pixel covers one DEVICE pixel.
 *
 * Not one CSS pixel: on a 2x screen a CSS pixel is two device pixels, and
 * "actual pixels" that quietly showed the picture at half its resolution
 * would be a lie the whole promotion machinery exists to avoid.
 */
function actualScale(source: Pixels, fitted: Size): number {
  const wanted = source.width / (window.devicePixelRatio || 1);
  return fitted.width > 0 ? wanted / fitted.width : 1;
}

/** Cover: the smaller dimension fills the stage, the larger overflows. */
function fillScale(fitted: Size, box: Size): number {
  if (fitted.width <= 0 || fitted.height <= 0) return 1;
  return Math.max(box.width / fitted.width, box.height / fitted.height);
}

export interface Viewer {
  /** Whether Escape found viewer state to unwind. False means "dismiss me". */
  unwind(): boolean;
  /** Release listeners bound outside the root, for a remounted overlay. */
  release(): void;
}

/**
 * What a container does with a step along the walk.
 *
 * The ARROWS are the viewer's, because walking is what somebody looking
 * at a picture does; where the next one is rendered is the container's,
 * because that is the whole of the difference between them. The overlay
 * replaces its mount so fifty pictures are one Back; the page navigates,
 * because a page is what it is.
 */
export type Walk = (href: string) => void;

const MIN_SCALE = 1;
const MAX_SCALE = 40;
/** How long the pointer must rest before the chrome fades. */
const IDLE_MS = 2200;
/** Seconds a slideshow shows each picture until somebody says otherwise. */
const DEFAULT_EVERY = 5;
/** Past this, the surrounding walk is not what anybody is looking at. */
const ABSORBED = 1.35;

export function mountViewer(root: HTMLElement, walk: Walk): Viewer | null {
  const stageBox = findElement(root, "[data-stage]", HTMLElement);
  if (!stageBox) return null;

  const stage: Stage = JSON.parse(requireData(stageBox, "stage"));
  const media = findElement(stageBox, "[data-stage-media]", HTMLElement);
  const still = isStill(stage) ? stage : null;

  let look: Look = { ...FIT };
  let promoted = false;
  let idle = 0;
  const bound: Array<() => void> = [];

  // Two helpers rather than one over EventTarget: `addEventListener` is
  // typed per interface, so the DOM hands each listener its real event and
  // nothing needs asserting. A single EventTarget-shaped helper would give
  // every listener a bare `Event` and put an `as` at each call site.
  const onElement = <K extends keyof HTMLElementEventMap>(
    target: HTMLElement,
    type: K,
    listener: (event: HTMLElementEventMap[K]) => void,
    options?: AddEventListenerOptions,
  ) => {
    target.addEventListener(type, listener, options);
    bound.push(() => target.removeEventListener(type, listener));
  };

  const onDocument = <K extends keyof DocumentEventMap>(type: K, listener: (event: DocumentEventMap[K]) => void) => {
    document.addEventListener(type, listener);
    bound.push(() => document.removeEventListener(type, listener));
  };

  // --- what the picture measures ------------------------------------------
  // The fitted size is what CSS laid the picture out at, divided back out of
  // whatever scale is currently applied -- never recomputed from the source's
  // aspect against the stage. The stage's size is the stylesheet's business
  // and changes with the inspector and the window; deriving it
  // here would be a second layout engine, free to disagree with the real one.
  const fitted = (): { width: number; height: number } => {
    const rect = (media ?? stageBox).getBoundingClientRect();
    return { width: rect.width / look.scale, height: rect.height / look.scale };
  };

  /**
   * Pan far enough to see every part of a zoomed picture, and no further.
   *
   * Untethered, `x`/`y` were accumulated pointer deltas: one long drag
   * flung the photograph off the screen and left an empty stage with no
   * way back except Escape. The bound is the OVERHANG -- how far the
   * scaled picture exceeds the stage on each axis -- so a picture that
   * fits on an axis stays centred there and one that overflows can be
   * pushed exactly to its own edge.
   */
  const tethered = (x: number, y: number, scale: number): { x: number; y: number } => {
    const size = fitted();
    const box = stageBox.getBoundingClientRect();
    const room = (picture: number, stage: number) => Math.max(0, (picture * scale - stage) / 2);
    const across = room(size.width, box.width);
    const down = room(size.height, box.height);
    return {
      x: Math.min(across, Math.max(-across, x)),
      y: Math.min(down, Math.max(-down, y)),
    };
  };

  /** Whether any part of the picture is off the stage's edge. */
  const canPan = (): boolean => {
    const size = fitted();
    const box = stageBox.getBoundingClientRect();
    return size.width * look.scale > box.width + 1 || size.height * look.scale > box.height + 1;
  };

  /**
   * Whether the picture is currently bigger than the stage fits it at.
   *
   * The question every toggle should ask, because it is about the state
   * and not about which of four words the state was last given.
   */
  const zoomedIn = (): boolean => look.scale > 1 || canPan();

  const paint = () => {
    if (!media) return;
    media.style.transform = `translate(${look.x}px, ${look.y}px) scale(${look.scale})`;
    stageBox.dataset.framing = look.framing;
    // The walk recedes once the picture is being inspected rather than
    // browsed. A class would hide the state; the attribute IS the state.
    root.dataset.absorbed = look.scale > ABSORBED ? "true" : "false";
    root.dataset.zoom = String(Math.round(look.scale * 100));
    // A picture being examined belongs ON TOP of the furniture around
    // it. Fitted, the stage sits in the layout like any other panel;
    // zoomed, it is the thing being done, and the chrome, the filmstrip
    // and the inspector are in its way. The attribute IS the state and
    // the stylesheet decides what rising looks like.
    root.dataset.magnified = zoomedIn() ? "true" : "false";
  };

  // --- quality --------------------------------------------------------------
  /**
   * Promote the preview to the original once the stage is asking for more
   * device pixels than the preview holds.
   *
   * `promotable` is the server's answer to "does the original have more to
   * give", and it has to be, because the preview is `ImageOps.contain`d to a
   * 1440 box that ENLARGES a smaller source: a 400px picture is served as a
   * 1440px preview, and a browser comparing its own naturalWidth would
   * promote to something four times smaller. The decode happens off-screen
   * and the swap keeps the transform, so nothing jumps.
   */
  const promote = () => {
    if (!still || promoted || !still.promotable || !still.shown) return;
    const wanted = fitted().width * look.scale * (window.devicePixelRatio || 1);
    if (wanted <= still.shown.width) return;
    promoted = true;
    const full = new Image();
    full.src = still.original;
    void full
      .decode()
      .then(() => {
        const img = findElement(stageBox, "img[data-stage-media]", HTMLImageElement);
        if (img) img.src = still.original;
        stageBox.dataset.quality = "original";
      })
      .catch(() => {
        // the original is unreachable; the preview is still a picture
        promoted = false;
      });
  };

  // --- framing --------------------------------------------------------------
  const clamp = (scale: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));

  const frame = (framing: Framing) => {
    if (!still) return;
    const box = stageBox.getBoundingClientRect();
    const size = fitted();
    const scale =
      framing === "fit"
        ? 1
        : framing === "fill"
          ? fillScale(size, box)
          : still.source
            ? actualScale(still.source, size)
            : 1;
    look = { framing, scale: clamp(scale), x: 0, y: 0 };
    paint();
    promote();
  };

  /** Zoom about a point, so what is under the pointer stays under it. */
  const zoomAbout = (factor: number, clientX: number, clientY: number) => {
    if (!still) return;
    const box = stageBox.getBoundingClientRect();
    const next = clamp(look.scale * factor);
    if (next === look.scale) return;
    // the pointer, relative to the stage's centre -- which is what the
    // transform's origin is
    const px = clientX - (box.left + box.width / 2);
    const py = clientY - (box.top + box.height / 2);
    const ratio = next / look.scale;
    const held = tethered(px - (px - look.x) * ratio, py - (py - look.y) * ratio, next);
    look = { framing: "free", scale: next, ...held };
    paint();
    promote();
  };

  /**
   * Put the framing back where it says it is, after the stage changed size.
   *
   * `fill` and `actual` are scales computed ONCE from the fitted size, and
   * the fitted size moves whenever the stage does -- the inspector opening,
   * a window resized, a phone turned, the narrow layout swapping the
   * inspector from a column to a sheet. Left alone, a picture still
   * labelled "actual" quietly stopped being 1:1, which is the one thing
   * that label promises. `fit` is the browser's own doing, so it needs
   * only its offsets cleared; `free` keeps its scale and is re-tethered,
   * so a resize cannot strand a pan outside the new bounds.
   */
  const resettle = () => {
    if (!still) return;
    if (look.framing === "fit" || look.framing === "fill" || look.framing === "actual") {
      frame(look.framing);
      return;
    }
    look = { ...look, ...tethered(look.x, look.y, look.scale) };
    paint();
  };

  const watching = new ResizeObserver(() => resettle());
  watching.observe(stageBox);
  bound.push(() => watching.disconnect());

  // --- chrome ---------------------------------------------------------------
  const wake = () => {
    if (root.dataset.chrome === "focus") return; // F is a decision, not a mood
    root.dataset.chrome = "visible";
    window.clearTimeout(idle);
    idle = window.setTimeout(() => {
      if (root.dataset.chrome === "visible") root.dataset.chrome = "resting";
    }, IDLE_MS);
  };

  const focus = () => {
    window.clearTimeout(idle);
    root.dataset.chrome = root.dataset.chrome === "focus" ? "visible" : "focus";
    if (root.dataset.chrome === "visible") wake();
  };

  // --- inspector ------------------------------------------------------------
  // Open or closed is this file's business. Docked beside the stage or slid
  // up from the bottom is the stylesheet's, decided by the space there is.
  // the PANEL, not the root's data-inspector state attribute
  const inspector = findElement(root, "[data-inspector-panel]", HTMLElement);

  const showInspector = (open: boolean, arranged = true) => {
    if (!inspector) return;
    root.dataset.inspector = open ? "open" : "closed";
    for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
      button.setAttribute("aria-expanded", String(open));
    }
    // `arranged` separates a person's decision from the viewer restoring
    // one. Recording a restore would be harmless here and wrong in
    // principle: only what somebody chose is worth keeping, and a
    // surface that writes its own defaults back can never tell the two
    // apart afterwards.
    if (arranged) remember({ inspector: open ? "open" : "closed" });
  };

  /**
   * Reveal one named section.
   *
   * The sections are `<details>`, so opening and closing them by hand,
   * by keyboard, or by assistive technology costs no JavaScript at all --
   * they were headings with `cursor: pointer` and no handler, which is a
   * control that looks like a control and is not one. This is only for the
   * chips that point INTO a section ("date disputed" -> technical); the
   * disclosure itself is the browser's.
   */
  const panel = (named: string) => {
    if (!inspector) return;
    for (const section of everyElement(inspector, "[data-panel]", HTMLDetailsElement)) {
      if (section.dataset.panel !== named) continue;
      section.open = true;
      section.scrollIntoView({ block: "nearest" });
    }
  };

  // --- the walk, on the wheel -----------------------------------------------

  /**
   * Whether Alt walks the library on this run.
   *
   * The run's answer, rendered onto the root (db/settings.py
   * `viewer_wheel_modifier`). Alt is the ONLY modifier that can mean this:
   * ctrl+wheel is how a browser delivers a trackpad pinch, and shift+wheel
   * is its horizontal scroll, so neither is available to mean "next
   * picture" without stealing a gesture that already means something.
   */
  const walksOnWheel = () => root.dataset.wheelModifier === "alt";

  /** One wheel notch, roughly, on the platforms that disagree about size. */
  const NOTCH = 90;
  /**
   * How long the wheel must be still before the NEXT gesture may step.
   *
   * A boundary, not a cooldown. A cooldown counts from the step, so an
   * inertial flick still arriving after it expires steps again and one
   * physical gesture walks two pictures; this counts from the last EVENT,
   * so a gesture's own inertia keeps its own boundary pushed out ahead of
   * it and lands nowhere.
   */
  const QUIET_MS = 260;

  /** What the wheel is presently doing, if anything. */
  let rolled = 0;
  let spent = false;
  let lastWheel = Number.NEGATIVE_INFINITY;

  /**
   * Step the walk once per gesture, however many events the gesture is.
   *
   * A wheel does not emit one event per notch: a hard flick or a trackpad
   * swipe is a stream of dozens, decaying over hundreds of milliseconds.
   * So a gesture is a run of events with no long silence in it -- the
   * first crossing of a notch's worth steps, and everything after it is
   * that same gesture's inertia and is swallowed. Only silence ends it.
   *
   * A reversal ends it too, and immediately: turning the wheel back is a
   * person correcting themselves, and making them wait out the inertia of
   * the flick they are undoing would feel broken.
   */
  const stepped = (by: number) => {
    if (by === 0) return;
    const now = performance.now();
    const reversed = rolled !== 0 && Math.sign(by) !== Math.sign(rolled);
    if (now - lastWheel > QUIET_MS || reversed) {
      rolled = 0;
      spent = false;
    }
    lastWheel = now;
    rolled += by;
    if (spent || Math.abs(rolled) < NOTCH) return;
    const step = stepTo(rolled > 0 ? "next" : "previous");
    // An end of the walk is an answer, not a thing to force -- and the
    // gesture is spent either way, so leaning on the wheel at the last
    // picture does not fire again the moment a next one exists.
    spent = true;
    if (step) walk(step.href);
  };

  // --- the pointer ----------------------------------------------------------

  // The wheel is bound for EVERY kind, not only the ones that zoom: a clip
  // and a sound file sit in the same walk as a photograph, and stepping
  // past them is the whole point of putting the walk on the wheel.
  onElement(
    stageBox,
    "wheel",
    (event) => {
      // The viewer cancels only what it ACTS on. It claims two gestures
      // over its stage and no others: a plain wheel, which zooms, and
      // Alt+wheel when the run asks for it, which walks. Anything else --
      // a trackpad pinch, which arrives as ctrl+wheel; a shifted wheel,
      // which is the browser's horizontal scroll -- is left alone and
      // reaches the browser, which is what db/settings.py promises when it
      // explains why those two are not offered as walk modifiers.
      //
      // Deciding BEFORE cancelling is the whole point: an unconditional
      // preventDefault at the top of this handler made that promise false
      // for every gesture the viewer then ignored.
      const plain = !event.altKey && !event.ctrlKey && !event.shiftKey && !event.metaKey;
      const walking = event.altKey && !event.ctrlKey && !event.shiftKey && !event.metaKey && walksOnWheel();
      if (!plain && !walking) return;
      event.preventDefault();

      // deltaMode 1 is lines, 2 is pages: a trackpad and a wheel must not
      // mean different amounts of picture
      const pixels = (delta: number) => (event.deltaMode === 0 ? delta : delta * 16);

      if (walking) {
        // some platforms move a modified wheel's amount into deltaX, so
        // both axes are read rather than either being trusted
        stepped(pixels(event.deltaY || event.deltaX));
        return;
      }
      // a plain wheel zooms -- which a stage with nothing to zoom declines
      zoomAbout(Math.exp(-pixels(event.deltaY) / 400), event.clientX, event.clientY);
    },
    // not passive: the page must not scroll out from under a zoom
    { passive: false },
  );

  if (still && media) {
    // Double-click goes IN from fit, and OUT from anywhere else.
    //
    // It used to ask `framing === "actual"`, which is the LABEL rather
    // than the state -- and panning sets the label to "free". So a
    // person who went to 1:1 and then dragged the picture (the whole
    // reason to be at 1:1) could double-click as often as they liked and
    // never get back to fit: it saw "free", decided that meant "not
    // zoomed", and re-applied `actual` on top of itself. The question
    // that matters is whether the picture is currently bigger than its
    // fitted size, and that is a number, not a name.
    onElement(stageBox, "dblclick", (event) => {
      event.preventDefault();
      frame(zoomedIn() ? "fit" : "actual");
    });

    let dragging: number | null = null;
    let from = { x: 0, y: 0, ox: 0, oy: 0 };

    onElement(stageBox, "pointerdown", (event) => {
      // Whenever there is something off the edge to drag TO, not merely
      // when the scale reads above 1. A picture at "actual" that is
      // wider than the stage but shorter than it has real overhang on
      // one axis at a scale below 1, and refusing the drag there left a
      // person looking at the middle of a panorama with no way to reach
      // its ends.
      if (event.button !== 0 || !canPan()) return;
      // Capture, so the pan owns the pointer until it is released. Without
      // it a drag that ends outside the picture lands on the overlay's
      // backdrop, whose click IS dismissal -- and panning would close the
      // viewer (frontend/src/overlay.ts: a click on the root is Back).
      stageBox.setPointerCapture(event.pointerId);
      dragging = event.pointerId;
      from = { x: event.clientX, y: event.clientY, ox: look.x, oy: look.y };
      stageBox.dataset.panning = "true";
    });

    onElement(stageBox, "pointermove", (event) => {
      if (dragging !== event.pointerId) return;
      const held = tethered(from.ox + (event.clientX - from.x), from.oy + (event.clientY - from.y), look.scale);
      look = { framing: "free", scale: look.scale, ...held };
      paint();
    });

    const release = (event: PointerEvent) => {
      if (dragging !== event.pointerId) return;
      dragging = null;
      delete stageBox.dataset.panning;
      if (stageBox.hasPointerCapture(event.pointerId)) stageBox.releasePointerCapture(event.pointerId);
    };
    onElement(stageBox, "pointerup", release);
    onElement(stageBox, "pointercancel", release);
  }

  // --- the keys -------------------------------------------------------------
  //
  // Registered rather than listened for, so a key this claims cannot also
  // mean something to another module in the same bundle (frontend/src/keys.ts).
  // The letters here are the ones LEFT: the authored strip had F, 1-5, 0 and
  // A long before this file existed, and a viewer that took F and 1 back was
  // rating photographs while somebody looked at them.
  //
  //   Z  fit <-> actual pixels     L  lights out
  //   I  information               + -  zoom about the middle
  //   arrows  walk
  //
  // There is no T. It toggled a filmstrip nothing renders -- a key that
  // advertises a feature the viewer does not have is fake UI, and this
  // repo spends its effort deleting those. It comes back with the strip.
  /**
   * The anchor for a step, which is `[data-nav]` and nothing cleverer.
   *
   * `data-nav` means "a step you can take right now" -- to this file, to
   * gallery.ts's overlay intercept, and to every browser test. Wrapping
   * does not weaken that: the server spells the wrapping arrow
   * `data-nav-wrap`, and it becomes a step only where this person turned
   * wrapping on, by the promotion below. A hidden anchor that still said
   * `data-nav` would be the DOM promising a step nothing could take.
   */
  const stepTo = (wanted: "next" | "previous"): HTMLAnchorElement | null =>
    findElement(root, `[data-nav="${wanted}"]`, HTMLAnchorElement);

  const stepping = (wanted: "next" | "previous") => () => {
    const step = stepTo(wanted);
    if (step) walk(step.href); // an end of the walk is an answer, not a step
  };
  const middle = (by: number) => () => zoomAbout(by, window.innerWidth / 2, window.innerHeight / 2);
  bound.push(
    register([
      { key: "z", by: "viewer: fit/actual", run: () => frame(zoomedIn() ? "fit" : "actual") },
      { key: "l", by: "viewer: focus", run: focus },
      { key: "i", by: "viewer: inspector", run: () => showInspector(root.dataset.inspector !== "open") },
      { key: "+", by: "viewer: zoom in", run: middle(1.3) },
      { key: "=", by: "viewer: zoom in", run: middle(1.3) },
      { key: "-", by: "viewer: zoom out", run: middle(1 / 1.3) },
      { key: "ArrowRight", by: "viewer: next", run: stepping("next") },
      { key: "ArrowLeft", by: "viewer: previous", run: stepping("previous") },
      { key: "s", by: "viewer: slideshow", run: () => playing(!isPlaying()) },
    ]),
  );

  onDocument("pointermove", wake);

  // --- the slideshow ---------------------------------------------------------
  //
  // The walk on a timer, and nothing more than that: it takes the same
  // step an arrow takes, to the same address, through the same `walk` the
  // container gave us. There is no second ordering, no prefetch queue and
  // no list of slugs -- which is what keeps it honest when the answer
  // moves under it.
  //
  // Whether it is RUNNING lives in the workspace rather than in a variable
  // here, because the viewer is remounted on every step: the overlay
  // replaces its contents and the page navigates, so a timer held in this
  // closure dies with each picture. Playing is a fact about the walk; this
  // mount is one frame of it. That one decision is also what makes the
  // slideshow work identically in both containers.
  const shown = findElement(root, "[data-slideshow]", HTMLElement);
  let ticking = 0;

  const isPlaying = () => workspace().showPlaying === true;
  const every = () => {
    const held = workspace().showEvery;
    return typeof held === "number" && held > 0 ? held : DEFAULT_EVERY;
  };

  /**
   * Turn wrapping on or off, on the picture already open.
   *
   * Two things at once, and they have to stay together: the attribute
   * the stylesheet reads, and the PROMOTION of the server's wrapping
   * arrow into a real `data-nav` step. Without the second, `data-nav`
   * would mean "an arrow the server drew" rather than "a step you can
   * take", and everything that reads it -- the keys, the wheel, the
   * overlay's click intercept, every browser test -- would have to
   * remember to ask about wrap as well.
   */
  const wrapping = (on: boolean) => {
    root.dataset.wrap = on ? "on" : "off";
    for (const arrow of everyElement(root, "[data-nav-wrap]", HTMLAnchorElement)) {
      if (on) arrow.dataset.nav = requireData(arrow, "navWrap");
      else delete arrow.dataset.nav;
    }
  };

  /** Step, or reach the end and answer the way `loop` says to. */
  const advance = () => {
    const step = findElement(root, '[data-nav="next"]:not([data-nav-wrap])', HTMLAnchorElement);
    if (step) {
      walk(step.href);
      return;
    }
    // The end. `loop` decides this, NEVER `wrap`: wrap is what the arrows
    // do, and somebody can reasonably want arrows that stop -- so they can
    // tell when they have seen everything -- over a slideshow that repeats
    // all night. So this reads the server's wrapping arrow directly rather
    // than whatever wrap promoted, because there is only one first member
    // and its address is the same either way.
    const around = findElement(root, '[data-nav-wrap="next"]', HTMLAnchorElement);
    if (workspace().loop === true && around) {
      walk(around.href);
      return;
    }
    playing(false);
  };

  const playing = (on: boolean) => {
    window.clearTimeout(ticking);
    remember({ showPlaying: on });
    if (shown) {
      shown.dataset.playing = on ? "yes" : "no";
      const control = findElement(shown, "[data-show-play]", HTMLElement);
      if (control) {
        control.setAttribute("aria-pressed", on ? "true" : "false");
        control.setAttribute("aria-label", on ? "pause the slideshow" : "start the slideshow");
        control.textContent = on ? "⏸" : "▶";
      }
    }
    // Never schedule where there is no walk to take: a one-member answer
    // would otherwise sit ticking, reaching the end and stopping itself.
    if (on && shown) ticking = window.setTimeout(advance, every() * 1000);
  };

  if (shown) {
    const settings = findElement(shown, "[data-show-settings]", HTMLElement);
    const opener = findElement(shown, "[data-show-settings-toggle]", HTMLElement);
    const interval = findElement(shown, "[data-show-every]", HTMLSelectElement);
    const wraps = findElement(shown, "[data-show-wrap]", HTMLInputElement);
    const loops = findElement(shown, "[data-show-loop]", HTMLInputElement);

    // The controls open holding what this person last said, because a
    // slideshow whose speed resets every picture is a slideshow nobody
    // can set.
    const kept = workspace();
    if (interval) interval.value = String(every());
    if (wraps) wraps.checked = kept.wrap === true;
    if (loops) loops.checked = kept.loop === true;
    wrapping(kept.wrap === true);

    if (opener && settings) {
      onElement(opener, "click", () => {
        settings.hidden = !settings.hidden;
        opener.setAttribute("aria-expanded", settings.hidden ? "false" : "true");
      });
    }
    const play = findElement(shown, "[data-show-play]", HTMLElement);
    if (play) onElement(play, "click", () => playing(!isPlaying()));
    if (interval) {
      onElement(interval, "change", () => {
        remember({ showEvery: Number(interval.value) });
        // Re-arm at the new speed rather than waiting out the old one:
        // changing 15s to 2s and then watching fifteen seconds pass looks
        // exactly like a control that did nothing.
        if (isPlaying()) playing(true);
      });
    }
    if (wraps) {
      onElement(wraps, "change", () => {
        remember({ wrap: wraps.checked });
        // The arrows change NOW, on the picture already open -- no step,
        // no reload.
        wrapping(wraps.checked);
      });
    }
    if (loops) onElement(loops, "change", () => remember({ loop: loops.checked }));

    // Pick the walk back up where the last mount left it. This IS the
    // slideshow: every step is a fresh viewer that finds `showPlaying`
    // still true and schedules the next one.
    playing(isPlaying());
  }

  // --- the filmstrip ---------------------------------------------------------
  // Rendered by the server in answer order, with the walked question
  // already on every href. Nothing here sorts, pages or invents a member:
  // the only two jobs are putting the current one where the eye expects
  // it, and making a click mean the same thing an arrow key means.
  const strip = findElement(root, "[data-filmstrip-track]", HTMLElement);
  if (strip) {
    const here = findElement(strip, "[data-filmstrip-item][aria-current='true']", HTMLElement);
    // no smooth: on mount it should simply already be there
    here?.scrollIntoView({ block: "nearest", inline: "center" });

    onElement(strip, "click", (event) => {
      const near = closestFrom(event.target, "[data-filmstrip-item]", HTMLAnchorElement);
      if (!near || !isPlainClick(event, near)) return; // a modified click is the browser's
      event.preventDefault();
      walk(near.href);
    });
  }

  for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
    onElement(button, "click", () => showInspector(root.dataset.inspector !== "open"));
  }
  for (const button of everyElement(root, "[data-panel-open]", HTMLElement)) {
    onElement(button, "click", () => {
      showInspector(true);
      panel(requireData(button, "panelOpen"));
    });
  }
  // Two, not one: the chrome's toggle turns the lights off, and the one
  // that outlives the chrome turns them back on. Same command, so binding
  // only the first would leave whichever came second inert.
  for (const button of everyElement(root, "[data-focus]", HTMLElement)) {
    onElement(button, "click", focus);
  }

  // --- what this person arranged, restored ----------------------------------
  //
  // The viewer is remounted on every walk, so without this each picture
  // reopened the template's guesses: the panel shut, `about` disclosed
  // because the markup says `open`, and a person who had set it up once
  // set it up again nineteen more times.
  //
  // Defaults still exist, and they differ by what the picture IS -- a
  // generated image leads with how it was made, a photograph with where
  // and when. They apply only until somebody decides; after that the
  // decision wins on every picture of every kind.
  if (inspector) {
    const kept = workspace();
    const generated = inspector.querySelector("[data-panel='creation']") !== null && root.dataset.made === "generated";
    showInspector(kept.inspector ? kept.inspector === "open" : false, false);
    for (const section of everyElement(inspector, "[data-panel]", HTMLDetailsElement)) {
      const named = section.dataset.panel ?? "";
      const said = panelState(named);
      section.open = said ?? (generated ? named === "creation" : named === "about");
      onElement(section, "toggle", () => rememberPanel(named, section.open));
    }
  }

  // The recipe's scratch fields are wired per mount, which is also what
  // discards their edits: the viewer remounts on every walk, so the next
  // picture arrives with the text its own file carries.
  mountRecipe(root);

  root.dataset.inspector = root.dataset.inspector ?? "closed";
  root.dataset.chrome = "visible";
  stageBox.dataset.quality = "preview";
  paint();
  wake();

  return {
    /**
     * Escape unwinds what the viewer is doing before it means "leave".
     *
     * The ladder, outermost state first: a picture pushed off centre comes
     * back, a zoomed picture fits, an open inspector closes, hidden chrome
     * returns. Only a viewer with nothing left to undo hands Escape to its
     * container -- which is why the container asks rather than owning the
     * key: one shared shell, one dismissal, and no competing listener.
     */
    unwind: () => {
      // A running slideshow is the outermost thing the viewer is doing,
      // and the first thing Escape should stop: a person reaching for it
      // while pictures are moving means "stop", not "leave".
      if (isPlaying()) {
        playing(false);
        return true;
      }
      if (look.scale > 1 || look.x !== 0 || look.y !== 0) {
        frame("fit");
        return true;
      }
      if (root.dataset.inspector === "open") {
        showInspector(false);
        return true;
      }
      if (root.dataset.chrome === "focus") {
        focus();
        return true;
      }
      return false;
    },
    release: () => {
      window.clearTimeout(idle);
      // The pending step goes with the mount that scheduled it. The NEXT
      // mount schedules its own from `showPlaying`, so releasing here
      // cannot stop the slideshow -- but leaving it would let a released
      // viewer walk a container that has moved on.
      window.clearTimeout(ticking);
      for (const off of bound) off();
      bound.length = 0;
    },
  };
}
