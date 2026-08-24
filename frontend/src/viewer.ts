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
//   chrome      visible -> auto-hidden while the pointer rests -> hidden by
//               F, restored by the next movement
//   inspector   closed | open (I). WHERE it sits is geometry's business:
//               CSS docks it beside a wide stage and sheets it under a
//               narrow one, so nothing here knows which
//   filmstrip   present when the walk has neighbours and there is room, and
//               it recedes while zoomed in, because pixels matter more than
//               neighbours once somebody is inspecting them
//   zoom/pan    fit | fill | 1:1 | anywhere between, with pan
//   quality     preview -> original, decided by arithmetic, never by a button
//
// Nothing here is persisted. Zoom, pan, which panel is open and whether the
// chrome is hidden are what the person is doing RIGHT NOW; the moment they
// become a setting is the moment somebody has to be asked a question about
// a photograph they were looking at.
import { everyElement, findElement, requireData } from "./dom";
import type { components } from "./generated/api";

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
  // and changes with the inspector, the filmstrip and the window; deriving it
  // here would be a second layout engine, free to disagree with the real one.
  const fitted = (): { width: number; height: number } => {
    const rect = (media ?? stageBox).getBoundingClientRect();
    return { width: rect.width / look.scale, height: rect.height / look.scale };
  };

  const paint = () => {
    if (!media) return;
    media.style.transform = `translate(${look.x}px, ${look.y}px) scale(${look.scale})`;
    stageBox.dataset.framing = look.framing;
    // The walk recedes once the picture is being inspected rather than
    // browsed. A class would hide the state; the attribute IS the state.
    root.dataset.absorbed = look.scale > ABSORBED ? "true" : "false";
    root.dataset.zoom = String(Math.round(look.scale * 100));
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
    look = {
      framing: "free",
      scale: next,
      x: px - (px - look.x) * ratio,
      y: py - (py - look.y) * ratio,
    };
    paint();
    promote();
  };

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

  const showInspector = (open: boolean) => {
    if (!inspector) return;
    root.dataset.inspector = open ? "open" : "closed";
    for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
      button.setAttribute("aria-expanded", String(open));
    }
  };

  const panel = (named: string | null) => {
    if (!inspector) return;
    for (const section of everyElement(inspector, "[data-panel]", HTMLElement)) {
      const mine = section.dataset.panel === named;
      section.dataset.open = String(mine);
    }
  };

  // --- the walk, on the wheel -----------------------------------------------

  /**
   * Whether the key that walks the library is down for this event.
   *
   * The run's answer, rendered onto the root (db/settings.py
   * `viewer_wheel_modifier`). An unknown word -- or "none" -- means no
   * modifier walks, and the wheel only ever zooms; that is the setting
   * doing its job, not a failure to read it.
   */
  const held = (event: WheelEvent): boolean => {
    const asked = root.dataset.wheelModifier;
    if (asked === "alt") return event.altKey;
    if (asked === "shift") return event.shiftKey;
    if (asked === "ctrl") return event.ctrlKey;
    return false;
  };

  /** One wheel notch, roughly, on the platforms that disagree about size. */
  const NOTCH = 90;
  /** A fling is one gesture; a picture per event would cross the library. */
  const SETTLE_MS = 320;
  let rolled = 0;
  // NOT 0: `performance.now()` counts from the page's load, so zero would
  // mean "stepped at load" and the cooldown would swallow the first flick
  // of anyone who opened a picture and immediately reached for the wheel.
  let lastStep = Number.NEGATIVE_INFINITY;

  /**
   * Step the walk once a gesture has actually asked for it.
   *
   * A wheel does not emit one event per notch -- a trackpad emits dozens
   * of small ones for a single flick -- so the deltas are accumulated to
   * a notch's worth and the counter is reset on each step. The cooldown
   * is the second half: a hard fling arrives as one burst, and without it
   * a single gesture would walk past everything it crossed. Reversing
   * direction drops whatever was accumulated the other way, so a
   * correction is immediate rather than having to undo itself first.
   */
  const stepped = (by: number) => {
    if (by === 0) return;
    if (rolled !== 0 && Math.sign(by) !== Math.sign(rolled)) rolled = 0;
    rolled += by;
    if (Math.abs(rolled) < NOTCH) return;
    const now = performance.now();
    if (now - lastStep < SETTLE_MS) return;
    const wanted = rolled > 0 ? "next" : "previous";
    rolled = 0;
    const step = findElement(root, `[data-nav="${wanted}"]`, HTMLAnchorElement);
    if (!step) return; // an end of the walk is an answer, not a thing to force
    lastStep = now;
    walk(step.href);
  };

  // --- the pointer ----------------------------------------------------------

  // The wheel is bound for EVERY kind, not only the ones that zoom: a clip
  // and a sound file sit in the same walk as a photograph, and stepping
  // past them is the whole point of putting the walk on the wheel.
  onElement(
    stageBox,
    "wheel",
    (event) => {
      // Always: a wheel over the stage is the viewer's, so the page never
      // scrolls out from under a zoom and ctrl never reaches the browser's
      // own page zoom.
      event.preventDefault();
      // deltaMode 1 is lines, 2 is pages: a trackpad and a wheel must not
      // mean different amounts of picture
      const pixels = (delta: number) => (event.deltaMode === 0 ? delta : delta * 16);

      // The chosen modifier turns the wheel into the WALK: the next
      // picture, the previous one. Exactly ONE key does this, named by the
      // run's setting, so the other two keep whatever the browser does
      // with them and nobody has to remember three answers. Some browsers
      // move a shifted wheel's amount into deltaX, so both axes are read
      // rather than either being trusted.
      if (held(event)) {
        stepped(pixels(event.deltaY || event.deltaX));
        return;
      }

      // and without it, the wheel zooms -- which a stage with nothing to
      // zoom simply declines
      zoomAbout(Math.exp(-pixels(event.deltaY) / 400), event.clientX, event.clientY);
    },
    // not passive: the page must not scroll out from under a zoom
    { passive: false },
  );

  if (still && media) {
    onElement(stageBox, "dblclick", (event) => {
      event.preventDefault();
      frame(look.framing === "actual" ? "fit" : "actual");
    });

    let dragging: number | null = null;
    let from = { x: 0, y: 0, ox: 0, oy: 0 };

    onElement(stageBox, "pointerdown", (event) => {
      if (event.button !== 0 || look.scale <= 1) return;
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
      look = {
        framing: "free",
        scale: look.scale,
        x: from.ox + (event.clientX - from.x),
        y: from.oy + (event.clientY - from.y),
      };
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
  onDocument("keydown", (event) => {
    if (!root.isConnected) return;
    const target = event.target;
    if (target instanceof HTMLElement && target.closest("input, textarea, select, [contenteditable]")) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const acted: Record<string, (() => void) | undefined> = {
      i: () => showInspector(root.dataset.inspector !== "open"),
      f: focus,
      t: () => {
        root.dataset.filmstrip = root.dataset.filmstrip === "hidden" ? "shown" : "hidden";
      },
      "1": () => frame("actual"),
      "0": () => frame("fit"),
      "+": () => zoomAbout(1.3, innerWidth / 2, innerHeight / 2),
      "=": () => zoomAbout(1.3, innerWidth / 2, innerHeight / 2),
      "-": () => zoomAbout(1 / 1.3, innerWidth / 2, innerHeight / 2),
    };
    const stepped: Record<string, string | undefined> = { ArrowRight: "next", ArrowLeft: "previous" };
    const wanted = stepped[event.key];
    if (wanted) {
      const step = findElement(root, `[data-nav="${wanted}"]`, HTMLAnchorElement);
      if (!step) return; // an end of the walk is an answer, not a key to eat
      event.preventDefault();
      walk(step.href);
      return;
    }
    const run = acted[event.key.toLowerCase()];
    if (!run) return;
    event.preventDefault();
    run();
  });

  onDocument("pointermove", wake);

  for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
    onElement(button, "click", () => showInspector(root.dataset.inspector !== "open"));
  }
  for (const button of everyElement(root, "[data-panel-open]", HTMLElement)) {
    onElement(button, "click", () => {
      showInspector(true);
      panel(requireData(button, "panelOpen"));
    });
  }
  const focusButton = findElement(root, "[data-focus]", HTMLElement);
  if (focusButton) onElement(focusButton, "click", focus);

  root.dataset.inspector = root.dataset.inspector ?? "closed";
  root.dataset.filmstrip = root.dataset.filmstrip ?? "shown";
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
      for (const off of bound) off();
      bound.length = 0;
    },
  };
}
