// Interaction only. The surface is rendered by the server -- one window
// of the human timeline: the axis with its frames and pictures, the rule
// with its brush, the body (templates/_timeline_surface.html) -- and
// every move here is a request for that same fragment at another
// window, swapped in place with the URL updated. The brush and the
// axis map pointer geometry onto time; a bar and a preset are links
// that work without this file. The story button drives the story
// routes and waits on the job feed.
//
// Two kinds of request, and the difference is deliberate. The surface is
// asked for as HTML, because a fragment swapped in place IS the answer.
// Everything else -- the mosaics in the scrubber, the picture under the
// pointer -- is asked for through the generated client, so the shapes come
// from the application's own contract.
import { api, refusal } from "./api";
import { closestFrom, everyElement, findElement, requireData } from "./dom";
import type { components, paths } from "./generated/api";
import { decodeJobFrame } from "./jobframes";

type JobKind = components["schemas"]["JobListed"]["kind"];
type JobState = components["schemas"]["JobListed"]["state"];
type TimelineNth = components["schemas"]["TimelineNth"];

/**
 * The job kinds whose settling can change what a timeline window says.
 *
 * `satisfies` is the point: these are the schema's own kinds, so a name
 * that is not one is a compile error. Three here used to be route names
 * rather than kinds -- `ingest` queues a `scan`, `faces` a `detect_faces`,
 * `cluster` a `cluster_faces` -- and the page had quietly not refreshed
 * after any of those three since they were written.
 *
 * A kind belongs here when it moves a picture's moment (`scan`,
 * `context`), regroups them (`events`), changes who is in a session
 * (`detect_faces`, `cluster_faces`), or tells its story (`story_plan`).
 * Embedding and captioning change none of that.
 */
const INVALIDATES: ReadonlySet<string> = new Set([
  "scan",
  "context",
  "events",
  "detect_faces",
  "cluster_faces",
  "story_plan",
] satisfies JobKind[]);

const SETTLED: ReadonlySet<string> = new Set(["done", "failed", "cancelled"] satisfies JobState[]);

/** The window on screen, and the library it sits in. */
type WindowState = {
  start: number;
  end: number;
  extentStart: number;
  extentEnd: number;
  scope: string;
};

/** One of the overview's bars, as something a window is drawn toward. */
type Mass = { at: number; end: number; weight: number };

type DragMode = "new" | "start" | "end" | "move";

/** The rule in hand. `box` and `held` are read ONCE, at pointerdown. */
type DragState = {
  overview: SVGSVGElement;
  box: DOMRect;
  held: WindowState;
  mode: DragMode;
  at: number;
  last: number;
};

/** The axis in hand. `px` is its width, read once for the same reason. */
type PanState = {
  axis: HTMLElement;
  px: number;
  x: number;
  start: number;
  end: number;
  moved: boolean;
  held: WindowState;
};

/** The scrubber in hand. No capture until the hand actually moves. */
type ScrubState = {
  held: WindowState;
  rail: HTMLElement;
  pointer: number;
  x: number;
  y: number;
  moved: boolean;
};

type PinchState = { distance: number; row: number };

/**
 * The scope every ask carries, derived from the contract's own query.
 *
 * `Required` is what makes it hold: a scope the server grows is a missing
 * property here rather than a filter this file silently stops sending.
 */
type Scope = Required<
  Omit<NonNullable<paths["/timeline/spread"]["get"]["parameters"]["query"]>, "start" | "end" | "n">
>;

(() => {
  const swap = findElement(document, "#timeline-swap", HTMLElement);
  if (!swap) return;
  const NARROWEST = 3600;
  const W = 1000;

  const surface = () => findElement(swap, "[data-surface]", HTMLElement);
  const read = (): WindowState | null => {
    const s = surface();
    if (!s || s.dataset.extentStart === undefined) return null;
    return {
      start: Number(s.dataset.windowStart),
      end: Number(s.dataset.windowEnd),
      extentStart: Number(s.dataset.extentStart),
      extentEnd: Number(s.dataset.extentEnd),
      scope: s.dataset.scopeQs ?? "",
    };
  };

  // the window's URL: the scope's own parameters ride every move; `snap`
  // is the scrubber's ask, answered by the server with a window on pictures
  const urlFor = (start: number, end: number, snap = false): string => {
    const qs = new URLSearchParams(read()?.scope ?? "");
    qs.set("start", String(start));
    qs.set("end", String(end));
    if (snap) qs.set("snap", "true");
    return `/timeline?${qs}`;
  };

  /** The same scope, as the typed client's query rather than a string. */
  const scopeOf = (): Scope => {
    const qs = new URLSearchParams(read()?.scope ?? "");
    const rating = qs.get("rating_min");
    return {
      folder: qs.get("folder"),
      album: qs.get("album"),
      person: qs.get("person"),
      artifact: qs.get("artifact"),
      kind: qs.get("kind"),
      favorite: qs.get("favorite"),
      rating_min: rating === null ? null : Number(rating),
      f: qs.getAll("f"),
    };
  };

  // the newest move is the only one that lands; while it is in flight the
  // swap root says so (data-loading), and says nothing once it has landed
  // -- what a hand, a stylesheet or a test waits on
  let drag: DragState | null = null;
  let generation = 0;
  const settled = (mine: number) => {
    if (mine === generation) delete swap.dataset.loading;
  };
  // an arrow, not a declaration: a hoisted function could be called from
  // anywhere, so TypeScript will not carry the `swap` narrowing into it
  const move = async (url: string, push: boolean | null): Promise<void> => {
    const mine = ++generation;
    swap.dataset.loading = "";
    // HTML on purpose: the surface fragment IS the answer, swapped in place
    const answer = await fetch(url, { headers: { "hx-request": "true", accept: "text/html" } });
    if (mine !== generation) return; // a newer move superseded this one
    if (!answer.ok) {
      // the refusal's body is unmodelled, so it is read as unknown and
      // proven rather than reached into
      const why: unknown = await answer.json().catch(() => null);
      const note = findElement(swap, "[data-note]", HTMLElement);
      if (note) note.textContent = refusal(why, answer.statusText);
      settled(mine);
      return;
    }
    const body = await answer.text();
    if (mine !== generation) return; // superseded while the body was arriving
    const held = findElement(swap, "[data-strip]", HTMLElement);
    if (held) held.dataset.settling = "";
    swap.innerHTML = body;
    // the rule under a hand is never swapped out from under it: while a
    // drag holds the overview, the fresh surface takes the held node in
    // place of its own, so capture, geometry and the release all speak
    // of one element; the release's own move replaces everything
    if (drag) {
      const fresh = findElement(swap, "[data-overview]", SVGSVGElement);
      if (fresh && fresh !== drag.overview) fresh.replaceWith(drag.overview);
    }
    thin();
    if (push === true) history.pushState({ url }, "", url);
    else if (push === false) history.replaceState({ url }, "", url);
    // push === null: a refresh of the same window; the URL already says it
    settled(mine);
  };

  /** This window, read again from the rows. */
  const revalidate = () => void move(location.pathname + location.search, null);

  // the pictures on the axis sit at their moment and never wrap: when
  // two would overlap, the later one yields -- a thinner strip, not a pile
  const thin = () => {
    const row = findElement(swap, "[data-samples]", HTMLElement);
    if (!row) return;
    const width = row.getBoundingClientRect().width || 1;
    let edge = Number.NEGATIVE_INFINITY;
    for (const a of everyElement(row, ".surface-sample", HTMLElement)) {
      const left = (Number.parseFloat(a.style.left) / 100) * width;
      if (left < edge) {
        a.hidden = true;
        continue;
      }
      a.hidden = false;
      edge = left + 42;
    }
  };
  thin();
  window.addEventListener("resize", thin);

  // While the hand moves, the surface moves: at most one fetch in flight
  // per LIVE_MS, the newest window always the one that lands.
  const LIVE_MS = 120;
  let liveAt = 0;
  let liveTimer = 0;
  const live = (start: number, end: number, snap = false) => {
    const now = performance.now();
    clearTimeout(liveTimer);
    const run = () => {
      liveAt = performance.now();
      void move(urlFor(Math.round(start), Math.round(end), snap), false);
    };
    if (now - liveAt >= LIVE_MS) run();
    else liveTimer = window.setTimeout(run, LIVE_MS - (now - liveAt));
  };

  // Time refreshes itself: when a job that dates or groups pictures
  // settles, the window is fetched again -- nobody reloads a timeline.
  (() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const open = (): void => {
      const feed = new WebSocket(`${proto}://${location.host}/ws/jobs`);
      feed.onmessage = (msg: MessageEvent<unknown>) => {
        const frame = decodeJobFrame(msg.data);
        if (frame === null) {
          // a payload this build cannot read is a transport failure, not a
          // message to skip: close, and the reconnect opens with a snapshot
          feed.close();
          return;
        }
        if (frame.type === "snapshot") {
          // A SNAPSHOT IS A RESYNCHRONISATION BOUNDARY, never a frame to
          // ignore. It says "your view may have changed while you were not
          // looking" -- and it cannot say more than that, because it lists
          // only UNSETTLED jobs: a job that settled between this page's
          // render and this connection is simply absent from it. Waiting
          // for a terminal delta that already happened, or was lost with a
          // dropped socket, leaves the window stale forever. So the answer
          // is to read the rows again. One fragment fetch per connection.
          revalidate();
          return;
        }
        if (SETTLED.has(frame.state) && INVALIDATES.has(frame.kind)) revalidate();
      };
      feed.onclose = () => window.setTimeout(open, 2000);
      feed.onerror = () => feed.close();
    };
    open();
  })();

  window.addEventListener("popstate", (e: PopStateEvent) => {
    const held: unknown = e.state;
    const url =
      typeof held === "object" && held !== null && "url" in held && typeof held.url === "string"
        ? held.url
        : location.pathname + location.search;
    void move(url, false);
  });

  // a bar or a preset is a link to another window: swap instead of navigate
  swap.addEventListener("click", (e) => {
    const a = closestFrom(e.target, "[data-preset], [data-bin-window], [data-month-window]", HTMLAnchorElement);
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return; // a modified click is the browser's
    e.preventDefault();
    void move(a.getAttribute("href") ?? location.pathname + location.search, true);
  });

  // --- the pull ----------------------------------------------------------
  // Pictures have mass. A hand-placed moment inside the reach of a bin
  // that holds pictures is drawn toward it -- continuously, stronger the
  // nearer and the heavier, nothing at all past the reach -- so a window
  // settles on pictures instead of beside them. The masses are the
  // overview's bars, read from the page each time: the swap replaces them.
  const REACH = 0.025; // of the library's extent, either side
  const masses = (): Mass[] => {
    const out: Mass[] = [];
    for (const bar of everyElement(swap, ".overview-bar[data-pictures]", SVGRectElement)) {
      const n = Number(bar.dataset.pictures);
      if (n > 0) out.push({ at: Number(bar.dataset.at), end: Number(bar.dataset.end), weight: Math.sqrt(n) });
    }
    return out;
  };
  const pull = (held: WindowState, t: number, field: Mass[] = masses()): number => {
    const reach = REACH * (held.extentEnd - held.extentStart);
    let force = 0;
    let toward = 0;
    for (const m of field) {
      const d = t < m.at ? m.at - t : t > m.end ? t - m.end : 0; // inside a bin: no pull
      if (d === 0 || d > reach) continue;
      const w = m.weight * (1 - d / reach) ** 2;
      force += w;
      toward += w * (t < m.at ? m.at : m.end);
    }
    if (!force) return t;
    const heaviest = Math.max(...field.map((m) => m.weight));
    const grip = Math.min(1, force / heaviest); // how much of the way it is drawn
    return t + (toward / force - t) * grip;
  };

  // --- the brush ---------------------------------------------------------
  const ox = (held: WindowState, t: number) =>
    ((t - held.extentStart) / Math.max(1, held.extentEnd - held.extentStart)) * W;
  const ot = (held: WindowState, x: number) =>
    held.extentStart + (Math.min(W, Math.max(0, x)) / W) * (held.extentEnd - held.extentStart);
  // the rule's box is read ONCE, at pointerdown: the live swap replaces
  // the element under the hand, and a detached element's box is all zeros
  const overviewX = (box: DOMRect, clientX: number) => ((clientX - box.left) / (box.width || 1)) * W;
  const placeBrush = (overview: SVGSVGElement, held: WindowState, start: number, end: number) => {
    const x0 = ox(held, start);
    const x1 = ox(held, end);
    const body = findElement(overview, "[data-brush]", SVGRectElement);
    const from = findElement(overview, '[data-brush-edge="start"]', SVGRectElement);
    const to = findElement(overview, '[data-brush-edge="end"]', SVGRectElement);
    if (body) {
      body.setAttribute("x", String(x0));
      body.setAttribute("width", String(Math.max(2, x1 - x0)));
    }
    if (from) from.setAttribute("x", String(x0 - 3));
    if (to) to.setAttribute("x", String(x1 - 3));
  };

  // the hand's place on the rule: inside the box, the pointer; outside
  // it, the last place it was inside -- letting go past the edge keeps
  // the last valid edge, it does not fling the window to the end
  const handAt = (held: DragState, event: PointerEvent): number => {
    if (event.clientX >= held.box.left && event.clientX <= held.box.right) held.last = event.clientX;
    return held.last;
  };
  const dragged = (state: DragState, event: PointerEvent): { start: number; end: number } => {
    const { held, mode, at } = state;
    const x = overviewX(state.box, handAt(state, event));
    const dt = ot(held, x) - ot(held, at);
    // a window is never narrower than an hour, or than the library itself
    const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
    let start = held.start;
    let end = held.end;
    const field = masses();
    if (mode === "move") {
      const width = end - start;
      end = pull(held, Math.min(Math.max(held.extentStart + width, end + dt), held.extentEnd), field);
      end = Math.min(held.extentEnd, Math.max(held.extentStart + width, end));
      start = end - width;
    } else if (mode === "start") {
      start = Math.max(held.extentStart, Math.min(pull(held, start + dt, field), end - narrowest));
    } else if (mode === "end") {
      end = Math.min(held.extentEnd, Math.max(pull(held, end + dt, field), start + narrowest));
    } else {
      const a = pull(held, ot(held, at), field);
      const b = pull(held, ot(held, x), field);
      start = Math.max(held.extentStart, Math.min(a, b));
      end = Math.min(held.extentEnd, Math.max(a, b, start + narrowest));
    }
    return { start, end };
  };

  swap.addEventListener("pointerdown", (event) => {
    const overview = closestFrom(event.target, "[data-overview]", SVGSVGElement);
    const held = read();
    if (!overview || !held) return;
    const box = overview.getBoundingClientRect();
    const x = overviewX(box, event.clientX);
    const x0 = ox(held, held.start);
    const x1 = ox(held, held.end);
    const grip = 8;
    let mode: DragMode = "new";
    if (Math.abs(x - x0) <= grip) mode = "start";
    else if (Math.abs(x - x1) <= grip) mode = "end";
    else if (x > x0 && x < x1) mode = "move";
    drag = { overview, box, held, mode, at: x, last: event.clientX };
    overview.setPointerCapture(event.pointerId);
    overview.dataset.dragging = mode;
    event.preventDefault();
  });
  swap.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const { start, end } = dragged(drag, event);
    placeBrush(drag.overview, drag.held, start, end);
    live(start, end, true);
  });
  const release = (event: PointerEvent) => {
    if (!drag) return;
    const { start, end } = dragged(drag, event);
    delete drag.overview.dataset.dragging;
    drag = null;
    clearTimeout(liveTimer);
    // the hand's window, landed on pictures by the server (snap), then
    // the URL says the window the page actually shows
    void move(urlFor(Math.round(start), Math.round(end), true), true).then(() => {
      const held = read();
      if (!held) return;
      const url = urlFor(Math.round(held.start), Math.round(held.end));
      history.replaceState({ url }, "", url);
    });
  };

  // the axis pans under the hand: a drag moves the window, a click is a
  // click (a bar is a link) -- the hand decides by moving
  let pan: PanState | null = null;
  swap.addEventListener("pointerdown", (event) => {
    const axis = closestFrom(event.target, "[data-strip]", HTMLElement);
    const held = read();
    if (!axis || !held || event.button !== 0) return;
    // the axis is swapped under the hand while it moves: its width is
    // read once here, never from an element that may since be detached
    pan = {
      axis,
      px: axis.getBoundingClientRect().width || 1,
      x: event.clientX,
      start: held.start,
      end: held.end,
      moved: false,
      held,
    };
  });
  swap.addEventListener("pointermove", (event) => {
    if (!pan) return;
    if (!pan.moved && Math.abs(event.clientX - pan.x) < 4) return;
    if (!pan.moved) {
      pan.moved = true;
      pan.axis.dataset.dragging = "";
      pan.axis.setPointerCapture(event.pointerId);
    }
    const width = pan.end - pan.start;
    const dt = ((pan.x - event.clientX) / pan.px) * width;
    let end = pull(pan.held, pan.end + dt);
    end = Math.min(pan.held.extentEnd, Math.max(pan.held.extentStart + width, end));
    live(end - width, end, true);
  });
  let panned = false; // the click that ends a drag is not a click
  swap.addEventListener(
    "click",
    (e) => {
      if (panned && closestFrom(e.target, "[data-strip]", Element)) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
      panned = false;
    },
    true,
  );
  const unpan = () => {
    if (!pan) return;
    const was = pan;
    pan = null;
    panned = was.moved;
    if (!was.moved) return;
    delete was.axis.dataset.dragging;
    clearTimeout(liveTimer);
    const held = read();
    if (held) void move(urlFor(Math.round(held.start), Math.round(held.end)), true);
  };
  window.addEventListener("pointerup", unpan);
  window.addEventListener("pointercancel", unpan);

  // ctrl+wheel over the axis or the rule zooms around the cursor; shift+wheel
  // pans; a plain wheel is the page's, so the river below stays reachable
  swap.addEventListener(
    "wheel",
    (e) => {
      const stage = closestFrom(e.target, "[data-strip], [data-overview]", Element);
      const held = read();
      if (!stage || !held || !(e.ctrlKey || e.metaKey || e.shiftKey)) return;
      e.preventDefault();
      const width = held.end - held.start;
      const box = stage.getBoundingClientRect();
      const at = held.start + ((e.clientX - box.left) / (box.width || 1)) * width;
      let start: number;
      let end: number;
      if (e.shiftKey) {
        const step = ((e.deltaY > 0 ? 1 : -1) * width) / 5;
        start = held.start + step;
        end = held.end + step;
      } else {
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        start = at - (at - held.start) * factor;
        end = at + (held.end - at) * factor;
      }
      const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
      if (end - start < narrowest) {
        start = at - narrowest / 2;
        end = at + narrowest / 2;
      }
      start = Math.max(held.extentStart, start);
      end = Math.min(held.extentEnd, Math.max(end, start + narrowest));
      live(start, end);
    },
    { passive: false },
  );
  // the release is heard on the window: the live swap may have detached
  // the element holding the capture, and an event on a detached element
  // never reaches the stage
  window.addEventListener("pointerup", release);
  window.addEventListener("pointercancel", release);

  swap.addEventListener("keydown", (e) => {
    if (!closestFrom(e.target, "[data-overview]", Element)) return;
    const held = read();
    if (!held) return;
    const width = held.end - held.start;
    const step = width / 4;
    const go = (s: number, t: number) => {
      e.preventDefault();
      void move(urlFor(Math.round(s), Math.round(t)), true);
    };
    if (e.key === "ArrowLeft")
      go(Math.max(held.extentStart, held.start - step), Math.max(held.extentStart + width, held.end - step));
    if (e.key === "ArrowRight")
      go(Math.min(held.extentEnd - width, held.start + step), Math.min(held.extentEnd, held.end + step));
    if (e.key === "+" || e.key === "=") go(held.start + width / 4, held.end - width / 4);
    if (e.key === "-")
      go(Math.max(held.extentStart, held.start - width / 2), Math.min(held.extentEnd, held.end + width / 2));
  });

  // --- the scrubber ----------------------------------------------------------
  // The library top to bottom, newest first, a segment per month sized
  // by its pictures. The hand's y lands in a segment; the fraction of
  // the way down it is the fraction of the way back through the month;
  // an empty month hands the ask to the nearest one with pictures, and
  // the server lands the window on them (`snap`). Geometry is read from
  // the elements under the pointer, never kept from an element the live
  // swap may have replaced.
  const segmentAt = (x: number, y: number): HTMLElement | null => {
    for (const el of document.elementsFromPoint(x, y)) {
      const seg = el.closest(".segment");
      if (seg instanceof HTMLElement) return seg;
    }
    return null;
  };
  const nearestWithPictures = (seg: HTMLElement, y: number): HTMLElement => {
    if (Number(seg.dataset.pictures) > 0) return seg;
    let best = seg;
    let nearest = Number.POSITIVE_INFINITY;
    for (const other of everyElement(swap, ".segment", HTMLElement)) {
      if (!(Number(other.dataset.pictures) > 0)) continue;
      const box = other.getBoundingClientRect();
      const d = y < box.top ? box.top - y : y > box.bottom ? y - box.bottom : 0;
      if (d < nearest) {
        nearest = d;
        best = other;
      }
    }
    return best;
  };
  // Each segment holds as many of its pictures as ITS pixels can show --
  // a mosaic of tiles, filled from /timeline/spread with exactly that
  // many, spread through the segment's whole span. Nothing presumes a
  // count: a segment an inch tall shows a dozen, a screen tall a hundred.
  const TILE = 30;
  const fillSegments = () => {
    for (const seg of everyElement(swap, ".segment.held", HTMLElement)) {
      const strip = findElement(seg, "[data-segment-strip]", HTMLElement);
      if (!strip || strip.dataset.filled) continue;
      const box = seg.getBoundingClientRect();
      const cols = Math.max(1, Math.floor(box.width / TILE));
      const rows = Math.max(1, Math.floor(box.height / (TILE + 1)));
      strip.style.setProperty("--cols", String(cols));
      strip.style.setProperty("--tile", `${TILE}px`);
      const n = Math.min(400, cols * rows);
      strip.dataset.filled = String(n);
      void api
        .GET("/timeline/spread", {
          params: {
            query: {
              ...scopeOf(),
              start: Number(requireData(seg, "at")),
              end: Number(requireData(seg, "end")),
              n,
            },
          },
        })
        .then(({ data }) => {
          if (data === undefined || !strip.isConnected) return;
          strip.replaceChildren(
            ...data.pictures.map((p) => {
              const img = document.createElement("img");
              img.src = `/thumb/${p.slug}`;
              img.alt = "";
              img.loading = "lazy";
              img.draggable = false;
              img.dataset.moment = String(p.moment);
              return img;
            }),
          );
        }, console.error);
    }
  };
  fillSegments();
  new MutationObserver(fillSegments).observe(swap, { childList: true });
  window.addEventListener("resize", () => {
    for (const s of everyElement(swap, "[data-segment-strip]", HTMLElement)) delete s.dataset.filled;
    fillSegments();
  });

  // The hand a fraction of the way down a segment points at a picture by
  // RANK -- the k-th of its n in moment order, newest at the top -- never
  // by time: a burst of thousands in one minute would otherwise map every
  // position to its first or last. /timeline/nth answers it; one ask in
  // flight at a time, the newest always the one that lands.
  const rankAt = (seg: HTMLElement, y: number): number => {
    const box = seg.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (y - box.top) / (box.height || 1)));
    const n = Number(seg.dataset.pictures);
    return Math.min(n - 1, Math.max(0, Math.round((1 - f) * (n - 1))));
  };
  let asking = 0;
  const nth = async (seg: HTMLElement, y: number): Promise<TimelineNth | null> => {
    const mine = ++asking;
    const { data } = await api.GET("/timeline/nth", {
      params: {
        query: {
          ...scopeOf(),
          start: Number(requireData(seg, "at")),
          end: Number(requireData(seg, "end")),
          k: rankAt(seg, y),
        },
      },
    });
    if (mine !== asking || data === undefined) return null;
    return data;
  };
  const peek = (seg: HTMLElement | null, y: number) => {
    const card = findElement(swap, "[data-scrubber-peek]", HTMLElement);
    if (!card) return;
    for (const was of everyElement(swap, ".segment-strip img.under", HTMLElement)) was.classList.remove("under");
    if (!seg) {
      card.hidden = true;
      return;
    }
    const rail = findElement(swap, "[data-scrubber]", HTMLElement);
    const img = findElement(card, "img", HTMLImageElement);
    const label = findElement(card, ".scrubber-peek-label", HTMLElement);
    const count = findElement(card, ".scrubber-peek-count", HTMLElement);
    if (!rail || !img || !label || !count) return;
    const box = rail.getBoundingClientRect();
    card.hidden = false;
    card.style.top = `${Math.min(box.height - 60, Math.max(40, y - box.top))}px`;
    if (!Number(seg.dataset.pictures)) {
      img.removeAttribute("src");
      img.hidden = true;
      label.textContent = seg.dataset.label ?? "";
      count.textContent = "nothing";
      return;
    }
    void nth(seg, y).then((told) => {
      if (!told) return;
      img.src = `/thumb/${told.slug}`;
      img.hidden = false;
      label.textContent = told.spelled;
      count.textContent = `${(told.k + 1).toLocaleString()} of ${told.of.toLocaleString()}`;
      // the mosaic tile nearest that moment lights up
      let best: HTMLElement | null = null;
      let nearest = Number.POSITIVE_INFINITY;
      for (const tile of everyElement(seg, ".segment-strip img[data-moment]", HTMLElement)) {
        const d = Math.abs(Number(tile.dataset.moment) - told.moment);
        if (d < nearest) {
          nearest = d;
          best = tile;
        }
      }
      if (best) best.classList.add("under");
    }, console.error);
  };

  let scrub: ScrubState | null = null;
  let scrubbed = false; // the click that ends a drag is not a click
  swap.addEventListener(
    "click",
    (e) => {
      if (scrubbed && closestFrom(e.target, "[data-scrubber]", Element)) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
      scrubbed = false;
    },
    true,
  );
  swap.addEventListener("pointerdown", (event) => {
    const rail = closestFrom(event.target, "[data-scrubber]", HTMLElement);
    const held = read();
    if (!rail || !held || event.button !== 0) return;
    // no capture yet: a click on a month must reach the month; the hand
    // decides by moving, and only then is the pointer held
    scrub = { held, rail, pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    event.preventDefault(); // a month is a link: no native link-drag, no text selection
  });
  swap.addEventListener("pointermove", (event) => {
    const rail = closestFrom(event.target, "[data-scrubber]", HTMLElement);
    const seg = segmentAt(event.clientX, event.clientY);
    if (rail || scrub) peek(seg, event.clientY);
    const state = scrub;
    if (!state) return;
    if (!state.moved && Math.abs(event.clientY - state.y) < 3) return;
    if (!state.moved) {
      state.moved = true;
      const holding = state.rail.isConnected ? state.rail : findElement(swap, "[data-scrubber]", HTMLElement);
      if (holding) {
        holding.setPointerCapture(state.pointer);
        holding.dataset.dragging = "";
      }
    }
    if (!seg) return;
    const width = state.held.end - state.held.start;
    const target = nearestWithPictures(seg, event.clientY);
    const held = state.held;
    const land = (t: number) => {
      const end = Math.min(held.extentEnd, Math.max(held.extentStart + width, t));
      live(end - width, end, true);
    };
    // the window's newest edge lands on the picture the hand points at,
    // by rank within the segment; an empty segment hands on to the nearest
    if (target !== seg) {
      land(Number(requireData(target, "end")) - 1);
      return;
    }
    void nth(seg, event.clientY).then((told) => {
      if (told && scrub) land(told.moment + 1);
    }, console.error);
  });
  const unscrub = () => {
    if (!scrub) return;
    const was = scrub;
    scrub = null;
    scrubbed = was.moved;
    for (const rail of everyElement(swap, "[data-scrubber]", HTMLElement)) delete rail.dataset.dragging;
    if (!was.moved) return;
    clearTimeout(liveTimer);
    const held = read();
    if (held) void move(urlFor(Math.round(held.start), Math.round(held.end)), true);
  };
  window.addEventListener("pointerup", unscrub);
  window.addEventListener("pointercancel", unscrub);
  swap.addEventListener(
    "pointerleave",
    (e) => {
      if (!scrub && closestFrom(e.target, "[data-scrubber]", Element)) peek(null, 0);
    },
    true,
  );

  // --- the size of the pictures ----------------------------------------------
  // ctrl+wheel or a pinch over the days resizes the rows, and the size is
  // the viewer's from then on; a plain wheel is the page's
  const ROW = { least: 120, most: 520, fallback: 200, key: "timeline.row" };
  const rowOf = (): number => {
    try {
      return Number(localStorage.getItem(ROW.key)) || ROW.fallback;
    } catch {
      return ROW.fallback;
    }
  };
  const sizeRows = (px: number) => {
    const row = Math.min(ROW.most, Math.max(ROW.least, Math.round(px)));
    const s = surface();
    if (s) s.style.setProperty("--row", `${row}px`);
    try {
      localStorage.setItem(ROW.key, String(row));
    } catch {
      /* a private window keeps no size */
    }
  };
  const sized = () => {
    const s = surface();
    if (s) s.style.setProperty("--row", `${rowOf()}px`);
  };
  sized();
  new MutationObserver(sized).observe(swap, { childList: true });
  swap.addEventListener(
    "wheel",
    (e) => {
      if (!(e.ctrlKey || e.metaKey) || !closestFrom(e.target, "[data-sessions]", Element)) return;
      e.preventDefault();
      sizeRows(rowOf() * (e.deltaY > 0 ? 0.9 : 1.1));
    },
    { passive: false },
  );
  let pinch: PinchState | null = null;
  const apart = (touches: TouchList): number | null => {
    const a = touches.item(0);
    const b = touches.item(1);
    if (a === null || b === null) return null;
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  };
  swap.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 2 || !closestFrom(e.target, "[data-sessions]", Element)) return;
      const distance = apart(e.touches);
      if (distance !== null) pinch = { distance, row: rowOf() };
    },
    { passive: true },
  );
  swap.addEventListener(
    "touchmove",
    (e) => {
      if (!pinch || e.touches.length !== 2) return;
      const distance = apart(e.touches);
      if (distance === null) return;
      e.preventDefault();
      sizeRows(pinch.row * (distance / pinch.distance));
    },
    { passive: false },
  );
  swap.addEventListener("touchend", () => {
    pinch = null;
  });
})();
