// Interaction only. The surface is rendered by the server -- one window
// of the human timeline: the axis with its frames and pictures, the rule
// with its brush, the body (templates/_timeline_surface.html) -- and
// every move here is a request for that same fragment at another
// window, swapped in place with the URL updated. The brush and the
// axis map pointer geometry onto time; a bar and a preset are links
// that work without this file. The story button drives the story
// routes and waits on the job feed.
(() => {
  const swap = document.getElementById("timeline-swap");
  if (!swap) return;
  const NARROWEST = 3600;
  const W = 1000;

  const surface = () => swap.querySelector("[data-surface]");
  const read = () => {
    const s = surface();
    if (!s || s.dataset.extentStart == null) return null;
    return {
      start: Number(s.dataset.windowStart),
      end: Number(s.dataset.windowEnd),
      extentStart: Number(s.dataset.extentStart),
      extentEnd: Number(s.dataset.extentEnd),
      scope: s.dataset.scopeQs,
    };
  };

  // the window's URL: the scope's own parameters ride every move; `snap`
  // is the scrubber's ask, answered by the server with a window on pictures
  const urlFor = (start, end, snap = false) => {
    const qs = new URLSearchParams(read()?.scope || "");
    qs.set("start", String(start));
    qs.set("end", String(end));
    if (snap) qs.set("snap", "true");
    return `/timeline?${qs}`;
  };

  // the newest move is the only one that lands; while it is in flight the
  // swap root says so (data-loading), and says nothing once it has landed
  // -- what a hand, a stylesheet or a test waits on
  let drag = null; // the brush in hand: {overview, box, held, mode, at, last}
  let generation = 0;
  const settled = (mine) => {
    if (mine === generation) delete swap.dataset.loading;
  };
  async function move(url, push) {
    const mine = ++generation;
    swap.dataset.loading = "";
    const answer = await fetch(url, { headers: { "hx-request": "true", accept: "text/html" } });
    if (mine !== generation) return; // a newer move superseded this one
    if (!answer.ok) {
      const why = await answer.json().catch(() => ({}));
      const note = swap.querySelector("[data-note]");
      if (note) note.textContent = why.detail || answer.statusText;
      settled(mine);
      return;
    }
    const body = await answer.text();
    if (mine !== generation) return; // superseded while the body was arriving
    const held = swap.querySelector("[data-strip]");
    if (held) held.dataset.settling = "";
    swap.innerHTML = body;
    // the rule under a hand is never swapped out from under it: while a
    // drag holds the overview, the fresh surface takes the held node in
    // place of its own, so capture, geometry and the release all speak
    // of one element; the release's own move replaces everything
    if (drag?.overview) {
      const fresh = swap.querySelector("[data-overview]");
      if (fresh && fresh !== drag.overview) fresh.replaceWith(drag.overview);
    }
    thin();
    if (push === true) history.pushState({ url }, "", url);
    else if (push === false) history.replaceState({ url }, "", url);
    // push === null: a refresh of the same window; the URL already says it
    settled(mine);
  }

  // the pictures on the axis sit at their moment and never wrap: when
  // two would overlap, the later one yields -- a thinner strip, not a pile
  const thin = () => {
    const row = swap.querySelector("[data-samples]");
    if (!row) return;
    const width = row.getBoundingClientRect().width || 1;
    let edge = -Infinity;
    for (const a of row.querySelectorAll(".surface-sample")) {
      const left = (parseFloat(a.style.left) / 100) * width;
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
  const live = (start, end, snap = false) => {
    const now = performance.now();
    clearTimeout(liveTimer);
    const run = () => {
      liveAt = performance.now();
      move(urlFor(Math.round(start), Math.round(end), snap), false);
    };
    if (now - liveAt >= LIVE_MS) run();
    else liveTimer = setTimeout(run, LIVE_MS - (now - liveAt));
  };

  // Time refreshes itself: when a job that dates or groups pictures
  // settles, the window is fetched again -- nobody reloads a timeline.
  (() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let socket;
    const open = () => {
      socket = new WebSocket(`${proto}://${location.host}/ws/jobs`);
      socket.onmessage = (msg) => {
        const frame = JSON.parse(msg.data);
        if (frame.type === "snapshot") return;
        if (
          ["done", "failed", "cancelled"].includes(frame.state) &&
          ["context", "events", "ingest", "scan", "faces", "cluster", "story_plan"].includes(frame.kind)
        ) {
          move(location.pathname + location.search, null);
        }
      };
      socket.onclose = () => setTimeout(open, 2000);
    };
    open();
  })();

  window.addEventListener("popstate", (e) => {
    const url = e.state?.url || location.pathname + location.search;
    move(url, false);
  });

  // a bar or a preset is a link to another window: swap instead of navigate
  swap.addEventListener("click", (e) => {
    const a = e.target.closest("[data-preset], [data-bin-window], [data-month-window]");
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return; // a modified click is the browser's
    e.preventDefault();
    move(a.getAttribute("href"), true);
  });

  // --- the pull ----------------------------------------------------------
  // Pictures have mass. A hand-placed moment inside the reach of a bin
  // that holds pictures is drawn toward it -- continuously, stronger the
  // nearer and the heavier, nothing at all past the reach -- so a window
  // settles on pictures instead of beside them. The masses are the
  // overview's bars, read from the page each time: the swap replaces them.
  const REACH = 0.025; // of the library's extent, either side
  const masses = () => {
    const out = [];
    for (const bar of swap.querySelectorAll(".overview-bar[data-pictures]")) {
      const n = Number(bar.dataset.pictures);
      if (n > 0) out.push({ at: Number(bar.dataset.at), end: Number(bar.dataset.end), weight: Math.sqrt(n) });
    }
    return out;
  };
  const pull = (held, t, field = masses()) => {
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
  const ox = (held, t) => ((t - held.extentStart) / Math.max(1, held.extentEnd - held.extentStart)) * W;
  const ot = (held, x) => held.extentStart + (Math.min(W, Math.max(0, x)) / W) * (held.extentEnd - held.extentStart);
  // the rule's box is read ONCE, at pointerdown: the live swap replaces
  // the element under the hand, and a detached element's box is all zeros
  const overviewX = (box, event) => ((event.clientX - box.left) / (box.width || 1)) * W;
  const placeBrush = (overview, held, start, end) => {
    const x0 = ox(held, start);
    const x1 = ox(held, end);
    overview.querySelector("[data-brush]").setAttribute("x", x0);
    overview.querySelector("[data-brush]").setAttribute("width", Math.max(2, x1 - x0));
    overview.querySelector('[data-brush-edge="start"]').setAttribute("x", x0 - 3);
    overview.querySelector('[data-brush-edge="end"]').setAttribute("x", x1 - 3);
  };

  // the hand's place on the rule: inside the box, the pointer; outside
  // it, the last place it was inside -- letting go past the edge keeps
  // the last valid edge, it does not fling the window to the end
  const inside = (box, event) => event.clientX >= box.left && event.clientX <= box.right;
  const handAt = (event) => {
    if (inside(drag.box, event)) drag.last = event.clientX;
    return { clientX: drag.last ?? event.clientX };
  };
  const dragged = (event) => {
    const { held, mode, at } = drag;
    const x = overviewX(drag.box, handAt(event));
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
    const overview = event.target.closest("[data-overview]");
    const held = read();
    if (!overview || !held) return;
    const box = overview.getBoundingClientRect();
    const x = overviewX(box, event);
    const x0 = ox(held, held.start);
    const x1 = ox(held, held.end);
    const grip = 8;
    let mode = "new";
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
    const { start, end } = dragged(event);
    placeBrush(drag.overview, drag.held, start, end);
    live(start, end, true);
  });
  const release = (event) => {
    if (!drag) return;
    const { start, end } = dragged(event);
    delete drag.overview.dataset.dragging;
    drag = null;
    clearTimeout(liveTimer);
    // the hand's window, landed on pictures by the server (snap), then
    // the URL says the window the page actually shows
    move(urlFor(Math.round(start), Math.round(end), true), true).then(() => {
      const held = read();
      if (held)
        history.replaceState(
          { url: urlFor(Math.round(held.start), Math.round(held.end)) },
          "",
          urlFor(Math.round(held.start), Math.round(held.end)),
        );
    });
  };

  // the axis pans under the hand: a drag moves the window, a click is a
  // click (a bar is a link) -- the hand decides by moving
  let pan = null; // {x, start, end, moved}
  swap.addEventListener("pointerdown", (event) => {
    const axis = event.target.closest("[data-strip]");
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
      if (panned && e.target.closest("[data-strip]")) {
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
    if (held) move(urlFor(Math.round(held.start), Math.round(held.end)), true);
  };
  window.addEventListener("pointerup", unpan);
  window.addEventListener("pointercancel", unpan);

  // ctrl+wheel over the axis or the rule zooms around the cursor; shift+wheel
  // pans; a plain wheel is the page's, so the river below stays reachable
  swap.addEventListener(
    "wheel",
    (e) => {
      const stage = e.target.closest("[data-strip], [data-overview]");
      const held = read();
      if (!stage || !held || !(e.ctrlKey || e.metaKey || e.shiftKey)) return;
      e.preventDefault();
      const width = held.end - held.start;
      const box = stage.getBoundingClientRect();
      const at = held.start + ((e.clientX - box.left) / box.width) * width;
      let start;
      let end;
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
    if (!e.target.closest("[data-overview]")) return;
    const held = read();
    if (!held) return;
    const width = held.end - held.start;
    const step = width / 4;
    const go = (s, t) => {
      e.preventDefault();
      move(urlFor(Math.round(s), Math.round(t)), true);
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
  const segmentAt = (x, y) => {
    for (const el of document.elementsFromPoint(x, y)) {
      const seg = el.closest ? el.closest(".segment") : null;
      if (seg) return seg;
    }
    return null;
  };
  const nearestWithPictures = (seg, y) => {
    if (Number(seg.dataset.pictures) > 0) return seg;
    let best = seg;
    let nearest = Infinity;
    for (const other of swap.querySelectorAll(".segment")) {
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
  const scopeOf = () => new URLSearchParams(read()?.scope || "");
  const fillSegments = () => {
    for (const seg of swap.querySelectorAll(".segment.held")) {
      const strip = seg.querySelector("[data-segment-strip]");
      if (!strip || strip.dataset.filled) continue;
      const box = seg.getBoundingClientRect();
      const cols = Math.max(1, Math.floor(box.width / TILE));
      const rows = Math.max(1, Math.floor(box.height / (TILE + 1)));
      strip.style.setProperty("--cols", String(cols));
      strip.style.setProperty("--tile", `${TILE}px`);
      const n = Math.min(400, cols * rows);
      strip.dataset.filled = String(n);
      const qs = scopeOf();
      qs.set("start", seg.dataset.at);
      qs.set("end", seg.dataset.end);
      qs.set("n", String(n));
      fetch(`/timeline/spread?${qs}`, { headers: { accept: "application/json" } })
        .then((r) => (r.ok ? r.json() : { pictures: [] }))
        .then((told) => {
          if (!strip.isConnected) return;
          strip.replaceChildren(
            ...told.pictures.map((p) => {
              const img = document.createElement("img");
              img.src = `/thumb/${p.slug}`;
              img.alt = "";
              img.loading = "lazy";
              img.draggable = false;
              img.dataset.moment = String(p.moment);
              return img;
            }),
          );
        });
    }
  };
  fillSegments();
  new MutationObserver(fillSegments).observe(swap, { childList: true });
  window.addEventListener("resize", () => {
    for (const s of swap.querySelectorAll("[data-segment-strip]")) delete s.dataset.filled;
    fillSegments();
  });

  // The hand a fraction of the way down a segment points at a picture by
  // RANK -- the k-th of its n in moment order, newest at the top -- never
  // by time: a burst of thousands in one minute would otherwise map every
  // position to its first or last. /timeline/nth answers it; one ask in
  // flight at a time, the newest always the one that lands.
  const rankAt = (seg, y) => {
    const box = seg.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (y - box.top) / (box.height || 1)));
    const n = Number(seg.dataset.pictures);
    return Math.min(n - 1, Math.max(0, Math.round((1 - f) * (n - 1))));
  };
  let asking = 0;
  const nth = (seg, y) => {
    const mine = ++asking;
    const qs = scopeOf();
    qs.set("start", seg.dataset.at);
    qs.set("end", seg.dataset.end);
    qs.set("k", String(rankAt(seg, y)));
    return fetch(`/timeline/nth?${qs}`, { headers: { accept: "application/json" } })
      .then((r) => (r.ok ? r.json() : null))
      .then((told) => (mine === asking ? told : null));
  };
  const peek = (seg, y) => {
    const card = swap.querySelector("[data-scrubber-peek]");
    if (!card) return;
    for (const was of swap.querySelectorAll(".segment-strip img.under")) was.classList.remove("under");
    if (!seg) {
      card.hidden = true;
      return;
    }
    const rail = swap.querySelector("[data-scrubber]").getBoundingClientRect();
    card.hidden = false;
    card.style.top = `${Math.min(rail.height - 60, Math.max(40, y - rail.top))}px`;
    const n = Number(seg.dataset.pictures);
    const img = card.querySelector("img");
    if (!n) {
      img.removeAttribute("src");
      img.hidden = true;
      card.querySelector(".scrubber-peek-label").textContent = seg.dataset.label;
      card.querySelector(".scrubber-peek-count").textContent = "nothing";
      return;
    }
    nth(seg, y).then((told) => {
      if (!told) return;
      img.src = `/thumb/${told.slug}`;
      img.hidden = false;
      card.querySelector(".scrubber-peek-label").textContent = told.spelled;
      card.querySelector(".scrubber-peek-count").textContent =
        `${(told.k + 1).toLocaleString()} of ${told.of.toLocaleString()}`;
      // the mosaic tile nearest that moment lights up
      let best = null,
        nearest = Infinity;
      for (const tile of seg.querySelectorAll(".segment-strip img[data-moment]")) {
        const d = Math.abs(Number(tile.dataset.moment) - told.moment);
        if (d < nearest) {
          nearest = d;
          best = tile;
        }
      }
      if (best) best.classList.add("under");
    });
  };

  let scrub = null; // {held, rail, pointer, x, y, moved}
  let scrubbed = false; // the click that ends a drag is not a click
  swap.addEventListener(
    "click",
    (e) => {
      if (scrubbed && e.target.closest("[data-scrubber]")) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
      scrubbed = false;
    },
    true,
  );
  swap.addEventListener("pointerdown", (event) => {
    const rail = event.target.closest("[data-scrubber]");
    const held = read();
    if (!rail || !held || event.button !== 0) return;
    // no capture yet: a click on a month must reach the month; the hand
    // decides by moving, and only then is the pointer held
    scrub = { held, rail, pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    event.preventDefault(); // a month is a link: no native link-drag, no text selection
  });
  swap.addEventListener("pointermove", (event) => {
    const rail = event.target.closest("[data-scrubber]");
    const seg = segmentAt(event.clientX, event.clientY);
    if (rail || scrub) peek(seg, event.clientY);
    if (!scrub) return;
    if (!scrub.moved && Math.abs(event.clientY - scrub.y) < 3) return;
    if (!scrub.moved) {
      scrub.moved = true;
      const held = scrub.rail.isConnected ? scrub.rail : swap.querySelector("[data-scrubber]");
      if (held) {
        held.setPointerCapture(scrub.pointer);
        held.dataset.dragging = "";
      }
    }
    if (!seg) return;
    const width = scrub.held.end - scrub.held.start;
    const target = nearestWithPictures(seg, event.clientY);
    const held = scrub.held;
    const land = (t) => {
      const end = Math.min(held.extentEnd, Math.max(held.extentStart + width, t));
      live(end - width, end, true);
    };
    // the window's newest edge lands on the picture the hand points at,
    // by rank within the segment; an empty segment hands on to the nearest
    if (target !== seg) {
      land(Number(target.dataset.end) - 1);
      return;
    }
    nth(seg, event.clientY).then((told) => {
      if (told && scrub) land(told.moment + 1);
    });
  });
  const unscrub = () => {
    if (!scrub) return;
    const was = scrub;
    scrub = null;
    scrubbed = was.moved;
    for (const rail of swap.querySelectorAll("[data-scrubber]")) delete rail.dataset.dragging;
    if (!was.moved) return;
    clearTimeout(liveTimer);
    const held = read();
    if (held) move(urlFor(Math.round(held.start), Math.round(held.end)), true);
  };
  window.addEventListener("pointerup", unscrub);
  window.addEventListener("pointercancel", unscrub);
  swap.addEventListener(
    "pointerleave",
    (e) => {
      if (!scrub && e.target.closest && e.target.closest("[data-scrubber]")) peek(null);
    },
    true,
  );

  // --- the size of the pictures ----------------------------------------------
  // ctrl+wheel or a pinch over the days resizes the rows, and the size is
  // the viewer's from then on; a plain wheel is the page's
  const ROW = { least: 120, most: 520, fallback: 200, key: "timeline.row" };
  const rowOf = () => {
    try {
      return Number(localStorage.getItem(ROW.key)) || ROW.fallback;
    } catch {
      return ROW.fallback;
    }
  };
  const sizeRows = (px) => {
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
      if (!(e.ctrlKey || e.metaKey) || !e.target.closest("[data-sessions]")) return;
      e.preventDefault();
      sizeRows(rowOf() * (e.deltaY > 0 ? 0.9 : 1.1));
    },
    { passive: false },
  );
  let pinch = null; // {distance, row}
  const apart = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  swap.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length === 2 && e.target.closest("[data-sessions]"))
        pinch = { distance: apart(e.touches), row: rowOf() };
    },
    { passive: true },
  );
  swap.addEventListener(
    "touchmove",
    (e) => {
      if (!pinch || e.touches.length !== 2) return;
      e.preventDefault();
      sizeRows(pinch.row * (apart(e.touches) / pinch.distance));
    },
    { passive: false },
  );
  swap.addEventListener("touchend", () => {
    pinch = null;
  });
})();
