// Interaction only. The surface is rendered by the server -- one window
// of the human timeline: the axis with its frames and pictures, the rule
// with its brush, the body (templates/_timeline_surface.html) -- and
// every move here is a request for that same fragment at another
// window, swapped in place with the URL updated. The brush and the
// axis map pointer geometry onto time; a bar and a preset are links
// that work without this file. The story button drives the story
// routes and waits on the job feed.
(() => {
  "use strict";

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

  // the window's URL: the scope's own parameters ride every move
  const urlFor = (start, end) => {
    const qs = new URLSearchParams(read()?.scope || "");
    qs.set("start", String(start));
    qs.set("end", String(end));
    return `/timeline?${qs}`;
  };

  let generation = 0;
  async function move(url, push) {
    const mine = ++generation;
    const answer = await fetch(url, { headers: { "hx-request": "true", accept: "text/html" } });
    if (mine !== generation) return; // a newer move superseded this one
    if (!answer.ok) {
      const why = await answer.json().catch(() => ({}));
      const note = swap.querySelector("[data-note]");
      if (note) note.textContent = why.detail || answer.statusText;
      return;
    }
    const held = swap.querySelector("[data-strip]");
    if (held) held.dataset.settling = "";
    swap.innerHTML = await answer.text();
    thin();
    if (push === true) history.pushState({ url }, "", url);
    else if (push === false) history.replaceState({ url }, "", url);
    // push === null: a refresh of the same window; the URL already says it
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
      if (left < edge) { a.hidden = true; continue; }
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
  const live = (start, end) => {
    const now = performance.now();
    clearTimeout(liveTimer);
    const run = () => { liveAt = performance.now(); move(urlFor(Math.round(start), Math.round(end)), false); };
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
        if (["done", "failed", "cancelled"].includes(frame.state) && ["context", "events", "ingest", "scan", "faces", "cluster"].includes(frame.kind)) {
          move(location.pathname + location.search, null);
        }
      };
      socket.onclose = () => setTimeout(open, 2000);
    };
    open();
  })();

  window.addEventListener("popstate", (e) => {
    const url = (e.state && e.state.url) || location.pathname + location.search;
    move(url, false);
  });

  // a bar or a preset is a link to another window: swap instead of navigate
  swap.addEventListener("click", (e) => {
    const a = e.target.closest("[data-preset], [data-bin-window], [data-month-window]");
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return; // a modified click is the browser's
    e.preventDefault();
    move(a.getAttribute("href"), true);
  });

  // --- the brush ---------------------------------------------------------
  const ox = (held, t) => ((t - held.extentStart) / Math.max(1, held.extentEnd - held.extentStart)) * W;
  const ot = (held, x) => held.extentStart + (Math.min(W, Math.max(0, x)) / W) * (held.extentEnd - held.extentStart);
  const overviewX = (overview, event) => {
    const box = overview.getBoundingClientRect();
    return ((event.clientX - box.left) / box.width) * W;
  };
  const placeBrush = (overview, held, start, end) => {
    const x0 = ox(held, start);
    const x1 = ox(held, end);
    overview.querySelector("[data-brush]").setAttribute("x", x0);
    overview.querySelector("[data-brush]").setAttribute("width", Math.max(2, x1 - x0));
    overview.querySelector('[data-brush-edge="start"]').setAttribute("x", x0 - 3);
    overview.querySelector('[data-brush-edge="end"]').setAttribute("x", x1 - 3);
  };

  let drag = null; // {overview, held, mode, at}
  const dragged = (event) => {
    const { held, mode, at } = drag;
    const x = overviewX(drag.overview, event);
    const dt = ot(held, x) - ot(held, at);
    // a window is never narrower than an hour, or than the library itself
    const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
    let start = held.start;
    let end = held.end;
    if (mode === "move") {
      const width = end - start;
      start = Math.min(Math.max(held.extentStart, start + dt), held.extentEnd - width);
      end = start + width;
    } else if (mode === "start") {
      start = Math.max(held.extentStart, Math.min(start + dt, end - narrowest));
    } else if (mode === "end") {
      end = Math.min(held.extentEnd, Math.max(end + dt, start + narrowest));
    } else {
      const a = ot(held, at);
      const b = ot(held, x);
      start = Math.max(held.extentStart, Math.min(a, b));
      end = Math.min(held.extentEnd, Math.max(a, b, start + narrowest));
    }
    return { start, end };
  };

  swap.addEventListener("pointerdown", (event) => {
    const overview = event.target.closest("[data-overview]");
    const held = read();
    if (!overview || !held) return;
    const x = overviewX(overview, event);
    const x0 = ox(held, held.start);
    const x1 = ox(held, held.end);
    const grip = 8;
    let mode = "new";
    if (Math.abs(x - x0) <= grip) mode = "start";
    else if (Math.abs(x - x1) <= grip) mode = "end";
    else if (x > x0 && x < x1) mode = "move";
    drag = { overview, held, mode, at: x };
    overview.setPointerCapture(event.pointerId);
    overview.dataset.dragging = mode;
    event.preventDefault();
  });
  swap.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const { start, end } = dragged(event);
    placeBrush(drag.overview, drag.held, start, end);
    live(start, end);
  });
  const release = (event) => {
    if (!drag) return;
    const { start, end } = dragged(event);
    delete drag.overview.dataset.dragging;
    drag = null;
    clearTimeout(liveTimer);
    move(urlFor(Math.round(start), Math.round(end)), true);
  };
  // the brush swaps the surface under the pointer: keep dragging the new one
  swap.addEventListener("pointermove", (event) => {
    if (drag && !drag.overview.isConnected) {
      const overview = swap.querySelector("[data-overview]");
      if (overview) { drag.overview = overview; overview.setPointerCapture(event.pointerId); }
    }
  }, true);

  // the axis pans under the hand: a drag moves the window, a click is a
  // click (a bar is a link) -- the hand decides by moving
  let pan = null; // {x, start, end, moved}
  swap.addEventListener("pointerdown", (event) => {
    const axis = event.target.closest("[data-strip]");
    const held = read();
    if (!axis || !held || event.button !== 0) return;
    // the axis is swapped under the hand while it moves: its width is
    // read once here, never from an element that may since be detached
    pan = { axis, px: axis.getBoundingClientRect().width || 1, x: event.clientX, start: held.start, end: held.end, moved: false, held };
  });
  swap.addEventListener("pointermove", (event) => {
    if (!pan) return;
    if (!pan.moved && Math.abs(event.clientX - pan.x) < 4) return;
    if (!pan.moved) { pan.moved = true; pan.axis.dataset.dragging = ""; pan.axis.setPointerCapture(event.pointerId); }
    const width = pan.end - pan.start;
    const dt = ((pan.x - event.clientX) / pan.px) * width;
    let start = Math.max(pan.held.extentStart, pan.start + dt);
    start = Math.min(start, pan.held.extentEnd - width);
    live(start, start + width);
  });
  let panned = false; // the click that ends a drag is not a click
  swap.addEventListener("click", (e) => {
    if (panned && e.target.closest("[data-strip]")) { e.preventDefault(); e.stopImmediatePropagation(); }
    panned = false;
  }, true);
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
  swap.addEventListener("pointerup", unpan);
  swap.addEventListener("pointercancel", unpan);

  // ctrl+wheel over the axis or the rule zooms around the cursor; shift+wheel
  // pans; a plain wheel is the page's, so the river below stays reachable
  swap.addEventListener("wheel", (e) => {
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
      const step = (e.deltaY > 0 ? 1 : -1) * width / 5;
      start = held.start + step;
      end = held.end + step;
    } else {
      const factor = e.deltaY > 0 ? 1.25 : 0.8;
      start = at - (at - held.start) * factor;
      end = at + (held.end - at) * factor;
    }
    const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
    if (end - start < narrowest) { start = at - narrowest / 2; end = at + narrowest / 2; }
    start = Math.max(held.extentStart, start);
    end = Math.min(held.extentEnd, Math.max(end, start + narrowest));
    live(start, end);
  }, { passive: false });
  swap.addEventListener("pointerup", release);
  swap.addEventListener("pointercancel", release);

  swap.addEventListener("keydown", (e) => {
    if (!e.target.closest("[data-overview]")) return;
    const held = read();
    if (!held) return;
    const width = held.end - held.start;
    const step = width / 4;
    const go = (s, t) => { e.preventDefault(); move(urlFor(Math.round(s), Math.round(t)), true); };
    if (e.key === "ArrowLeft") go(Math.max(held.extentStart, held.start - step), Math.max(held.extentStart + width, held.end - step));
    if (e.key === "ArrowRight") go(Math.min(held.extentEnd - width, held.start + step), Math.min(held.extentEnd, held.end + step));
    if (e.key === "+" || e.key === "=") go(held.start + width / 4, held.end - width / 4);
    if (e.key === "-") go(Math.max(held.extentStart, held.start - width / 2), Math.min(held.extentEnd, held.end + width / 2));
  });

  // --- the story button ----------------------------------------------------
  function settled(jobId) {
    // the job feed: a snapshot, then every committed delta; resolve on
    // the job's terminal state, or on a snapshot that already holds it
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${proto}://${location.host}/ws/jobs`);
      const terminal = new Set(["done", "failed", "cancelled"]);
      const finish = (state) => { socket.close(); resolve(state); };
      socket.onmessage = (msg) => {
        const frame = JSON.parse(msg.data);
        if (frame.type === "snapshot") {
          const held = frame.jobs.find((j) => j.id === jobId);
          if (held && terminal.has(held.state)) finish(held.state);
          if (!held) fetch(`/jobs/${jobId}`, { headers: { accept: "application/json" } }).then((r) => r.json()).then((j) => { if (terminal.has(j.state)) finish(j.state); });
          return;
        }
        if (frame.job === jobId && terminal.has(frame.state)) finish(frame.state);
      };
      socket.onerror = () => reject(new Error("the job feed closed"));
    });
  }

  async function tell(eventId, planner, status) {
    // freeze -> plan -> render, each a story route; a refusal is shown verbatim
    const post = async (where, body) => {
      const r = await fetch(where, { method: "POST", headers: { "content-type": "application/json", accept: "application/json" }, body: JSON.stringify(body) });
      const told = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(told.detail || r.statusText);
      return { status: r.status, told };
    };
    try {
      status.textContent = "freezing…";
      const snap = await post("/stories/snapshots", { event_id: eventId });
      status.textContent = "planning…";
      let plan = await post("/stories/plans", { snapshot_id: snap.told.id, planner });
      if (plan.status === 202) {
        // durable work: the job feed says when it settles (no polling),
        // then the same request answers 200 with the plan it made
        status.textContent = `planning as job #${plan.told.job.id}…`;
        const state = await settled(plan.told.job.id);
        if (state !== "done") throw new Error(`the planning job ${state}; see /operations`);
        plan = await post("/stories/plans", { snapshot_id: snap.told.id, planner });
        if (plan.status !== 200) throw new Error("the plan job settled but no plan answers; see /operations");
      }
      status.textContent = "rendering…";
      const made = await post("/stories/renders", { plan_id: plan.told.plan_id });
      status.textContent = "";
      window.location.href = `/stories/renders/${made.told.id}`;
    } catch (why) {
      status.textContent = why.message;
    }
  }

  swap.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-session-tell]");
    if (!btn) return;
    const id = Number(btn.dataset.sessionTell);
    const status = swap.querySelector(`[data-session-status="${id}"]`);
    tell(id, btn.dataset.sessionPlanner, status);
  });
})();
