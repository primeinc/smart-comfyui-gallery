// Interaction only. The surface is rendered by the server -- one window
// of the human timeline, its overview and brush, its bars, thumbnails
// and session cards (templates/_timeline_surface.html) -- and every
// move here is a request for that same fragment at another window,
// swapped in place with the URL updated. The brush maps pointer
// geometry onto the extent; a bar and a preset are links that work
// without this file. The story button drives the story routes and
// waits on the job feed.
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
    if (mine !== generation) return;
    if (!answer.ok) {
      const why = await answer.json().catch(() => ({}));
      const note = swap.querySelector("[data-note]");
      if (note) note.textContent = why.detail || answer.statusText;
      return;
    }
    swap.innerHTML = await answer.text();
    if (push) history.pushState({ url }, "", url);
    else history.replaceState({ url }, "", url);
  }

  window.addEventListener("popstate", (e) => {
    const url = (e.state && e.state.url) || location.pathname + location.search;
    move(url, false);
  });

  // a bar or a preset is a link to another window: swap instead of navigate
  swap.addEventListener("click", (e) => {
    const a = e.target.closest("[data-preset], [data-bin-window]");
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
  });
  const release = (event) => {
    if (!drag) return;
    const { start, end } = dragged(event);
    delete drag.overview.dataset.dragging;
    drag = null;
    move(urlFor(Math.round(start), Math.round(end)), true);
  };
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
