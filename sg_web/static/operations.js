// The Operations Console client. The rows are the truth (db/inspecting.py),
// the ledger is history (db/ledger.py), /ws/events is transport. This
// script holds EVERY event it has been given -- never a sample -- and
// renders only the rows in view. Pausing pauses the painting; filtering
// hides rows; neither touches what is held. Ids are the order: a skipped
// id is a named gap, fetched from /operations/events, never papered over.
(function () {
  const root = document.querySelector("[data-console]");
  if (!root) return;
  const ROW_H = 24;
  const OVERSCAN = 12;
  const q = (sel, el) => (el || root).querySelector(sel);

  // --- state ----------------------------------------------------------------
  const held = []; // every event, ascending by id
  const ids = new Set();
  let lastId = Number(root.dataset.lastEventId || 0);
  let firstId = Infinity;
  const pendingByJob = new Map(); // job_id -> latest pending report
  let paused = false;
  let heldWhilePaused = 0;
  let selectedJob = null;
  let selectedEvent = null;
  let socket = null;
  let retry = 0;
  let lastFrameAt = null;
  const filter = { type: "", severity: "", job: "" };
  let view = []; // indexes into `held` after filtering

  // --- elements -------------------------------------------------------------
  const transport = q("[data-health-transport]");
  const transportState = q("[data-transport-state]");
  const transportLast = q("[data-transport-last]");
  const transportAge = q("[data-transport-age]");
  const matrixRows = q("[data-matrix-rows]");
  const inspectorBody = q("[data-inspector-body]");
  const inspectorHint = q("[data-inspector-hint]");
  const scroller = q("[data-tape-scroll]");
  const spacer = q("[data-tape-spacer]");
  const rows = q("[data-tape-rows]");
  const rawBody = q("[data-tape-raw-body]");
  const countEl = q("[data-tape-count]");
  const heldEl = q("[data-tape-held]");
  const pauseBtn = q("[data-tape-pause]");
  const follow = q("[data-tape-autoscroll]");

  // --- helpers --------------------------------------------------------------
  const pad = (n, w) => String(n).padStart(w || 2, "0");
  function clock(epoch) {
    const d = new Date(epoch * 1000);
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
  }
  function seconds(v) {
    if (v == null) return "—";
    if (v < 60) return `${v.toFixed(1)}s`;
    if (v < 3600) return `${Math.floor(v / 60)}m ${pad(Math.floor(v % 60))}s`;
    return `${Math.floor(v / 3600)}h ${pad(Math.floor((v % 3600) / 60))}m`;
  }
  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      node.setAttribute(k, v === true ? "" : v);
    }
    if (text != null) node.textContent = text;
    return node;
  }

  // --- ingestion (never drops) --------------------------------------------
  function ingest(event) {
    if (event.id == null || ids.has(event.id)) return false;
    ids.add(event.id);
    if (held.length && event.id < held[held.length - 1].id) {
      // an earlier page, or a gap fill: keep ascending order
      let i = held.length;
      while (i > 0 && held[i - 1].id > event.id) i--;
      held.splice(i, 0, event);
    } else {
      held.push(event);
    }
    if (event.id > lastId) lastId = event.id;
    if (event.id < firstId) firstId = event.id;
    if (event.job_id != null && event.type && !event.type.startsWith("phase.") && event.type !== "item.observed") {
      pendingByJob.delete(event.job_id);
    }
    return true;
  }

  function passes(e) {
    if (filter.type && !e.type.startsWith(filter.type)) return false;
    if (filter.severity === "warning" && e.severity === "info") return false;
    if (filter.severity === "error" && e.severity !== "error") return false;
    if (filter.job && String(e.job_id) !== String(filter.job)) return false;
    return true;
  }

  function gaps() {
    // [afterId, beforeId] pairs where ids skip -- only meaningful inside
    // the range this page has read contiguously, so we report skips
    // between consecutive held events.
    const found = [];
    for (let i = 1; i < held.length; i++) {
      if (held[i].id !== held[i - 1].id + 1) found.push([held[i - 1].id, held[i].id]);
    }
    return found;
  }

  // --- the tape ---------------------------------------------------------------
  function rebuildView() {
    view = [];
    for (let i = 0; i < held.length; i++) if (passes(held[i])) view.push(i);
    const skipped = gaps();
    heldEl.hidden = skipped.length === 0;
    if (skipped.length) heldEl.textContent = `${skipped.length} gap(s) in the held ids — click a dashed row to fetch`;
    countEl.textContent = `${view.length} of ${held.length} shown` + (paused ? ` · paused, ${heldWhilePaused} new held` : "");
    root.dataset.held = String(held.length);
    root.dataset.lastEventId = String(lastId);
    root.dataset.gaps = String(skipped.length);
  }

  function rowFor(e, isHead) {
    const li = el("li", {
      class: "tape-row",
      "data-event": e.id,
      "data-type": e.type,
      "data-severity": e.severity,
      "data-job": e.job_id,
      "data-condition": e.condition || null,
      "data-head": isHead || null,
      "aria-selected": selectedEvent === e.id ? "true" : "false",
      role: "option",
    });
    li.appendChild(el("span", { class: "tape-id" }, `#${e.id}`));
    li.appendChild(el("span", { class: "tape-at" }, clock(e.at)));
    li.appendChild(el("span", { class: "tape-type" }, e.type));
    li.appendChild(el("span", { class: "tape-job" }, `job ${e.job_id}${e.item_id != null ? ` · ${e.item_id}` : ""}`));
    li.appendChild(el("span", { class: "tape-text", title: e.text }, e.text));
    li.addEventListener("click", () => select(e));
    return li;
  }

  function paint() {
    if (paused) return;
    const total = view.length;
    spacer.style.height = `${total * ROW_H}px`;
    const top = scroller.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
    const last = Math.min(total, Math.ceil((top + scroller.clientHeight) / ROW_H) + OVERSCAN);
    rows.style.transform = `translateY(${first * ROW_H}px)`;
    rows.textContent = "";
    const headId = held.length ? held[held.length - 1].id : null;
    for (let i = first; i < last; i++) {
      const e = held[view[i]];
      const previous = i > 0 ? held[view[i - 1]] : null;
      if (previous && !filter.type && !filter.severity && !filter.job && e.id !== previous.id + 1) {
        const gap = el("li", { class: "tape-gap", role: "button", tabindex: "0" }, `── ${e.id - previous.id - 1} event(s) not held between #${previous.id} and #${e.id} — fetch ──`);
        gap.addEventListener("click", () => fill(previous.id, e.id));
        rows.appendChild(gap);
      }
      rows.appendChild(rowFor(e, e.id === headId));
    }
  }

  function repaint(scrollToEnd) {
    rebuildView();
    if (paused) return;
    paint();
    if (scrollToEnd && follow.checked) scroller.scrollTop = scroller.scrollHeight;
  }

  function select(e) {
    selectedEvent = e.id;
    rawBody.textContent = JSON.stringify(e, null, 2);
    for (const li of rows.children) li.setAttribute("aria-selected", li.dataset.event === String(e.id) ? "true" : "false");
  }

  scroller.addEventListener("scroll", () => { if (!paused) paint(); });
  window.addEventListener("resize", () => { if (!paused) paint(); });

  pauseBtn.addEventListener("click", () => {
    paused = !paused;
    pauseBtn.setAttribute("aria-pressed", String(paused));
    pauseBtn.textContent = paused ? "resume" : "pause";
    if (!paused) { heldWhilePaused = 0; repaint(true); } else { rebuildView(); }
  });
  q("[data-tape-filter-type]").addEventListener("change", (ev) => { filter.type = ev.target.value; repaint(true); });
  q("[data-tape-filter-severity]").addEventListener("change", (ev) => { filter.severity = ev.target.value; repaint(true); });
  q("[data-tape-filter-job]").addEventListener("input", (ev) => { filter.job = ev.target.value.trim(); repaint(true); });
  q("[data-tape-earlier]").addEventListener("click", earlier);

  // --- fetching what the rows hold ----------------------------------------
  async function fill(after, before) {
    // every id in (after, before): pages until caught up, never samples
    let cursor = after;
    while (cursor < before - 1) {
      const r = await fetch(`/operations/events?after=${cursor}&limit=2000`, { headers: { accept: "application/json" } });
      if (!r.ok) return;
      const told = await r.json();
      let advanced = false;
      for (const e of told.events) {
        if (e.id >= before) break;
        ingest(e);
        cursor = e.id;
        advanced = true;
      }
      if (!advanced) break;
    }
    repaint(false);
  }

  async function earlier() {
    if (!isFinite(firstId)) return;
    const r = await fetch(`/operations/events/before?before=${firstId}&limit=500`, { headers: { accept: "application/json" } });
    if (!r.ok) return;
    const told = await r.json();
    const keep = scroller.scrollHeight - scroller.scrollTop;
    for (const e of told.events) ingest(e);
    repaint(false);
    scroller.scrollTop = scroller.scrollHeight - keep;
  }

  // --- matrix + inspector ---------------------------------------------------
  let overviewTimer = null;
  function refreshOverviewSoon() {
    if (overviewTimer) return;
    overviewTimer = setTimeout(async () => {
      overviewTimer = null;
      const r = await fetch("/operations/overview", { headers: { accept: "application/json" } });
      if (!r.ok) return;
      const told = await r.json();
      paintHealth(told.overview);
      paintMatrix(told.matrix);
    }, 400);
  }

  function paintHealth(o) {
    q("[data-worker-state]").textContent = `${o.worker.enabled ? "enabled" : "disabled"} · ${o.worker.working ? "working" : "idle"}`;
    q("[data-worker-raw]").textContent = `${o.worker.owners.length ? o.worker.owners.join(", ") : "no owner"} · heartbeat ${o.worker.heartbeat_age != null ? o.worker.heartbeat_age.toFixed(1) + "s ago" : "none"}`;
    q("[data-queue-state]").textContent = `${o.queue.queued} queued · ${o.queue.running} running`;
    q("[data-queue-raw]").textContent = `oldest queued ${o.queue.oldest_queued_age != null ? Math.round(o.queue.oldest_queued_age) + "s" : "—"} · settled 24h ${JSON.stringify(o.queue.settled_24h)}`;
    q("[data-ledger-state]").textContent = `${o.ledger.events.toLocaleString()} events`;
    q("[data-ledger-raw]").textContent = `head #${o.ledger.last_id} · job_event · never sampled`;
  }

  function paintMatrix(jobs) {
    matrixRows.textContent = "";
    for (const j of jobs) {
      const cancelling = j.derived.cancellation === "requested";
      const li = el("li", { class: "matrix-row", "data-matrix-job": j.id, "data-state": j.state, "data-cancelling": cancelling || null, tabindex: "0", role: "button", "aria-current": selectedJob === j.id ? "true" : null });
      li.appendChild(el("span", { class: "matrix-id" }, `#${j.id}`));
      const kind = el("span", { class: "matrix-kind" });
      kind.appendChild(el("span", { class: "v" }, j.kind.replace(/_/g, " ")));
      kind.appendChild(el("code", { class: "raw" }, j.kind));
      li.appendChild(kind);
      li.appendChild(el("span", { class: "matrix-state", "data-state": j.state }, cancelling ? "cancelling" : j.state));
      const bar = el("progress", { class: "matrix-progress" });
      if (j.total) { bar.value = j.done_count; bar.max = j.total; }
      li.appendChild(bar);
      li.appendChild(el("code", { class: "matrix-count" }, `${j.done_count}${j.total != null ? "/" + j.total : ""}${j.failed_count ? " · " + j.failed_count + " failed" : ""}`));
      li.appendChild(el("code", { class: "matrix-exec" }, `a${j.attempt} f${j.fence ?? ""}${j.owner ? " · " + j.owner : ""}`));
      matrixRows.appendChild(li);
    }
    wireMatrix();
  }

  function wireMatrix() {
    for (const li of matrixRows.querySelectorAll("[data-matrix-job]")) {
      li.onclick = () => choose(Number(li.dataset.matrixJob));
      li.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); choose(Number(li.dataset.matrixJob)); } };
    }
  }

  // the inspector's own links: item pages load into the items slot; the
  // "every one" link filters the tape to the job instead of leaving the page
  inspectorBody.addEventListener("click", async (ev) => {
    const load = ev.target.closest("[data-items-load], [data-items-more]");
    if (load) {
      ev.preventDefault();
      const slot = q("[data-items-slot]", inspectorBody);
      if (!slot) return;
      const r = await fetch(load.getAttribute("href"), { headers: { accept: "text/html" } });
      if (!r.ok) { slot.textContent = `${r.status}`; return; }
      if (load.hasAttribute("data-items-more")) {
        load.remove();
        slot.insertAdjacentHTML("beforeend", await r.text());
      } else {
        slot.innerHTML = await r.text();
      }
      return;
    }
    const tapeFilter = ev.target.closest("[data-tape-job-filter]");
    if (tapeFilter) {
      ev.preventDefault();
      const input = q("[data-tape-filter-job]");
      input.value = tapeFilter.dataset.tapeJobFilter;
      filter.job = input.value;
      repaint(true);
      scroller.scrollIntoView({ block: "start" });
    }
  });

  let inspectorTimer = null;
  async function loadInspector() {
    if (selectedJob == null) return;
    const r = await fetch(`/operations/job/${selectedJob}`, { headers: { accept: "text/html" } });
    if (!r.ok) { inspectorBody.innerHTML = ""; inspectorBody.appendChild(el("p", { class: "empty" }, `job ${selectedJob}: ${r.status}`)); return; }
    inspectorBody.innerHTML = await r.text();
    if (window.htmx) window.htmx.process(inspectorBody);
    for (const node of inspectorBody.querySelectorAll("time[data-epoch]")) {
      const epoch = Number(node.dataset.epoch);
      const d = new Date(epoch * 1000);
      node.textContent = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${clock(epoch)}`;
      node.title = `epoch ${epoch}`;
    }
    inspectorHint.textContent = `job #${selectedJob} · refreshed ${clock(Date.now() / 1000)}`;
    paintPending();
  }
  function refreshInspectorSoon() {
    if (inspectorTimer) return;
    inspectorTimer = setTimeout(() => { inspectorTimer = null; loadInspector(); }, 350);
  }
  function choose(jobId) {
    selectedJob = jobId;
    for (const li of matrixRows.querySelectorAll("[data-matrix-job]")) {
      li.setAttribute("aria-current", Number(li.dataset.matrixJob) === jobId ? "true" : "false");
    }
    loadInspector();
  }
  function paintPending() {
    const slot = q("[data-current-phase]", inspectorBody);
    if (!slot) return;
    const p = selectedJob != null ? pendingByJob.get(selectedJob) : null;
    // the server already filled the slot from its live memory on a cold
    // load or a reconnect; only a fresher report replaces it
    if (!p) return;
    slot.textContent = "";
    slot.appendChild(el("span", { class: "v" }, p.phase || p.message || p.type));
    slot.appendChild(document.createTextNode(" "));
    slot.appendChild(el("code", { class: "raw" }, `${p.type} · ${p.message || ""} · live, not yet in the ledger`));
  }

  // --- the feed -----------------------------------------------------------
  function setTransport(state, text) {
    transport.dataset.transport = state;
    transportState.textContent = text;
  }
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws/events?after=${lastId}`);
    setTransport(retry ? "reconnecting" : "connecting", retry ? `reconnecting (${retry})` : "connecting");
    socket.onopen = () => { retry = 0; setTransport("connected", "connected"); refreshOverviewSoon(); };
    socket.onmessage = (msg) => {
      const frame = JSON.parse(msg.data);
      lastFrameAt = Date.now();
      if (frame.frame === "backlog") {
        let added = 0;
        for (const e of frame.events) if (ingest(e)) added++;
        if (paused) heldWhilePaused += added;
        repaint(true);
        return;
      }
      if (frame.frame === "pending") {
        pendingByJob.set(frame.job_id, frame);
        if (frame.job_id === selectedJob) paintPending();
        return;
      }
      if (frame.frame === "event") {
        const before = held.length ? held[held.length - 1].id : lastId;
        if (ingest(frame)) {
          if (paused) heldWhilePaused++;
          if (frame.id > before + 1 && before > 0) fill(before, frame.id);
          repaint(true);
          transportLast.textContent = String(lastId);
          if (frame.job_id === selectedJob) refreshInspectorSoon();
          if (!frame.type.startsWith("phase.") && frame.type !== "item.observed") refreshOverviewSoon();
        }
      }
    };
    socket.onclose = () => {
      setTransport("disconnected", "disconnected");
      retry += 1;
      setTimeout(connect, Math.min(4000, 250 * 2 ** Math.min(retry, 4)));
    };
    socket.onerror = () => socket.close();
  }

  // An operator's reconnect: close the socket; the close handler resumes
  // from the last id held, so nothing is repeated and nothing is lost.
  q("[data-transport-reconnect]").addEventListener("click", () => {
    if (socket && socket.readyState <= 1) socket.close();
  });

  setInterval(() => {
    transportLast.textContent = String(lastId);
    transportAge.textContent = lastFrameAt ? `${((Date.now() - lastFrameAt) / 1000).toFixed(1)}s since last frame` : "no frame yet";
    for (const node of inspectorBody.querySelectorAll("[data-age-of]")) {
      node.textContent = `${((Date.now() / 1000) - Number(node.dataset.ageOf)).toFixed(1)}s ago`;
    }
    for (const node of inspectorBody.querySelectorAll("[data-lease-until]")) {
      const left = Number(node.dataset.leaseUntil) - Date.now() / 1000;
      node.textContent = left >= 0 ? `expires in ${seconds(left)}` : `expired ${seconds(-left)} ago · reclaimable`;
      node.classList.toggle("warn", left < 0);
    }
    for (const node of inspectorBody.querySelectorAll("[data-elapsed-from]")) {
      const from = Number(node.dataset.elapsedFrom);
      if (!from || node.dataset.elapsedTo) continue;
      node.textContent = seconds(Date.now() / 1000 - from);
    }
  }, 1000);

  // --- boot: the rows first, then the feed ---------------------------------
  const state = JSON.parse(q("[data-console-state]").textContent);
  const cold = JSON.parse(q("[data-console-tape]").textContent);
  for (const e of cold) ingest(e);
  repaint(true);
  paintMatrix(state.matrix);
  connect();
})();
