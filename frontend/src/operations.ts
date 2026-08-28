// The Operations Console client. The rows are the truth (db/inspecting.py),
// the ledger is history (db/ledger.py), /ws/events is transport. This
// script holds EVERY event it has been given -- never a sample -- and
// renders only the rows in view. Pausing pauses the painting; filtering
// hides rows; neither touches what is held. Ids are the order: a skipped
// id is a named gap, fetched from /operations/events, never papered over.
//
// Cold boot is a single dividing line. The page states the ledger head it
// read (`data-last-event-id`); everything at or below it is read over HTTP
// from the typed routes, and everything above it arrives on the socket,
// which is asked to resume from that same head. Nothing is serialized into
// the page for the browser to parse back out, so there is no window in
// which the embedded tape and the embedded head disagree.
import { api } from "./api";
import { closestFrom, everyElement, findElement, requireData, requireElement } from "./dom";
import { decodeFrame, type Frame, type ReadableEvent, type ReadablePendingFrame } from "./frames";
import type { components } from "./generated/api";

declare global {
  interface Window {
    // htmx ships no types and arrives as a <script> tag rather than an
    // import, so nothing else declares it. Optional: a page without htmx is
    // a check the caller makes, not a crash.
    htmx?: { process(root: Element): void };
  }
}

type Overview = components["schemas"]["Overview"];
type MatrixRow = components["schemas"]["MatrixRow"];
type Collapsed = components["schemas"]["Collapsed"];
type LiveReport = components["schemas"]["LiveReport"];

(() => {
  const root = requireElement(document, "[data-console]", HTMLElement);
  const ROW_H = 24;
  const OVERSCAN = 12;
  //: how many of the newest rows the cold read asks for
  const TAPE_COLD = 500;
  //: the backfill page: the server's ledger ceiling (db/ledger.py
  //: PAGE_MOST). The generated contract carries types, not values, so
  //: this copy is held by eye -- lowering PAGE_MOST means lowering this.
  const TAPE_PAGE = 2000;
  //: how long the console coalesces overview refreshes; the same order
  //: as reread.ts POLL_MS, but a different decision
  const RENDER_DEBOUNCE_MS = 400;

  // --- state ----------------------------------------------------------------
  const held: ReadableEvent[] = []; // every event, ascending by id
  const ids = new Set<number>();
  const head = Number(requireData(root, "lastEventId"));
  let lastId = head;
  let firstId = Number.POSITIVE_INFINITY;
  const pendingByJob = new Map<number, ReadablePendingFrame>();
  let paused = false;
  let heldWhilePaused = 0;
  let selectedJob: number | null = null;
  let selectedEvent: number | null = null;
  let socket: WebSocket | null = null;
  let retry = 0;
  //: the pending backoff reconnect, so the operator can overtake it
  let resuming = 0;
  let lastFrameAt: number | null = null;
  const filter = { type: "", severity: "", job: "" };
  let view: ReadableEvent[] = []; // the events that pass the filter, ascending

  // --- elements -------------------------------------------------------------
  const transport = requireElement(root, "[data-health-transport]", HTMLElement);
  const transportState = requireElement(root, "[data-transport-state]", HTMLElement);
  const transportLast = requireElement(root, "[data-transport-last]", HTMLElement);
  const transportAge = requireElement(root, "[data-transport-age]", HTMLElement);
  const matrixRows = requireElement(root, "[data-matrix-rows]", HTMLOListElement);
  const inspectorBody = requireElement(root, "[data-inspector-body]", HTMLElement);
  const inspectorHint = requireElement(root, "[data-inspector-hint]", HTMLElement);
  const scroller = requireElement(root, "[data-tape-scroll]", HTMLElement);
  const spacer = requireElement(root, "[data-tape-spacer]", HTMLElement);
  const rows = requireElement(root, "[data-tape-rows]", HTMLOListElement);
  const rawBody = requireElement(root, "[data-tape-raw-body]", HTMLPreElement);
  const countEl = requireElement(root, "[data-tape-count]", HTMLElement);
  const heldEl = requireElement(root, "[data-tape-held]", HTMLElement);
  const pauseBtn = requireElement(root, "[data-tape-pause]", HTMLButtonElement);
  const follow = requireElement(root, "[data-tape-autoscroll]", HTMLInputElement);
  const jobFilter = requireElement(root, "[data-tape-filter-job]", HTMLInputElement);

  // --- helpers --------------------------------------------------------------
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  function clock(epoch: number): string {
    const d = new Date(epoch * 1000);
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
  }
  function seconds(v: number | null): string {
    if (v == null) return "—";
    if (v < 60) return `${v.toFixed(1)}s`;
    if (v < 3600) return `${Math.floor(v / 60)}m ${pad(Math.floor(v % 60))}s`;
    return `${Math.floor(v / 3600)}h ${pad(Math.floor((v % 3600) / 60))}m`;
  }

  type Attrs = Readonly<Record<string, string | number | boolean | null | undefined>>;

  /** An element of the tag named, so the caller keeps the concrete type. */
  function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: Attrs, text?: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs ?? {})) {
      if (v === false || v == null) continue;
      node.setAttribute(k, v === true ? "" : String(v));
    }
    if (text != null) node.textContent = text;
    return node;
  }

  // --- ingestion (never drops) --------------------------------------------
  function ingest(event: ReadableEvent): boolean {
    if (ids.has(event.id)) return false;
    ids.add(event.id);
    const newest = held.at(-1);
    if (newest !== undefined && event.id < newest.id) {
      // an earlier page, or a gap fill: keep ascending order
      let i = held.length;
      while (i > 0) {
        const before = held[i - 1];
        if (before === undefined || before.id <= event.id) break;
        i--;
      }
      held.splice(i, 0, event);
    } else {
      held.push(event);
    }
    if (event.id > lastId) lastId = event.id;
    if (event.id < firstId) firstId = event.id;
    if (settles(event.type)) pendingByJob.delete(event.job_id);
    return true;
  }

  /** Whether an event of this type ends whatever the handler was reporting. */
  function settles(type: string): boolean {
    return !type.startsWith("phase.") && type !== "item.observed";
  }

  function passes(e: ReadableEvent): boolean {
    if (filter.type && !e.type.startsWith(filter.type)) return false;
    if (filter.severity === "warning" && e.severity === "info") return false;
    if (filter.severity === "error" && e.severity !== "error") return false;
    if (filter.job && String(e.job_id) !== filter.job) return false;
    return true;
  }

  /** How many places the held ids skip -- only inside what has been read. */
  function gaps(): number {
    let found = 0;
    let previous: number | null = null;
    for (const e of held) {
      if (previous !== null && e.id !== previous + 1) found++;
      previous = e.id;
    }
    return found;
  }

  const unfiltered = () => !filter.type && !filter.severity && !filter.job;

  // --- the tape ---------------------------------------------------------------
  function rebuildView(): void {
    view = held.filter(passes);
    const skipped = gaps();
    heldEl.hidden = skipped === 0;
    if (skipped) heldEl.textContent = `${skipped} gap(s) in the held ids — click a dashed row to fetch`;
    countEl.textContent = `${view.length} of ${held.length} shown${paused ? ` · paused, ${heldWhilePaused} new held` : ""}`;
    root.dataset.held = String(held.length);
    root.dataset.lastEventId = String(lastId);
    root.dataset.gaps = String(skipped);
  }

  function rowFor(e: ReadableEvent, isHead: boolean): HTMLLIElement {
    const li = el("li", {
      class: "tape-row",
      "data-event": e.id,
      "data-type": e.type,
      "data-severity": e.severity,
      "data-job": e.job_id,
      "data-condition": e.condition,
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

  function paint(): void {
    if (paused) return;
    const total = view.length;
    spacer.style.height = `${total * ROW_H}px`;
    const top = scroller.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
    const last = Math.min(total, Math.ceil((top + scroller.clientHeight) / ROW_H) + OVERSCAN);
    rows.style.transform = `translateY(${first * ROW_H}px)`;
    rows.textContent = "";
    const headId = held.at(-1)?.id;
    let previous = first > 0 ? view[first - 1] : undefined;
    for (const e of view.slice(first, last)) {
      if (previous !== undefined && unfiltered() && e.id !== previous.id + 1) {
        const after = previous;
        const gap = el(
          "li",
          { class: "tape-gap", role: "button", tabindex: "0" },
          `── ${e.id - after.id - 1} event(s) not held between #${after.id} and #${e.id} — fetch ──`,
        );
        gap.addEventListener("click", () => void fill(after.id, e.id));
        rows.appendChild(gap);
      }
      rows.appendChild(rowFor(e, e.id === headId));
      previous = e;
    }
  }

  function repaint(scrollToEnd: boolean): void {
    rebuildView();
    if (paused) return;
    paint();
    if (scrollToEnd && follow.checked) scroller.scrollTop = scroller.scrollHeight;
  }

  function select(e: ReadableEvent): void {
    selectedEvent = e.id;
    // The panel is hidden while it holds only its placeholder, so the
    // ledger has the width until there is something to show in it.
    rawBody.classList.remove("empty");
    rawBody.textContent = JSON.stringify(e, null, 2);
    for (const li of everyElement(rows, "[data-event]", HTMLLIElement)) {
      li.setAttribute("aria-selected", li.dataset.event === String(e.id) ? "true" : "false");
    }
  }

  scroller.addEventListener("scroll", () => {
    if (!paused) paint();
  });
  window.addEventListener("resize", () => {
    if (!paused) paint();
  });

  pauseBtn.addEventListener("click", () => {
    paused = !paused;
    pauseBtn.setAttribute("aria-pressed", String(paused));
    pauseBtn.textContent = paused ? "resume" : "pause";
    if (paused) {
      rebuildView();
    } else {
      heldWhilePaused = 0;
      repaint(true);
    }
  });
  requireElement(root, "[data-tape-filter-type]", HTMLSelectElement).addEventListener("change", (ev) => {
    if (ev.currentTarget instanceof HTMLSelectElement) filter.type = ev.currentTarget.value;
    repaint(true);
  });
  requireElement(root, "[data-tape-filter-severity]", HTMLSelectElement).addEventListener("change", (ev) => {
    if (ev.currentTarget instanceof HTMLSelectElement) filter.severity = ev.currentTarget.value;
    repaint(true);
  });
  jobFilter.addEventListener("input", () => {
    filter.job = jobFilter.value.trim();
    repaint(true);
  });
  requireElement(root, "[data-tape-earlier]", HTMLButtonElement).addEventListener("click", () => void earlier());

  // --- fetching what the rows hold ----------------------------------------
  async function fill(after: number, before: number): Promise<void> {
    // every id in (after, before): pages until caught up, never samples
    let cursor = after;
    while (cursor < before - 1) {
      const { data } = await api.GET("/operations/events", { params: { query: { after: cursor, limit: TAPE_PAGE } } });
      if (!data) return;
      let advanced = false;
      for (const e of data.events) {
        if (e.id >= before) break;
        ingest(e);
        cursor = e.id;
        advanced = true;
      }
      if (!advanced) break;
    }
    repaint(false);
  }

  async function earlier(): Promise<void> {
    if (!Number.isFinite(firstId)) return;
    const { data } = await api.GET("/operations/events/before", {
      params: { query: { before: firstId, limit: TAPE_COLD } },
    });
    if (!data) return;
    const keep = scroller.scrollHeight - scroller.scrollTop;
    for (const e of data.events) ingest(e);
    repaint(false);
    scroller.scrollTop = scroller.scrollHeight - keep;
  }

  // --- matrix + inspector ---------------------------------------------------
  let overviewTimer: number | null = null;
  function refreshOverviewSoon(): void {
    if (overviewTimer !== null) return;
    overviewTimer = window.setTimeout(() => {
      overviewTimer = null;
      void loadOverview();
    }, RENDER_DEBOUNCE_MS);
  }

  async function loadOverview(): Promise<void> {
    const { data } = await api.GET("/operations/overview");
    if (!data) return;
    paintHealth(data.overview);
    paintMatrix(data.matrix, data.collections);
  }

  function paintHealth(o: Overview): void {
    const say = (selector: string, text: string) => {
      const node = findElement(root, selector, HTMLElement);
      if (node) node.textContent = text;
    };
    const heartbeat = o.worker.heartbeat_age != null ? `${o.worker.heartbeat_age.toFixed(1)}s ago` : "none";
    // A worker off with an empty queue is somebody's decision; a worker
    // off with work QUEUED is a stall -- nothing is going to happen, and
    // that is the one thing this strip exists to say. Same four
    // conditions the cold render computes.
    const stalled = !o.worker.enabled && o.queue.queued > 0;
    const condition = o.worker.working ? "working" : stalled ? "stalled" : o.worker.enabled ? "idle" : "off";
    const workerCell = findElement(root, "[data-health-worker]", HTMLElement);
    if (workerCell) workerCell.dataset.workerCondition = condition;
    say(
      "[data-worker-state]",
      stalled
        ? `disabled — ${o.queue.queued} queued, nothing will run`
        : `${o.worker.enabled ? "enabled" : "disabled"} · ${o.worker.working ? "working" : "idle"} · thread ${o.worker.thread_alive ? "alive" : "not running"}`,
    );
    say(
      "[data-worker-raw]",
      `${o.worker.thread || "no thread"} · ${o.worker.owners.length ? o.worker.owners.join(", ") : "no owner"} · heartbeat ${heartbeat}`,
    );
    say("[data-queue-state]", `${o.queue.queued} queued · ${o.queue.running} running`);
    const oldest = o.queue.oldest_queued_age != null ? `${Math.round(o.queue.oldest_queued_age)}s` : "—";
    // `settled 24h {"done":6}` was a JSON dict rendered to a person.
    // The health strip is the one thing on this page a reader glances
    // at rather than studies, and a glance does not parse an object
    // literal. Same words the server renders on the cold page.
    const settled =
      Object.entries(o.queue.settled_24h)
        .map(([state, n]) => `${n} ${state}`)
        .join(", ") || "nothing";
    say("[data-queue-raw]", `oldest queued ${oldest} · settled 24h ${settled}`);
    say("[data-ledger-state]", `${o.ledger.events.toLocaleString()} events`);
    say("[data-ledger-raw]", `head #${o.ledger.last_id} · job_event · never sampled`);
    say("[data-coverage-files]", String(o.coverage.files));
    for (const node of everyElement(document, "[data-missing]", HTMLElement)) {
      const n = o.coverage.missing[requireData(node, "missing")];
      if (n != null) node.textContent = `${n} missing`;
    }
  }

  /**
   * The matrix, with every collection folded into one row.
   *
   * Eight rows where a person asked for one thing. Each was honest and
   * none was the answer: somebody watching a catch-up wants to know
   * whether the catch-up is going well, and eight rows made that a sum
   * they had to do themselves while the rows moved.
   *
   * The steps are still there, nested under the fold. Collapsing must
   * not be the same as hiding -- the step that FAILED is the row
   * somebody actually needs, and a console that swallowed it would have
   * made the page quieter and less useful at the same time.
   *
   * Open by default while the collection is running or has failed,
   * because those are the two states somebody is looking at it for.
   */
  function paintMatrix(jobs: MatrixRow[], collections: Collapsed[]): void {
    matrixRows.textContent = "";
    const grouped = new Set<number>();
    for (const group of collections) for (const id of group.steps) grouped.add(id);
    const byId = new Map(jobs.map((j) => [j.id, j]));

    for (const group of collections) {
      const holder = el("li", { class: "matrix-collection", "data-matrix-collection": group.name });
      const fold = el("details", {});
      if (group.state === "running" || group.state === "failed") fold.open = true;
      const head = el("summary", { class: "matrix-row", "data-collection-state": group.state });
      head.appendChild(el("span", { class: "matrix-id" }, `${group.steps.length} steps`));
      const kind = el("span", { class: "matrix-kind" });
      kind.appendChild(el("span", { class: "v" }, group.name));
      kind.appendChild(el("code", { class: "raw" }, `${group.settled}/${group.steps.length} settled`));
      head.appendChild(kind);
      head.appendChild(el("span", { class: "matrix-state", "data-state": group.state }, group.state));
      const bar = el("progress", { class: "matrix-progress" });
      if (group.total) {
        bar.value = group.done;
        bar.max = group.total;
      }
      head.appendChild(bar);
      head.appendChild(
        el("code", { class: "matrix-count" }, `${group.done}${group.total != null ? `/${group.total}` : ""}`),
      );
      // On the fold, because the collection is the unit somebody wants
      // to stop -- and a schedule can start one at 3am, so the thing to
      // stop may be something nobody started by hand.
      if (group.state === "running" || group.state === "queued") {
        const stop = el(
          "button",
          {
            type: "button",
            class: "matrix-stop",
            "data-stop-collection": group.name,
            title: "stop this collection: queued steps end now, a running one stops at its next item",
          },
          "stop",
        );
        stop.addEventListener("click", async (event) => {
          // The summary toggles the fold; stopping is not opening.
          event.preventDefault();
          event.stopPropagation();
          stop.disabled = true;
          await fetch(`/operations/collections/${encodeURIComponent(group.name)}/stop`, { method: "POST" });
        });
        head.appendChild(stop);
      }
      fold.appendChild(head);
      const steps = el("ol", { class: "matrix matrix-steps" });
      for (const id of group.steps) {
        const step = byId.get(id);
        if (step) steps.appendChild(matrixRow(step));
      }
      fold.appendChild(steps);
      holder.appendChild(fold);
      matrixRows.appendChild(holder);
    }

    for (const j of jobs) {
      if (grouped.has(j.id)) continue;
      matrixRows.appendChild(matrixRow(j));
    }
    wireMatrix();
  }

  function matrixRow(j: MatrixRow): HTMLElement {
    {
      const cancelling = j.derived.cancellation === "requested";
      const li = el("li", {
        class: "matrix-row",
        "data-matrix-job": j.id,
        "data-state": j.state,
        "data-cancelling": cancelling || null,
        tabindex: "0",
        role: "button",
        "aria-current": selectedJob === j.id ? "true" : null,
      });
      li.appendChild(el("span", { class: "matrix-id" }, `#${j.id}`));
      const kind = el("span", { class: "matrix-kind" });
      kind.appendChild(el("span", { class: "v" }, j.what || j.kind.replace(/_/g, " ")));
      kind.appendChild(el("code", { class: "raw" }, j.kind));
      li.appendChild(kind);
      li.appendChild(el("span", { class: "matrix-state", "data-state": j.state }, cancelling ? "cancelling" : j.state));
      const bar = el("progress", { class: "matrix-progress" });
      if (j.total) {
        bar.value = j.done_count;
        bar.max = j.total;
      }
      li.appendChild(bar);
      li.appendChild(
        el(
          "code",
          { class: "matrix-count" },
          `${j.done_count}${j.total != null ? `/${j.total}` : ""}${j.failed_count ? ` · ${j.failed_count} failed` : ""}`,
        ),
      );
      li.appendChild(
        el("code", { class: "matrix-exec" }, `a${j.attempt} f${j.fence ?? ""}${j.owner ? ` · ${j.owner}` : ""}`),
      );
      const live: ReadablePendingFrame | LiveReport | null = pendingByJob.get(j.id) ?? j.live;
      if (live && j.state === "running") {
        li.appendChild(el("span", { class: "matrix-live", "data-matrix-live": "" }, liveWords(live)));
      }
      return li;
    }
  }

  function liveWords(live: ReadablePendingFrame | LiveReport): string {
    return `${live.phase || live.type}${live.item_id != null ? ` · item ${live.item_id}` : ""}`;
  }

  function wireMatrix(): void {
    for (const li of everyElement(matrixRows, "[data-matrix-job]", HTMLLIElement)) {
      const jobId = Number(requireData(li, "matrixJob"));
      li.onclick = () => choose(jobId);
      li.onkeydown = (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          choose(jobId);
        }
      };
    }
  }

  // the inspector's own links: item pages load into the items slot; the
  // "every one" link filters the tape to the job instead of leaving the page.
  // These answer HTML fragments, not JSON, so they are `fetch` and not the
  // typed client -- the contract they honour is the template's, not the
  // document's.
  inspectorBody.addEventListener("click", (ev) => {
    const load = closestFrom(ev.target, "[data-items-load], [data-items-more]", HTMLAnchorElement);
    if (load) {
      ev.preventDefault();
      void loadItems(load);
      return;
    }
    const tapeFilter = closestFrom(ev.target, "[data-tape-job-filter]", HTMLElement);
    if (tapeFilter) {
      ev.preventDefault();
      jobFilter.value = requireData(tapeFilter, "tapeJobFilter");
      filter.job = jobFilter.value;
      repaint(true);
      scroller.scrollIntoView({ block: "start" });
    }
  });

  async function loadItems(link: HTMLAnchorElement): Promise<void> {
    const slot = findElement(inspectorBody, "[data-items-slot]", HTMLElement);
    if (!slot) return;
    const r = await fetch(link.href, { headers: { accept: "text/html" } });
    if (!r.ok) {
      slot.textContent = `${r.status}`;
      return;
    }
    const fragment = await r.text();
    if (link.hasAttribute("data-items-more")) {
      link.remove();
      slot.insertAdjacentHTML("beforeend", fragment);
    } else {
      slot.innerHTML = fragment;
    }
  }

  let inspectorTimer: number | null = null;
  async function loadInspector(): Promise<void> {
    const job = selectedJob;
    if (job == null) return;
    const r = await fetch(`/operations/job/${job}`, { headers: { accept: "text/html" } });
    if (!r.ok) {
      inspectorBody.textContent = "";
      inspectorBody.appendChild(el("p", { class: "empty" }, `job ${job}: ${r.status}`));
      return;
    }
    inspectorBody.innerHTML = await r.text();
    window.htmx?.process(inspectorBody);
    for (const node of everyElement(inspectorBody, "time[data-epoch]", HTMLTimeElement)) {
      const epoch = Number(requireData(node, "epoch"));
      const d = new Date(epoch * 1000);
      node.textContent = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${clock(epoch)}`;
      node.title = `epoch ${epoch}`;
    }
    inspectorHint.textContent = `job #${job} · refreshed ${clock(Date.now() / 1000)}`;
    paintPending();
  }
  function refreshInspectorSoon(): void {
    if (inspectorTimer !== null) return;
    inspectorTimer = window.setTimeout(() => {
      inspectorTimer = null;
      void loadInspector();
    }, 350);
  }
  function choose(jobId: number): void {
    selectedJob = jobId;
    for (const li of everyElement(matrixRows, "[data-matrix-job]", HTMLLIElement)) {
      li.setAttribute("aria-current", Number(li.dataset.matrixJob) === jobId ? "true" : "false");
    }
    void loadInspector();
  }
  function paintPending(): void {
    const slot = findElement(inspectorBody, "[data-current-phase]", HTMLElement);
    if (!slot) return;
    const p = selectedJob != null ? pendingByJob.get(selectedJob) : undefined;
    // the server already filled the slot from its live memory on a cold
    // load or a reconnect; only a fresher report replaces it
    if (!p) return;
    slot.textContent = "";
    slot.appendChild(el("span", { class: "v" }, p.phase || p.message || p.type));
    slot.appendChild(document.createTextNode(" "));
    slot.appendChild(el("code", { class: "raw" }, `${p.type} · ${p.message || ""} · live, not yet in the ledger`));
  }

  // --- the feed -----------------------------------------------------------
  function setTransport(state: string, text: string): void {
    transport.dataset.transport = state;
    transportState.textContent = text;
  }

  function connect(after: number): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const live = new WebSocket(`${proto}://${location.host}/ws/events?after=${after}`);
    socket = live;
    let unreadable = false;
    setTransport(retry ? "reconnecting" : "connecting", retry ? `reconnecting (${retry})` : "connecting");
    live.onopen = () => {
      retry = 0;
      setTransport("connected", "connected");
      refreshOverviewSoon();
    };
    live.onmessage = (msg: MessageEvent<unknown>) => {
      const frame = decodeFrame(msg.data);
      if (frame === null) {
        // An unreadable transport message is not evidence that a ledger row
        // may be discarded, and skipping it can lose rows nothing will ever
        // point at again: a malformed BACKLOG carries committed events whose
        // absence leaves no id gap if the socket then goes quiet. So this is
        // a transport failure. Close, and let the reconnect resume from the
        // last durable id held -- every committed row arrives again in the
        // next backlog. Live reports are ephemeral by design (they are not
        // rows) and one may be lost across the reconnect, which is what
        // "not yet in the ledger" means.
        unreadable = true;
        root.dataset.unreadableFrames = String(Number(root.dataset.unreadableFrames ?? 0) + 1);
        live.close();
        return;
      }
      lastFrameAt = Date.now();
      receive(frame);
    };
    live.onclose = () => {
      setTransport(
        unreadable ? "error" : "disconnected",
        unreadable ? `unreadable frame; resuming from #${lastId}` : "disconnected",
      );
      retry += 1;
      // resume from the newest id held, so nothing repeats and nothing is lost
      resuming = window.setTimeout(() => connect(lastId), Math.min(4000, 250 * 2 ** Math.min(retry, 4)));
    };
    live.onerror = () => live.close();
  }

  function receive(frame: Frame): void {
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
      const row = findElement(matrixRows, `[data-matrix-job="${frame.job_id}"]`, HTMLLIElement);
      if (row) {
        let slot = findElement(row, "[data-matrix-live]", HTMLElement);
        if (!slot) {
          slot = el("span", { class: "matrix-live", "data-matrix-live": "" });
          row.appendChild(slot);
        }
        slot.textContent = liveWords(frame);
      }
      return;
    }
    const before = held.at(-1)?.id ?? lastId;
    if (!ingest(frame)) return;
    if (paused) heldWhilePaused++;
    if (frame.id > before + 1 && before > 0) void fill(before, frame.id);
    repaint(true);
    transportLast.textContent = String(lastId);
    if (frame.job_id === selectedJob) refreshInspectorSoon();
    if (settles(frame.type)) refreshOverviewSoon();
  }

  // An operator's reconnect: close the socket; the close handler resumes
  // from the last id held, so nothing is repeated and nothing is lost.
  //
  // A closed socket is the case the button EXISTS for. It used to be the
  // one case it did nothing: `readyState <= 1` is open or connecting, so
  // pressing reconnect while the transport was down -- the only state an
  // operator would press it in -- fell through, and the console sat out
  // the rest of a backoff that reaches four seconds. So a dead socket
  // overtakes the pending resume instead: the wait is the recovery being
  // patient on its own, and a person asking is not that.
  requireElement(root, "[data-transport-reconnect]", HTMLButtonElement).addEventListener("click", () => {
    if (socket && socket.readyState <= 1) {
      socket.close();
      return;
    }
    window.clearTimeout(resuming);
    retry = 0;
    connect(lastId);
  });

  window.setInterval(() => {
    transportLast.textContent = String(lastId);
    transportAge.textContent = lastFrameAt
      ? `${((Date.now() - lastFrameAt) / 1000).toFixed(1)}s since last frame`
      : "no frame yet";
    for (const node of everyElement(inspectorBody, "[data-age-of]", HTMLElement)) {
      node.textContent = `${(Date.now() / 1000 - Number(requireData(node, "ageOf"))).toFixed(1)}s ago`;
    }
    for (const node of everyElement(inspectorBody, "[data-lease-until]", HTMLElement)) {
      const left = Number(requireData(node, "leaseUntil")) - Date.now() / 1000;
      node.textContent = left >= 0 ? `expires in ${seconds(left)}` : `expired ${seconds(-left)} ago · reclaimable`;
      node.classList.toggle("warn", left < 0);
    }
    for (const node of everyElement(inspectorBody, "[data-elapsed-from]", HTMLElement)) {
      const from = Number(requireData(node, "elapsedFrom"));
      if (!from || node.dataset.elapsedTo) continue;
      node.textContent = seconds(Date.now() / 1000 - from);
    }
  }, 1000);

  // --- boot: the rows first, then the feed ---------------------------------
  //
  // `head` is the dividing line the page read. The HTTP reads below carry
  // ids <= head; the socket is asked for ids > head, and anything committed
  // while those reads were in flight arrives in its backlog. The server
  // rendered the matrix and the strip already, so the page is usable before
  // any of this resolves -- these replace what it drew, they do not reveal it.
  async function cold(): Promise<void> {
    if (head <= 0) return;
    const { data } = await api.GET("/operations/events/before", {
      params: { query: { before: head + 1, limit: TAPE_COLD } },
    });
    if (!data) return;
    for (const e of data.events) ingest(e);
    repaint(true);
  }

  wireMatrix();
  repaint(true);
  // the feed starts whether or not the cold reads landed: a console with no
  // tape still has to show what happens next
  Promise.allSettled([loadOverview(), cold()]).then(() => connect(head), console.error);
})();
