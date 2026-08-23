// The timeline surface: density per bin at the current zoom, drawn from
// /timeline/density -- the page never counts anything itself. The URL
// owns the zoom (?bin=&start=&end=), so reload, back, forward and a
// pasted link all land where the person was. Click a bar to zoom into
// it (week -> day -> hour -> quarter -> minute); at the finest zoom a bar
// opens the gallery on exactly its pictures, ordered by the moment.
// Coarser claims are drawn as spans across the bins they cover. Few
// enough bins carry a strip of thumbnails. Sessions are cards under the
// strip in their own domain, each a door to its pictures and, when a
// planner exists for its kind, a button that freezes, plans and renders
// its story through the story routes.
(function () {
  const surface = document.querySelector("[data-surface]");
  const ORDER = ["week", "day", "hour", "quarter", "minute"];
  const W = 1000;
  const H = 140;
  const BAR_H = 100;
  const pad = (n) => String(n).padStart(2, "0");

  function spell(epoch, bin, domain) {
    const d = new Date(epoch * 1000);
    const day = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
    const suffix = domain === "instant" ? "Z" : domain === "wall" ? " wall" : "";
    if (bin === "day" || bin === "week") return day + suffix;
    return `${day} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}${suffix}`;
  }

  // the events shelf and any <time data-epoch> on the page speak the domain
  for (const node of document.querySelectorAll("time[data-epoch]")) {
    node.textContent = spell(Number(node.dataset.epoch), "hour", node.dataset.domain);
  }
  if (!surface) return;

  const strip = surface.querySelector("[data-strip]");
  const zoomNav = surface.querySelector("[data-zoom]");
  const samplesBox = surface.querySelector("[data-samples]");
  const sessionsList = surface.querySelector("[data-sessions]");
  const note = surface.querySelector("[data-note]");
  let current = null;

  // the surface's scope -- the gallery question in the URL, every
  // parameter but the zoom's own -- rides every ask and every zoom, so
  // what is drawn is what the doors open
  const scope = [...new URLSearchParams(location.search).entries()].filter(([k]) => !["bin", "start", "end"].includes(k));
  const withScope = (qs) => {
    for (const [k, v] of scope) qs.append(k, v);
    return qs;
  };

  function load(bin, start, end) {
    const qs = withScope(new URLSearchParams({ bin }));
    if (start != null) qs.set("start", String(start));
    if (end != null) qs.set("end", String(end));
    return fetch(`/timeline/density?${qs}`, { headers: { accept: "application/json" } }).then((r) => {
      if (!r.ok) return r.json().then((why) => { throw new Error(why.detail || r.statusText); });
      return r.json();
    });
  }

  function svg(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  function draw(view) {
    while (strip.firstChild) strip.removeChild(strip.firstChild);
    const span = Math.max(1, view.end - view.start);
    const most = Math.max(1, ...view.bins.map((b) => b.pictures));
    for (const held of view.spans) {
      const x = ((held.start - view.start) / span) * W;
      const w = Math.max(1, ((held.end - held.start) / span) * W);
      const r = svg("rect", { x, y: BAR_H + 6, width: w, height: 10, class: "span" });
      r.appendChild(svg("title", {})).textContent = `${held.pictures} pictures claim a ${held.precision} from ${spell(held.start, "hour")}`;
      strip.appendChild(r);
    }
    const barW = Math.max(1, (view.bin_seconds / span) * W - 0.5);
    for (const b of view.bins) {
      const x = ((b.at - view.start) / span) * W;
      const h = (b.pictures / most) * BAR_H;
      const deeper = ORDER[ORDER.indexOf(view.bin) + 1];
      const g = svg("g", { class: "bin", "data-bin-at": b.at, role: "listitem", tabindex: "0" });
      const wallH = (b.wall / b.pictures) * h;
      g.appendChild(svg("rect", { x, y: BAR_H - h, width: barW, height: wallH, class: "wall" }));
      g.appendChild(svg("rect", { x, y: BAR_H - h + wallH, width: barW, height: h - wallH, class: "instant" }));
      const o = b.origin;
      g.appendChild(svg("title", {})).textContent =
        `${spell(b.at, view.bin)} · ${b.pictures} pictures (${b.wall} wall clock, ${b.instant} instant)` +
        ` · ${o.captured} captured, ${o.generated} generated, ${o.mixed} mixed, ${o.imported} imported` +
        (deeper ? ` · open: zoom to ${deeper}` : " · open: the gallery");
      const open = () => {
        if (!deeper) { window.location.href = `/g?${b.qs}`; return; }
        go(deeper, b.at, b.at + view.bin_seconds, true);
      };
      g.addEventListener("click", open);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      strip.appendChild(g);
    }
    const axis = svg("text", { x: 0, y: H - 4, class: "axis" });
    axis.textContent = `${spell(view.start, "hour")} – ${spell(view.end, "hour")} · ${view.bin} bins`;
    strip.appendChild(axis);
  }

  function samples(view) {
    samplesBox.textContent = "";
    if (!view.sampled) {
      samplesBox.dataset.state = "too-many";
      samplesBox.textContent = `${view.bins.length} bins is too many for a thumbnail strip — zoom in to see pictures`;
      return;
    }
    samplesBox.dataset.state = "shown";
    for (const b of view.bins) {
      if (!b.samples.length) continue;
      const cell = document.createElement("a");
      cell.className = "surface-sample";
      cell.href = `/g?${b.qs}`;
      cell.title = `${spell(b.at, view.bin)} · ${b.pictures} pictures`;
      cell.dataset.binAt = b.at;
      for (const slug of b.samples) {
        const img = document.createElement("img");
        img.src = `/thumb/${slug}`;
        img.alt = "";
        img.loading = "lazy";
        cell.appendChild(img);
      }
      const n = document.createElement("span");
      n.textContent = `${b.pictures}`;
      cell.appendChild(n);
      samplesBox.appendChild(cell);
    }
  }

  function crumbs(view, stack) {
    zoomNav.textContent = "";
    stack.forEach((held) => {
      const a = document.createElement("a");
      a.href = `/timeline?${withScope(new URLSearchParams({ bin: held.bin, start: held.start, end: held.end }))}`;
      a.textContent = held.bin;
      a.addEventListener("click", (e) => { e.preventDefault(); go(held.bin, held.start, held.end, true); });
      zoomNav.appendChild(a);
      zoomNav.appendChild(document.createTextNode(" › "));
    });
    const here = document.createElement("strong");
    here.textContent = view.bin;
    zoomNav.appendChild(here);
  }

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

  async function tell(s, status) {
    // freeze -> plan -> render, each a story route; a refusal is shown verbatim
    const post = async (where, body) => {
      const r = await fetch(where, { method: "POST", headers: { "content-type": "application/json", accept: "application/json" }, body: JSON.stringify(body) });
      const told = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(told.detail || r.statusText);
      return { status: r.status, told };
    };
    try {
      status.textContent = "freezing…";
      const snap = await post("/stories/snapshots", { event_id: s.id });
      status.textContent = "planning…";
      let plan = await post("/stories/plans", { snapshot_id: snap.told.id, planner: s.planner });
      if (plan.status === 202) {
        // durable work: the job feed says when it settles (no polling),
        // then the same request answers 200 with the plan it made
        status.textContent = `planning as job #${plan.told.job.id}…`;
        const state = await settled(plan.told.job.id);
        if (state !== "done") throw new Error(`the planning job ${state}; see /operations`);
        plan = await post("/stories/plans", { snapshot_id: snap.told.id, planner: s.planner });
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

  function sessions(view) {
    sessionsList.textContent = "";
    if (view.sessions_total > view.sessions.length) {
      const more = document.createElement("li");
      more.className = "session-more";
      more.dataset.sessionsMore = String(view.sessions_total - view.sessions.length);
      more.textContent = `${view.sessions_total} sessions touch this range; the first ${view.sessions.length} are listed — zoom in for the rest`;
      sessionsList.appendChild(more);
    } else if (!view.sessions_sampled && view.sessions.length) {
      const note = document.createElement("li");
      note.className = "session-more";
      note.textContent = `${view.sessions.length} sessions — too many for thumbnails; zoom in to see pictures`;
      sessionsList.appendChild(note);
    }
    for (const s of view.sessions) {
      const li = document.createElement("li");
      li.className = "session";
      li.dataset.sessionKind = s.kind;
      li.dataset.sessionDomain = s.domain;
      li.dataset.sessionId = s.id;
      const strip = document.createElement("a");
      strip.className = "session-strip";
      strip.href = `/g?${s.qs}`;
      for (const slug of s.samples) {
        const img = document.createElement("img");
        img.src = `/thumb/${slug}`;
        img.alt = "";
        img.loading = "lazy";
        strip.appendChild(img);
      }
      li.appendChild(strip);
      const body = document.createElement("div");
      body.className = "session-body";
      const head = document.createElement("div");
      head.className = "session-head";
      const counted = s.in_scope != null && s.in_scope !== s.pictures ? `${s.in_scope} of ${s.pictures} pictures in scope` : `${s.pictures} pictures`;
      head.textContent = `${s.kind.replace("_", " ")} · ${counted} · ${spell(s.start, "hour", s.domain)} – ${spell(s.end, "hour", s.domain)}`;
      body.appendChild(head);
      if (s.place) {
        const at = document.createElement("div");
        at.className = "session-people";
        at.dataset.sessionPlace = String(s.place.id);
        at.appendChild(document.createTextNode("in "));
        const a = document.createElement("a");
        a.href = `/g?${s.place.qs}`;
        a.textContent = s.place.name;
        at.appendChild(a);
        body.appendChild(at);
      }
      if (s.people && s.people.length) {
        const who = document.createElement("div");
        who.className = "session-people";
        who.dataset.sessionPeople = String(s.people.length);
        who.appendChild(document.createTextNode("with "));
        s.people.forEach((p, i) => {
          if (i) who.appendChild(document.createTextNode(", "));
          const a = document.createElement("a");
          a.href = p.href;
          a.textContent = p.name || p.slug;
          a.title = `${p.pictures} of these pictures`;
          who.appendChild(a);
        });
        if (s.people_total > s.people.length) {
          who.appendChild(document.createTextNode(` and ${s.people_total - s.people.length} others`));
        }
        body.appendChild(who);
      }
      const doors = document.createElement("div");
      doors.className = "session-doors";
      const open = document.createElement("a");
      open.href = `/g?${s.qs}`;
      open.textContent = "open in gallery";
      open.dataset.sessionOpen = s.id;
      doors.appendChild(open);
      if (s.story) {
        const a = document.createElement("a");
        a.href = s.story;
        a.textContent = "read the story";
        a.dataset.sessionStory = s.id;
        doors.appendChild(a);
      } else if (s.tellable) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "tell the story";
        btn.dataset.sessionTell = s.id;
        const status = document.createElement("span");
        status.className = "session-status";
        btn.addEventListener("click", () => tell(s, status));
        doors.appendChild(btn);
        doors.appendChild(status);
      } else {
        const none = document.createElement("span");
        none.className = "session-status";
        none.textContent = `no planner tells a ${s.kind.replace("_", " ")} yet`;
        doors.appendChild(none);
      }
      body.appendChild(doors);
      li.appendChild(body);
      sessionsList.appendChild(li);
    }
  }

  function stackFor(bin, start, end) {
    // the crumbs: every coarser level that contains this window
    // a crumb is a zoom level, not a window: the coarser view redraws
    // over its whole extent
    return ORDER.slice(0, ORDER.indexOf(bin)).map((coarser) => ({ bin: coarser, start: null, end: null }));
  }

  function show(bin, start, end) {
    note.textContent = "";
    return load(bin, start, end)
      .then((view) => {
        current = view;
        draw(view);
        crumbs(view, stackFor(bin, start, end));
        samples(view);
        sessions(view);
        const fine = view.bins.reduce((n, b) => n + b.pictures, 0);
        const coarse = view.spans.reduce((n, s) => n + s.pictures, 0);
        const c = view.coverage;
        note.textContent =
          `${fine} pictures placed at ${bin} resolution` +
          (coarse ? `; ${coarse} claim only a coarser window, drawn as spans` : "") +
          (c && !c.complete ? ` · ${c.present - c.interpreted} files not yet interpreted` : "") +
          (c && c.present && !c.events_current ? " · sessions need the events job: the interpretation moved since they were grouped" : "");
      })
      .catch((why) => {
        // a library too wide for day bins opens at the week; the user zooms in
        if (bin === "day" && start == null) return go("week", null, null, false);
        note.textContent = why.message;
      });
  }

  function go(bin, start, end, push) {
    const qs = withScope(new URLSearchParams({ bin }));
    if (start != null) qs.set("start", String(start));
    if (end != null) qs.set("end", String(end));
    const url = `/timeline?${qs}`;
    if (push) history.pushState({ bin, start, end }, "", url);
    else history.replaceState({ bin, start, end }, "", url);
    return show(bin, start, end);
  }

  window.addEventListener("popstate", (e) => {
    const held = e.state || fromUrl();
    show(held.bin, held.start, held.end);
  });

  function fromUrl() {
    const qs = new URLSearchParams(location.search);
    const bin = ORDER.includes(qs.get("bin")) ? qs.get("bin") : "day";
    const start = qs.get("start") != null ? Number(qs.get("start")) : null;
    const end = qs.get("end") != null ? Number(qs.get("end")) : null;
    return { bin, start, end };
  }

  const first = fromUrl();
  go(first.bin, first.start, first.end, false);
})();
