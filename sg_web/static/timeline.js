// The timeline surface: density per bin at the current zoom, drawn from
// /timeline/density -- the page never counts anything itself. Click a bar
// to zoom into it (day -> hour -> quarter -> minute); at the finest zoom a
// bar opens the gallery on exactly its pictures. Coarser claims are drawn
// as spans across the bins they cover, at the width their signal has.
// Sessions are listed under the strip in their own domain, each a door
// to its story when one has been told.
(function () {
  const surface = document.querySelector("[data-surface]");
  if (!surface) return;
  const strip = surface.querySelector("[data-strip]");
  const zoomNav = surface.querySelector("[data-zoom]");
  const sessionsList = surface.querySelector("[data-sessions]");
  const note = surface.querySelector("[data-note]");
  const ORDER = ["week", "day", "hour", "quarter", "minute"];
  const W = 1000;
  const H = 140;
  const BAR_H = 100;
  const stack = [];

  function spell(epoch, bin) {
    const d = new Date(epoch * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    const day = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
    if (bin === "day" || bin === "week") return day;
    return `${day} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  }

  function load(bin, start, end) {
    const qs = new URLSearchParams({ bin });
    if (start != null) qs.set("start", String(start));
    if (end != null) qs.set("end", String(end));
    return fetch(`/timeline/density?${qs}`, { headers: { accept: "application/json" } }).then((r) => {
      if (!r.ok) return r.json().then((why) => { throw new Error(why.detail || r.statusText); });
      return r.json();
    });
  }

  function draw(view) {
    while (strip.firstChild) strip.removeChild(strip.firstChild);
    const span = Math.max(1, view.end - view.start);
    const most = Math.max(1, ...view.bins.map((b) => b.pictures));
    const svg = (tag, attrs) => {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
      return el;
    };
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
      const g = svg("g", { class: "bin", "data-bin-at": b.at });
      const wallH = (b.wall / b.pictures) * h;
      g.appendChild(svg("rect", { x, y: BAR_H - h, width: barW, height: wallH, class: "wall" }));
      g.appendChild(svg("rect", { x, y: BAR_H - h + wallH, width: barW, height: h - wallH, class: "instant" }));
      g.appendChild(svg("title", {})).textContent = `${spell(b.at, view.bin)} · ${b.pictures} pictures (${b.wall} wall clock, ${b.instant} instant)`;
      g.addEventListener("click", () => {
        const deeper = ORDER[ORDER.indexOf(view.bin) + 1];
        if (!deeper) {
          window.location.href = `/g?${b.qs}`;
          return;
        }
        stack.push(view);
        show(deeper, b.at, b.at + view.bin_seconds);
      });
      strip.appendChild(g);
    }
    const axis = svg("text", { x: 0, y: H - 4, class: "axis" });
    axis.textContent = `${spell(view.start, "hour")} – ${spell(view.end, "hour")} · ${view.bin} bins`;
    strip.appendChild(axis);
  }

  function crumbs(view) {
    zoomNav.textContent = "";
    stack.forEach((held, i) => {
      const a = document.createElement("a");
      a.href = "#";
      a.textContent = `${held.bin}`;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        stack.length = i;
        show(held.bin, held.start, held.end);
      });
      zoomNav.appendChild(a);
      zoomNav.appendChild(document.createTextNode(" › "));
    });
    const here = document.createElement("strong");
    here.textContent = view.bin;
    zoomNav.appendChild(here);
  }

  function sessions(view) {
    sessionsList.textContent = "";
    for (const s of view.sessions) {
      const li = document.createElement("li");
      li.dataset.sessionKind = s.kind;
      li.dataset.sessionDomain = s.domain;
      const when = `${spell(s.start, "hour")} – ${spell(s.end, "hour")}`;
      li.textContent = `${s.kind.replace("_", " ")} · ${s.pictures} pictures · ${when} (${s.domain})`;
      if (s.story) {
        const a = document.createElement("a");
        a.href = s.story;
        a.textContent = " story";
        li.appendChild(a);
      }
      sessionsList.appendChild(li);
    }
  }

  function show(bin, start, end) {
    note.textContent = "";
    load(bin, start, end)
      .then((view) => {
        draw(view);
        crumbs(view);
        sessions(view);
        const fine = view.bins.reduce((n, b) => n + b.pictures, 0);
        const coarse = view.spans.reduce((n, s) => n + s.pictures, 0);
        note.textContent = `${fine} pictures placed at ${bin} resolution` + (coarse ? `; ${coarse} claim only a coarser window, drawn as spans` : "");
      })
      .catch((why) => {
        // a library too wide for day bins opens at the week; the user zooms in
        if (bin === "day" && start == null) {
          show("week");
          return;
        }
        note.textContent = why.message;
      });
  }

  show("day");
})();
