// The Generation Evolution Explorer page: several presentations of ONE
// EvolutionView (embedded JSON from db/evolution.py). This file draws;
// it decides nothing -- phases, families and chronology come from the
// plan, every number from the view. No fetches, no writes.
(() => {
  const root = document.querySelector("[data-evolution]");
  if (!root) return;
  const view = JSON.parse(root.querySelector("[data-evolution-view]").textContent);
  const main = root.querySelector("[data-main]");
  const selectedPane = root.querySelector("[data-selected]");
  const inspector = root.querySelector("[data-inspector]");
  const members = new Map(view.members.map((m) => [m.ref, m]));
  const transitionTo = new Map(view.transitions.map((t) => [t.to, t]));
  const state = { tab: view.plan.sequenced ? "sequence" : "families", selected: null, pair: null };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
  const thumb = (m, cls = "member") =>
    m.media.thumbnail
      ? `<img class="${cls}" data-ref="${m.ref}" src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}" title="${esc(m.ref)} · ${esc(m.media.name)}">`
      : `<span class="${cls}" data-ref="${m.ref}" title="${esc(m.media.name)} (file gone)"></span>`;

  // --- token diff: longest common subsequence over whitespace tokens ----
  function diffTokens(a, b) {
    const x = a ? a.split(/\s+/) : [], y = b ? b.split(/\s+/) : [];
    const n = x.length, m = y.length;
    const L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const out = [];
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (x[i] === y[j]) { out.push(esc(x[i])); i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) { out.push(`<del>${esc(x[i])}</del>`); i++; }
      else { out.push(`<ins>${esc(y[j])}</ins>`); j++; }
    }
    while (i < n) out.push(`<del>${esc(x[i++])}</del>`);
    while (j < m) out.push(`<ins>${esc(y[j++])}</ins>`);
    return out.join(" ");
  }

  // --- presentations -----------------------------------------------------
  function sequence() {
    const strip = view.phases
      .map((p) => `<div class="phase" data-phase="${p.id}"><h3>${esc(p.label)}</h3><div class="members">${p.member_refs.map((r) => thumb(members.get(r))).join("")}</div></div>`)
      .join("");
    const rows = [
      ["prompt vs previous", (t) => t.prompt_cosine, (t) => t.prompt_cosine_unavailable],
      ["image vs previous", (t) => t.visual_cosine, (t) => t.visual_cosine_unavailable],
    ];
    const head = `<tr><th></th>${view.members.map((m) => `<th>${esc(m.ref.replace("member-", ""))}</th>`).join("")}</tr>`;
    const body = rows
      .map(([label, get, why]) => {
        const cells = view.members.map((m, i) => {
          if (i === 0) return "<td>·</td>";
          const t = transitionTo.get(m.ref);
          const v = get(t);
          const cls = (t.phase_boundary ? "boundary " : "") + (v === null ? "unavailable" : "");
          return `<td class="${cls}" title="${esc(why(t) || "")}">${pct(v)}</td>`;
        });
        return `<tr><th>${label}</th>${cells.join("")}</tr>`;
      })
      .join("");
    const facts = ["model", "loras", "sampler", "seed"]
      .map((key) => `<tr><th>${key}</th>${view.members.map((m) => { const v = m.generation[key]; const t = transitionTo.get(m.ref); const cls = t && t.phase_boundary ? "boundary" : ""; return `<td class="${cls}">${esc(Array.isArray(v) ? v.join(", ") : v ?? "—")}</td>`; }).join("")}</tr>`)
      .join("");
    main.innerHTML = `<div class="filmstrip">${strip}</div><div class="tracks"><table>${head}${body}${facts}</table></div>`;
  }

  function drift() {
    const W = 420, H = 320, pad = 36;
    const dots = view.transitions
      .filter((t) => t.prompt_cosine !== null && t.visual_cosine !== null)
      .map((t) => {
        const x = pad + (1 - Math.max(0, t.prompt_cosine)) * (W - 2 * pad);
        const y = H - pad - (1 - Math.max(0, t.visual_cosine)) * (H - 2 * pad);
        return `<circle data-pair="${t.from}|${t.to}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${t.phase_boundary ? "#fc6" : "#6cf"}"><title>${t.from} → ${t.to}: prompt ${pct(t.prompt_cosine)}, image ${pct(t.visual_cosine)}</title></circle>`;
      })
      .join("");
    const missing = view.transitions.length - view.transitions.filter((t) => t.prompt_cosine !== null && t.visual_cosine !== null).length;
    main.innerHTML = `<div class="drift"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#555"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#555"/>
      <text x="${W / 2}" y="${H - 8}" fill="#aaa" font-size="11" text-anchor="middle">prompt change from previous →</text>
      <text x="12" y="${H / 2}" fill="#aaa" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">image change from previous →</text>
      ${dots}</svg>
      <p class="evolution-provenance">each dot is one transition; yellow dots cross a plan phase boundary${missing ? `; ${missing} transition(s) lack a vector and are not drawn` : ""}</p></div>`;
  }

  function families() {
    main.innerHTML = `<div class="families">${view.phases
      .map((p) => `<div class="family" data-phase="${p.id}"><h3>${esc(p.label)} <small>· ${p.member_refs.length}</small></h3>
        <p class="claims">${p.claims.map((c) => esc(c.kind)).join(" · ") || "no claims"}</p>
        <div class="members">${p.member_refs.map((r) => thumb(members.get(r), p.representative_refs.includes(r) ? "member hero" : "member")).join("")}</div></div>`)
      .join("")}</div>`;
  }

  function lineage() {
    if (!view.lineage.length) { main.innerHTML = `<p class="empty">no derivation edges among these images</p>`; return; }
    const children = new Map();
    view.lineage.forEach((e) => { if (!children.has(e.parent)) children.set(e.parent, []); children.get(e.parent).push(e); });
    const isChild = new Set(view.lineage.map((e) => e.child));
    const roots = [...new Set(view.lineage.map((e) => e.parent))].filter((p) => !isChild.has(p));
    const node = (ref, kind) => {
      const m = members.get(ref);
      const label = m ? thumb(m) + ` ${esc(ref)}` : `<span class="kind">outside the session</span> ${esc(ref.slice(0, 8))}`;
      const kids = (children.get(ref) || []).map((e) => node(e.child, e.kind)).join("");
      return `<li>${label}${kind ? ` <span class="kind">${esc(kind)}</span>` : ""}${kids ? `<ul>${kids}</ul>` : ""}</li>`;
    };
    main.innerHTML = `<div class="lineage"><ul>${roots.map((r) => node(r, null)).join("")}</ul></div>`;
  }

  function compare() {
    const [a, b] = state.pair || [];
    if (!a || !b) { main.innerHTML = `<p class="empty">select two images (click one, then shift-click another)</p>`; return; }
    const A = members.get(a), B = members.get(b);
    const t = view.transitions.find((x) => (x.from === a && x.to === b) || (x.from === b && x.to === a));
    const edge = view.lineage.find((e) => (e.parent === a && e.child === b) || (e.parent === b && e.child === a));
    const rows = ["model", "loras", "sampler", "steps", "cfg", "seed", "scheduler", "width", "height"]
      .map((k) => { const va = A.generation[k], vb = B.generation[k]; const same = JSON.stringify(va) === JSON.stringify(vb); return `<tr><th>${k}</th><td>${same ? "same" : `${esc(Array.isArray(va) ? va.join(", ") : va ?? "—")} → ${esc(Array.isArray(vb) ? vb.join(", ") : vb ?? "—")}`}</td></tr>`; })
      .join("");
    main.innerHTML = `<div class="compare"><div>${thumb(A, "big")}<p>${esc(A.ref)} · ${esc(A.media.name)}</p></div><div>${thumb(B, "big")}<p>${esc(B.ref)} · ${esc(B.media.name)}</p></div>
      <div class="metrics" style="grid-column: 1 / -1"><dl>
        <dt>prompt cosine (consecutive only)</dt><dd>${t ? pct(t.prompt_cosine) : "not consecutive"}</dd>
        <dt>visual cosine (consecutive only)</dt><dd>${t ? pct(t.visual_cosine) : "not consecutive"}</dd>
        <dt>lineage</dt><dd>${edge ? `${esc(edge.parent)} → ${esc(edge.child)} (${esc(edge.kind)})` : "no derivation edge"}</dd>
        <dt>prompt diff</dt><dd class="evolution-diff">${diffTokens(A.prompt.effective?.main, B.prompt.effective?.main)}</dd>
      </dl><table class="tracks">${rows}</table></div></div>`;
  }

  // --- panels ------------------------------------------------------------
  function selected() {
    const m = members.get(state.selected);
    if (!m) { selectedPane.innerHTML = `<p class="empty">select an image</p>`; return; }
    const eff = m.prompt.effective, org = m.prompt.original;
    const doors = [];
    if (m.media.page) doors.push(`<a href="${esc(m.media.page)}">open image</a>`);
    if (eff) doors.push(`<a href="/search?q=${encodeURIComponent(eff.main)}">images like this prompt</a>`);
    if (eff && eff.prompt_id !== null && view.semantic.provider) doors.push(`<a href="/prompts/${eff.prompt_id}/neighbours?space=${encodeURIComponent(view.semantic.provider)}">prompts like this</a>`);
    if (view.doors.gallery_day) doors.push(`<a href="${esc(view.doors.gallery_day)}">this day in the gallery</a>`);
    selectedPane.innerHTML = `${m.media.thumbnail ? `<img src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}">` : ""}
      <h2>${esc(m.ref)} · ${esc(m.media.name)}</h2>
      <p class="evolution-provenance">${esc(m.phase_ref || "")}</p>
      <h3>effective prompt</h3><pre>${esc(eff ? eff.text : "— not frozen —")}</pre>
      ${org ? `<h3>as written</h3><pre>${esc(org.text)}</pre><h3>written → ran</h3><p class="evolution-diff">${diffTokens(org.main, eff ? eff.main : "")}</p>` : `<p class="evolution-provenance">no original prompt was recorded by the generator</p>`}
      <p class="doors">${doors.join("")}</p>`;
  }

  function inspect() {
    const m = members.get(state.selected);
    if (!m) { inspector.innerHTML = ""; return; }
    const t = transitionTo.get(m.ref);
    const metric = (label, v, why) => `<dt>${label}</dt><dd title="${esc(why || "")}">${pct(v)}${v === null ? ` <small>${esc(why || "")}</small>` : ""}</dd>`;
    let html = `<dl class="metrics">${metric("written → ran", m.metrics.original_effective_cosine, m.metrics.original_effective_cosine_unavailable)}${metric("prompt ↔ image", m.metrics.text_image_cosine, m.metrics.text_image_cosine_unavailable)}</dl>`;
    if (t) {
      const changes = Object.entries(t.changes).filter(([, v]) => !(Array.isArray(v) && !v.length)).map(([k, v]) => `<tr><th>${k}</th><td>${Array.isArray(v) ? esc(v.join(", ")) : `${esc(v.from ?? "—")} → ${esc(v.to ?? "—")}`}</td></tr>`).join("");
      html += `<h3>from ${esc(t.from)}${t.phase_boundary ? " · phase boundary" : ""}</h3><dl class="metrics">${metric("prompt similarity", t.prompt_cosine, t.prompt_cosine_unavailable)}${metric("visual similarity", t.visual_cosine, t.visual_cosine_unavailable)}</dl><table class="tracks">${changes || "<tr><td>nothing else changed</td></tr>"}</table>`;
    } else if (!view.plan.sequenced) {
      html += `<p class="evolution-provenance">no transitions: the evidence does not establish an order</p>`;
    }
    const edges = view.lineage.filter((e) => e.parent === m.ref || e.child === m.ref);
    if (edges.length) html += `<h3>lineage</h3><ul class="lineage">${edges.map((e) => `<li>${esc(e.parent)} → ${esc(e.child)} <span class="kind">${esc(e.kind)}</span></li>`).join("")}</ul>`;
    inspector.innerHTML = html;
  }

  function draw() {
    ({ sequence, drift, families, lineage, compare })[state.tab]();
    root.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === state.tab));
    root.querySelectorAll("[data-ref]").forEach((el) => {
      el.classList.toggle("on", el.dataset.ref === state.selected);
      el.classList.toggle("pair", !!state.pair && state.pair.includes(el.dataset.ref) && el.dataset.ref !== state.selected);
    });
    root.querySelectorAll("[data-pair]").forEach((el) => el.classList.toggle("on", !!state.pair && el.dataset.pair === state.pair.join("|")));
    selected();
    inspect();
  }

  root.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-tab]");
    if (tab) { state.tab = tab.dataset.tab; draw(); return; }
    const dot = event.target.closest("[data-pair]");
    if (dot) { state.pair = dot.dataset.pair.split("|"); state.selected = state.pair[1]; draw(); return; }
    const el = event.target.closest("[data-ref]");
    if (!el) return;
    if (event.shiftKey && state.selected && state.selected !== el.dataset.ref) { state.pair = [state.selected, el.dataset.ref]; state.tab = "compare"; }
    else state.selected = el.dataset.ref;
    draw();
  });
  draw();
})();
