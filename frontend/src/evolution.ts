// The Generation Evolution Explorer page: several presentations of ONE
// EvolutionView. This file draws; it decides nothing -- phases, families and
// chronology come from the plan, every number from the view. No writes, no
// model work.
//
// The document arrives from the same route that rendered the shell
// (/stories/plans/{id}/evolution, by Accept), typed from the application's
// own contract. It is not serialized into the HTML for this file to parse
// back out: there is one document and one description of it.
import { api } from "./api";
import { closestFrom, everyElement, requireData, requireElement } from "./dom";
import type { components } from "./generated/api";

type EvolutionView = components["schemas"]["EvolutionView"];
type Member = components["schemas"]["EvolutionMember"];
type Transition = components["schemas"]["EvolutionTransition"];
type Generation = components["schemas"]["EvolutionGeneration"];

/** The recipe facts the tables show, in the order they show them. */
const SEQUENCE_FACTS = ["model", "loras", "sampler", "seed"] satisfies (keyof Generation)[];
const COMPARE_FACTS = [
  "model",
  "loras",
  "sampler",
  "steps",
  "cfg",
  "seed",
  "scheduler",
  "width",
  "height",
] satisfies (keyof Generation)[];

const NOUNS: Readonly<Record<string, string>> = { capture_session: "photographs", file_session: "files" };
const ENTITIES: Readonly<Record<string, string>> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

const esc = (s: unknown): string => String(s ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c] ?? c);
const pct = (v: number | null): string => (v === null ? "—" : `${Math.round(v * 100)}%`);

/** A generation fact as one cell: a list joined, a missing value a dash. */
function spell(value: Generation[keyof Generation]): string {
  if (Array.isArray(value)) return esc(value.join(", "));
  return esc(value ?? "—");
}

(() => {
  const root = requireElement(document, "[data-evolution]", HTMLElement);
  const main = requireElement(root, "[data-main]", HTMLElement);
  const selectedPane = requireElement(root, "[data-selected]", HTMLElement);
  const inspector = requireElement(root, "[data-inspector]", HTMLElement);

  const planId = Number(requireData(root, "plan"));
  // the page was rendered in one semantic space; the document is asked for
  // in the same one, so the numbers on screen are all from one measurement
  const space = new URLSearchParams(window.location.search).get("space");

  api
    .GET("/stories/plans/{plan_id}/evolution", {
      params: { path: { plan_id: planId }, query: space === null ? {} : { space } },
    })
    .then(({ data }) => {
      if (data === undefined) {
        main.textContent = "the plan's measurements could not be read";
        return;
      }
      explore(data);
    }, console.error);

  function explore(view: EvolutionView): void {
    const members = new Map<string, Member>(view.members.map((m) => [m.ref, m]));
    const transitionTo = new Map<string, Transition>(view.transitions.map((t) => [t.after, t]));
    const noun = NOUNS[view.snapshot.subject] ?? "images";
    let selected: string | null = null;
    let pair: readonly [string, string] | null = null;

    const member = (ref: string | null): Member | undefined => (ref === null ? undefined : members.get(ref));

    const thumb = (m: Member | undefined, cls = "member"): string => {
      if (m === undefined) return "";
      if (m.media.thumbnail === null) {
        return `<span class="${cls}" data-ref="${esc(m.ref)}" title="${esc(m.media.name)} (file gone)"></span>`;
      }
      return `<img class="${cls}" data-ref="${esc(m.ref)}" src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}" title="${esc(m.ref)} · ${esc(m.media.name)}">`;
    };

    // the claimed time, and the filesystem's estimate beside it -- marked
    const clock = (wall: number): string => new Date(wall * 1000).toISOString().slice(11, 19);
    const when = (m: Member): string => {
      const o = m.occurrence;
      if (o === null || o.local_at === null) return "";
      let told = ` · ${esc(o.precision)} ${clock(o.local_at)} (${esc(o.basis)})`;
      if (o.estimated_at !== null) {
        const finish = o.finished_at === null ? "" : `finish ${clock(o.finished_at)} minus generation time`;
        told += ` <span class="chip chip-inferred" title="${esc(finish)}">≈ ${clock(o.estimated_at)} inferred</span>`;
      }
      if (o.conflicts.length) {
        told += ` <span class="chip chip-conflict" title="${esc(o.conflicts.join("; "))}">contested</span>`;
      }
      return told;
    };

    const metric = (label: string, v: number | null, why: string | null): string =>
      `<dt>${label}</dt><dd title="${esc(why)}">${pct(v)}${v === null ? ` <small>${esc(why)}</small>` : ""}</dd>`;

    // --- presentations ---------------------------------------------------
    function sequence(): void {
      const strip = view.phases
        .map(
          (p) =>
            `<div class="phase" data-phase="${esc(p.id)}"><h3>${esc(p.label)}</h3><div class="members">${p.member_refs
              .map((r) => thumb(members.get(r)))
              .join("")}</div></div>`,
        )
        .join("");
      // a file or capture session carries no prompt and no generator
      // parameters: only the rows its evidence can fill are drawn
      const generated = view.snapshot.subject === "generation_session";
      const rows: [string, (t: Transition) => number | null, (t: Transition) => string | null][] = [
        ...(generated
          ? ([["prompt vs previous", (t) => t.prompt_cosine, (t) => t.prompt_cosine_unavailable ?? null]] as [
              string,
              (t: Transition) => number | null,
              (t: Transition) => string | null,
            ][])
          : []),
        ["image vs previous", (t) => t.visual_cosine, (t) => t.visual_cosine_unavailable ?? null],
      ];
      const head = `<tr><th></th>${view.members.map((m) => `<th>${esc(m.ref.replace("member-", ""))}</th>`).join("")}</tr>`;
      const body = rows
        .map(([label, get, why]) => {
          const cells = view.members.map((m, i) => {
            const t = transitionTo.get(m.ref);
            if (i === 0 || t === undefined) return "<td>·</td>";
            const v = get(t);
            const cls = (t.phase_boundary ? "boundary " : "") + (v === null ? "unavailable" : "");
            return `<td class="${cls}" title="${esc(why(t))}">${pct(v)}</td>`;
          });
          return `<tr><th>${label}</th>${cells.join("")}</tr>`;
        })
        .join("");
      const facts = (generated ? SEQUENCE_FACTS : [])
        .map(
          (key) =>
            `<tr><th>${key}</th>${view.members
              .map((m) => {
                const boundary = transitionTo.get(m.ref)?.phase_boundary ?? false;
                return `<td class="${boundary ? "boundary" : ""}">${spell(m.generation[key])}</td>`;
              })
              .join("")}</tr>`,
        )
        .join("");
      main.innerHTML = `<div class="filmstrip">${strip}</div><div class="tracks"><table>${head}${body}${facts}</table></div>`;
    }

    function drift(): void {
      const W = 420;
      const H = 320;
      const pad = 36;
      const drawn = view.transitions.filter(
        (t): t is Transition & { prompt_cosine: number; visual_cosine: number } =>
          t.prompt_cosine !== null && t.visual_cosine !== null,
      );
      const dots = drawn
        .map((t) => {
          const x = pad + (1 - Math.max(0, t.prompt_cosine)) * (W - 2 * pad);
          const y = H - pad - (1 - Math.max(0, t.visual_cosine)) * (H - 2 * pad);
          return `<circle data-pair="${esc(t.before)}|${esc(t.after)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${t.phase_boundary ? "#fc6" : "#6cf"}"><title>${esc(t.before)} → ${esc(t.after)}: prompt ${pct(t.prompt_cosine)}, image ${pct(t.visual_cosine)}</title></circle>`;
        })
        .join("");
      const missing = view.transitions.length - drawn.length;
      main.innerHTML = `<div class="drift"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#555"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#555"/>
      <text x="${W / 2}" y="${H - 8}" fill="#aaa" font-size="11" text-anchor="middle">prompt change from previous →</text>
      <text x="12" y="${H / 2}" fill="#aaa" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">image change from previous →</text>
      ${dots}</svg>
      <p class="evolution-provenance">each dot is one transition; yellow dots cross a plan phase boundary${missing ? `; ${missing} transition(s) lack a vector and are not drawn` : ""}</p></div>`;
    }

    function families(): void {
      main.innerHTML = `<div class="families">${view.phases
        .map(
          (
            p,
          ) => `<div class="family" data-phase="${esc(p.id)}"><h3>${esc(p.label)} <small>· ${p.member_refs.length}</small></h3>
        <p class="claims">${p.claims.map((c) => esc(c.kind)).join(" · ") || "no claims"}</p>
        <div class="members">${p.member_refs
          .map((r) => thumb(members.get(r), p.representative_refs.includes(r) ? "member hero" : "member"))
          .join("")}</div></div>`,
        )
        .join("")}</div>`;
    }

    function lineage(): void {
      if (!view.lineage.length) {
        main.innerHTML = `<p class="empty">no derivation edges among these ${noun}</p>`;
        return;
      }
      const children = new Map<string, string[]>();
      for (const e of view.lineage) {
        const held = children.get(e.parent);
        if (held === undefined) children.set(e.parent, [e.child]);
        else held.push(e.child);
      }
      const kindOf = new Map(view.lineage.map((e) => [`${e.parent}|${e.child}`, e.kind]));
      const isChild = new Set(view.lineage.map((e) => e.child));
      const roots = [...new Set(view.lineage.map((e) => e.parent))].filter((p) => !isChild.has(p));
      const node = (ref: string, kind: string | null): string => {
        const m = members.get(ref);
        const label =
          m === undefined
            ? `<span class="kind">outside the session</span> ${esc(ref.slice(0, 8))}`
            : `${thumb(m)} ${esc(ref)}`;
        const kids = (children.get(ref) ?? [])
          .map((child) => node(child, kindOf.get(`${ref}|${child}`) ?? null))
          .join("");
        return `<li>${label}${kind === null ? "" : ` <span class="kind">${esc(kind)}</span>`}${kids ? `<ul>${kids}</ul>` : ""}</li>`;
      };
      main.innerHTML = `<div class="lineage"><ul>${roots.map((r) => node(r, null)).join("")}</ul></div>`;
    }

    function compare(): void {
      const A = pair === null ? undefined : members.get(pair[0]);
      const B = pair === null ? undefined : members.get(pair[1]);
      if (A === undefined || B === undefined) {
        main.innerHTML = `<p class="empty">select two images (click one, then shift-click another)</p>`;
        return;
      }
      const t = view.transitions.find(
        (x) => (x.before === A.ref && x.after === B.ref) || (x.before === B.ref && x.after === A.ref),
      );
      const edge = view.lineage.find(
        (e) => (e.parent === A.ref && e.child === B.ref) || (e.parent === B.ref && e.child === A.ref),
      );
      const rows = COMPARE_FACTS.map((k) => {
        const va = A.generation[k];
        const vb = B.generation[k];
        const same = JSON.stringify(va) === JSON.stringify(vb);
        return `<tr><th>${k}</th><td>${same ? "same" : `${spell(va)} → ${spell(vb)}`}</td></tr>`;
      }).join("");
      main.innerHTML = `<div class="compare"><div>${thumb(A, "big")}<p>${esc(A.ref)} · ${esc(A.media.name)}</p></div><div>${thumb(B, "big")}<p>${esc(B.ref)} · ${esc(B.media.name)}</p></div>
      <div class="metrics" style="grid-column: 1 / -1"><dl>
        <dt>prompt cosine (consecutive only)</dt><dd>${t === undefined ? "not consecutive" : pct(t.prompt_cosine)}</dd>
        <dt>visual cosine (consecutive only)</dt><dd>${t === undefined ? "not consecutive" : pct(t.visual_cosine)}</dd>
        <dt>lineage</dt><dd>${edge === undefined ? "no derivation edge" : `${esc(edge.parent)} → ${esc(edge.child)} (${esc(edge.kind)})`}</dd>
        <dt>prompt diff</dt><dd class="evolution-diff">${diffTokens(A.prompt.effective?.main ?? "", B.prompt.effective?.main ?? "")}</dd>
      </dl><table class="tracks">${rows}</table></div></div>`;
    }

    // --- panels ----------------------------------------------------------
    function panel(): void {
      const m = member(selected);
      if (m === undefined) {
        selectedPane.innerHTML = `<p class="empty">select an image</p>`;
        return;
      }
      const eff = m.prompt.effective;
      const org = m.prompt.original;
      const links: string[] = [];
      if (m.media.page !== null) links.push(`<a href="${esc(m.media.page)}">open image</a>`);
      if (eff !== null) {
        links.push(`<a href="${esc(view.links.search)}${encodeURIComponent(eff.main)}">images like this prompt</a>`);
      }
      if (eff !== null && eff.prompt_id !== null && view.semantic.provider !== null) {
        links.push(
          `<a href="/prompts/${eff.prompt_id}/neighbours?space=${encodeURIComponent(view.semantic.provider)}">prompts like this</a>`,
        );
      }
      if (view.links.gallery_day !== null) {
        links.push(`<a href="${esc(view.links.gallery_day)}">this day in the gallery</a>`);
      }
      selectedPane.innerHTML = `${m.media.thumbnail === null ? "" : `<img src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}">`}
      <h2>${esc(m.ref)} · ${esc(m.media.name)}</h2>
      <p class="evolution-provenance">${esc(m.phase_ref)}${when(m)}</p>
      <h3>effective prompt</h3><pre>${esc(eff === null ? "— not frozen —" : eff.text)}</pre>
      ${
        org === null
          ? `<p class="evolution-provenance">no original prompt was recorded by the generator</p>`
          : `<h3>as written</h3><pre>${esc(org.text)}</pre><h3>written → ran</h3><p class="evolution-diff">${diffTokens(org.main, eff?.main ?? "")}</p>`
      }
      <p class="links">${links.join("")}</p>`;
    }

    function inspect(): void {
      const m = member(selected);
      if (m === undefined) {
        inspector.innerHTML = "";
        return;
      }
      const t = transitionTo.get(m.ref);
      let html = `<dl class="metrics">${metric(
        "written → ran",
        m.metrics.original_effective_cosine,
        m.metrics.original_effective_cosine_unavailable ?? null,
      )}${metric("prompt ↔ image", m.metrics.text_image_cosine, m.metrics.text_image_cosine_unavailable ?? null)}</dl>`;
      if (t !== undefined) {
        const rows = [
          ...t.changes.parameters.map(
            (one) => `<tr><th>${esc(one.name)}</th><td>${spell(one.before)} → ${spell(one.after)}</td></tr>`,
          ),
          ...(t.changes.loras_added.length
            ? [`<tr><th>loras_added</th><td>${esc(t.changes.loras_added.join(", "))}</td></tr>`]
            : []),
          ...(t.changes.loras_removed.length
            ? [`<tr><th>loras_removed</th><td>${esc(t.changes.loras_removed.join(", "))}</td></tr>`]
            : []),
        ].join("");
        html += `<h3>from ${esc(t.before)}${t.phase_boundary ? " · phase boundary" : ""}</h3><dl class="metrics">${metric(
          "prompt similarity",
          t.prompt_cosine,
          t.prompt_cosine_unavailable ?? null,
        )}${metric("visual similarity", t.visual_cosine, t.visual_cosine_unavailable ?? null)}</dl><table class="tracks">${rows || "<tr><td>nothing else changed</td></tr>"}</table>`;
      } else if (!view.plan.sequenced) {
        html += `<p class="evolution-provenance">no transitions: the evidence does not establish an order</p>`;
      }
      const edges = view.lineage.filter((e) => e.parent === m.ref || e.child === m.ref);
      if (edges.length) {
        html += `<h3>lineage</h3><ul class="lineage">${edges
          .map((e) => `<li>${esc(e.parent)} → ${esc(e.child)} <span class="kind">${esc(e.kind)}</span></li>`)
          .join("")}</ul>`;
      }
      inspector.innerHTML = html;
    }

    const panels = { sequence, drift, families, lineage, compare };
    type Tab = keyof typeof panels;
    const isTab = (name: string): name is Tab => name in panels;
    let tab: Tab = view.plan.sequenced ? "sequence" : "families";

    function draw(): void {
      panels[tab]();
      for (const b of everyElement(root, "[data-tab]", HTMLElement)) {
        b.classList.toggle("on", b.dataset.tab === tab);
      }
      for (const el of everyElement(root, "[data-ref]", HTMLElement)) {
        const ref = el.dataset.ref;
        el.classList.toggle("on", ref === selected);
        el.classList.toggle("pair", pair !== null && ref !== undefined && pair.includes(ref) && ref !== selected);
      }
      for (const el of everyElement(root, "[data-pair]", HTMLElement)) {
        el.classList.toggle("on", pair !== null && el.dataset.pair === pair.join("|"));
      }
      panel();
      inspect();
    }

    root.addEventListener("click", (event) => {
      const chosen = closestFrom(event.target, "[data-tab]", HTMLElement);
      if (chosen !== null) {
        const name = requireData(chosen, "tab");
        if (isTab(name)) tab = name;
        draw();
        return;
      }
      const dot = closestFrom(event.target, "[data-pair]", Element);
      if (dot !== null) {
        const [before, after] = requireAttribute(dot, "data-pair").split("|");
        if (before !== undefined && after !== undefined) {
          pair = [before, after];
          selected = after;
        }
        draw();
        return;
      }
      const el = closestFrom(event.target, "[data-ref]", Element);
      if (el === null) return;
      const ref = requireAttribute(el, "data-ref");
      if (event.shiftKey && selected !== null && selected !== ref) {
        pair = [selected, ref];
        tab = "compare";
      } else {
        selected = ref;
      }
      draw();
    });

    draw();
  }
})();

/**
 * The attribute the markup must carry.
 *
 * `dataset` is only on HTMLElement, and the drift chart's dots are SVG
 * circles -- Element, not HTMLElement -- so those are read by attribute.
 */
function requireAttribute(node: Element, name: string): string {
  const held = node.getAttribute(name);
  if (held === null) throw new Error(`expected a ${name} on ${node.tagName.toLowerCase()}`);
  return held;
}

/** Longest common subsequence over whitespace tokens, as marked-up HTML. */
function diffTokens(a: string, b: string): string {
  const x = a ? a.split(/\s+/) : [];
  const y = b ? b.split(/\s+/) : [];
  const n = x.length;
  const m = y.length;
  const L: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  const at = (i: number, j: number): number => L[i]?.[j] ?? 0;
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      const row = L[i];
      if (row === undefined) continue;
      row[j] = x[i] === y[j] ? at(i + 1, j + 1) + 1 : Math.max(at(i + 1, j), at(i, j + 1));
    }
  }
  const out: string[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) {
      out.push(esc(x[i]));
      i++;
      j++;
    } else if (at(i + 1, j) >= at(i, j + 1)) {
      out.push(`<del>${esc(x[i])}</del>`);
      i++;
    } else {
      out.push(`<ins>${esc(y[j])}</ins>`);
      j++;
    }
  }
  while (i < n) out.push(`<del>${esc(x[i++])}</del>`);
  while (j < m) out.push(`<ins>${esc(y[j++])}</ins>`);
  return out.join(" ");
}
