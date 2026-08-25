/**
 * The filter surface: what am I asking for.
 *
 * The gallery header used to carry the whole vocabulary the browser knew
 * about -- kind, rating and favorite, as three `<select>` elements --
 * which is three facets out of a registry of thirty, with no room for
 * a fourth before the header is taller than the photographs. This is the
 * control that opens onto them, and the room the other twenty-seven came out
 * into.
 *
 * Three decisions worth knowing.
 *
 * THE URL IS THE QUERY. Nothing here holds filter state. A change
 * assembles a candidate URL and goes there; the server canonicalizes it
 * and renders chips, counts and grid from its own encoding, so reload,
 * Back, a bookmark and a pasted link are the same code path as a click.
 * The alternative -- filters living in JavaScript and the URL catching up
 * -- is how a shared link stops being the thing that was shared.
 *
 * WHAT IS REMEMBERED IS THE CHROME, NEVER THE QUERY. Whether the
 * drawer is open and which sections are disclosed are how a person has
 * arranged their tools: workspace state, kept until they rearrange it.
 * The filters themselves never are. A filter that outlived its URL would
 * mean a bookmark that responds differently for two people.
 *
 * FILTERING IS ONE EDIT SESSION, NOT FOURTEEN. Clicking six values while
 * the drawer is open leaves ONE history entry, so Back means "the
 * query I had before I started filtering" rather than six presses of
 * undo-one-clause. The first change navigates; the rest replace.
 */

import { everyElement, findElement } from "./dom";
import { register } from "./keys";
import { panelState, remember, rememberPanel, workspace } from "./workspace";

/** The query parameters that make up the query state, not the position in it. */
const NOT_THE_QUESTION = new Set(["page"]);

/** Marks that this browsing session has begun editing the query. */
const EDITING = "sg.filters.editing";

interface Option {
  value: string;
  label: string;
  count: number;
  chosen: boolean;
}

interface Options {
  key: string;
  label: string;
  note: string;
  value_kind: string;
  ops: string[];
  multi: string;
  options: Option[];
  more: number;
}

/** The query as it stands, ready to be changed. */
function question(): URLSearchParams {
  const held = new URLSearchParams(window.location.search);
  for (const name of NOT_THE_QUESTION) held.delete(name);
  return held;
}

/**
 * Go to a changed query.
 *
 * The first change of an editing session takes a history entry, so Back
 * returns to what the person was looking at before they opened the
 * drawer. Every change after it replaces, because nobody wants to press
 * Back six times to undo one afternoon of narrowing.
 */
function go(held: URLSearchParams): void {
  const spelled = held.toString();
  const url = spelled ? `${window.location.pathname}?${spelled}` : window.location.pathname;
  let editing = false;
  try {
    editing = sessionStorage.getItem(EDITING) === "1";
    sessionStorage.setItem(EDITING, "1");
  } catch {
    // storage refused: every change takes its own entry, which is the
    // old behaviour and still correct, only chattier
  }
  if (editing) window.location.replace(url);
  else window.location.assign(url);
}

function endSession(): void {
  try {
    sessionStorage.removeItem(EDITING);
  } catch {
    // nothing to do; the next change simply starts a new entry
  }
}

/** The clauses the query holds for one dimension, as URL values. */
function held(key: string, carried: string): Set<string> {
  const asked = question();
  if (carried === "scope") {
    const value = asked.get(key);
    return new Set(value ? [value] : []);
  }
  const found = new Set<string>();
  for (const spelled of asked.getAll("f")) {
    const parts = spelled.split(":");
    if (parts.length >= 3 && parts[0] === key) found.add(parts.slice(2).join(":"));
  }
  return found;
}

/**
 * Add or remove one clause.
 *
 * A scope holds one value, so choosing replaces and choosing the held
 * one clears -- which is what makes a list of them behave like the radio
 * group it looks like. A facet repeats, so choosing adds and choosing
 * again removes exactly that one.
 */
function toggled(key: string, carried: string, op: string, value: string, on: boolean): URLSearchParams {
  const asked = question();
  if (carried === "scope") {
    if (on) asked.set(key, value);
    else asked.delete(key);
    return asked;
  }
  const spelled = `${key}:${op}:${value}`;
  const rest = asked.getAll("f").filter((one) => one !== spelled);
  asked.delete("f");
  for (const one of rest) asked.append("f", one);
  if (on) asked.append("f", spelled);
  return asked;
}

/** Replace every clause of one dimension with a single new one. */
function onlyClause(key: string, carried: string, op: string, value: string | null): URLSearchParams {
  const asked = question();
  if (carried === "scope") {
    if (value === null) asked.delete(key);
    else asked.set(key, value);
    return asked;
  }
  const rest = asked.getAll("f").filter((one) => !one.startsWith(`${key}:`));
  asked.delete("f");
  for (const one of rest) asked.append("f", one);
  if (value !== null) asked.append("f", `${key}:${op}:${value}`);
  return asked;
}

function counted(n: number): string {
  return n.toLocaleString();
}

/**
 * Which operator a click on a value writes.
 *
 * `any` repeated means OR, `eq` repeated means AND, and which one a
 * dimension SHOULD take is a fact about the dimension rather than a
 * preference (db/vocabulary.py `multi`). A file has one kind, so ANDing
 * two kinds asks for a file that is two things and returns an empty result set --
 * that dimension is always `any`. A picture carries several LoRAs, so
 * both readings are real and the person picks; the stored choice is
 * per-dimension workspace state, because "all of these LoRAs" is an
 * arrangement somebody made, not part of the query's encoding.
 */
function operatorFor(told: Options, carried: string): string {
  if (carried === "scope" || !told.multi) return told.ops[0] ?? "eq";
  if (told.multi === "any") return "any";
  return panelState(`all:${told.key}`) ? "eq" : "any";
}

/** One dimension's list of values, drawn. */
function drawList(body: HTMLElement, told: Options, carried: string, again: () => void): void {
  body.replaceChildren();
  if (!told.options.length) {
    const empty = document.createElement("p");
    empty.className = "filter-note";
    empty.textContent = "nothing here answers this yet";
    body.append(empty);
    return;
  }

  // Any / All, only where a file can carry several of these at once and
  // the two readings are therefore different queries. Not offered on a
  // dimension a file has exactly one of, where "all" is a query that
  // returns an empty result set by construction.
  if (told.multi === "both") {
    const choice = document.createElement("div");
    choice.className = "filter-choice";
    choice.dataset.filterChoice = told.key;
    const all = panelState(`all:${told.key}`) === true;
    for (const [mode, said, why] of [
      ["any", "any of", `media with any one of these ${told.label}s`],
      ["all", "all of", `media carrying every one of these ${told.label}s`],
    ] as const) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-choice-mode";
      button.dataset.mode = mode;
      button.title = why;
      button.textContent = said;
      button.setAttribute("aria-pressed", String(all === (mode === "all")));
      button.addEventListener("click", () => {
        rememberPanel(`all:${told.key}`, mode === "all");
        // Every clause this dimension already holds is re-encoded, so
        // the switch changes the QUERY rather than only the next
        // click -- which would leave a list whose control disagrees
        // with the chips above it.
        const wanted = mode === "all" ? "eq" : "any";
        const asked = question();
        const rest = asked.getAll("f").filter((held) => !held.startsWith(`${told.key}:`));
        const mine = asked
          .getAll("f")
          .filter((held) => held.startsWith(`${told.key}:`))
          .map((held) => `${told.key}:${wanted}:${held.split(":").slice(2).join(":")}`);
        asked.delete("f");
        for (const held of [...rest, ...mine]) asked.append("f", held);
        if (mine.length) go(asked);
        else again();
      });
      choice.append(button);
    }
    body.append(choice);
  }

  // Search within the dimension, once it is big enough to need one. A
  // checkpoint list is 900 rows in a real library and nobody reads 900
  // rows; below a dozen the box is chrome in the way.
  if (told.options.length > 12 || told.more > 0) {
    const find = document.createElement("input");
    find.type = "search";
    find.className = "filter-find";
    find.placeholder = `search ${told.label}`;
    find.setAttribute("aria-label", `search ${told.label}`);
    find.addEventListener("input", () => {
      const wanted = find.value.trim().toLowerCase();
      for (const row of everyElement(body, "[data-option]", HTMLElement)) {
        row.hidden = wanted !== "" && !(row.dataset.label ?? "").toLowerCase().includes(wanted);
      }
    });
    body.append(find);
  }

  const list = document.createElement("ul");
  list.className = "filter-list";
  for (const one of told.options) {
    const row = document.createElement("li");
    row.dataset.option = one.value;
    row.dataset.label = one.label;

    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "filter-option";
    pick.dataset.chosen = one.chosen ? "true" : "false";
    pick.setAttribute("aria-pressed", one.chosen ? "true" : "false");

    const name = document.createElement("span");
    name.className = "filter-option-label";
    name.textContent = one.label;
    const tally = document.createElement("span");
    tally.className = "filter-option-count";
    tally.textContent = counted(one.count);
    pick.append(name, tally);

    // A value that would leave nothing is shown and not offered: seeing
    // that it exists and gives none is a fact about the library, and
    // hiding it makes the same library look like it never had one.
    if (one.count === 0 && !one.chosen) pick.disabled = true;

    pick.addEventListener("click", () => {
      go(toggled(told.key, carried, operatorFor(told, carried), one.value, !one.chosen));
    });
    row.append(pick);
    list.append(row);
  }
  body.append(list);

  if (told.more > 0) {
    const rest = document.createElement("p");
    rest.className = "filter-note";
    rest.textContent = `${counted(told.more)} more — search to narrow`;
    body.append(rest);
  }
}

/**
 * A dimension with no list to offer: a number, or a date.
 *
 * Nine hundred distinct CFG values is not a list anybody picks from, and
 * a date has no candidates at all. Both get the control their kind
 * deserves, written with the operators the vocabulary says they allow.
 */
function drawRange(body: HTMLElement, key: string, carried: string, kind: string, ops: string[]): void {
  body.replaceChildren();
  const now = held(key, carried);

  // A yes/no is two buttons, not a number box. `favorite` is carried as
  // 1/0 and rendering it as `<input type=number>` asked somebody to type
  // a boolean.
  if (kind === "bool") {
    const pair = document.createElement("div");
    pair.className = "filter-choice";
    for (const [value, said] of [
      ["1", "yes"],
      ["0", "no"],
    ] as const) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-choice-mode";
      button.dataset.option = value;
      button.dataset.label = said;
      button.textContent = said;
      const on = now.has(value);
      button.setAttribute("aria-pressed", String(on));
      // Choosing what is already chosen clears it, so the pair behaves
      // like the tri-state the query actually has: yes, no, or not
      // asked at all.
      button.addEventListener("click", () => go(onlyClause(key, carried, ops[0] ?? "eq", on ? null : value)));
      pair.append(button);
    }
    body.append(pair);
    return;
  }

  // A field this application has no name of its own for: the key is
  // typed because there is no curated list of them, and the whole point
  // of the advanced section is asking about one nothing here anticipated.
  if (kind === "pair") {
    const form = document.createElement("form");
    form.className = "filter-range";
    const field = document.createElement("input");
    field.type = "text";
    field.placeholder = "key=value";
    field.setAttribute("aria-label", `${key}, written key equals value`);
    field.value = [...now][0] ?? "";
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.textContent = "apply";
    form.append(field, apply);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const wanted = field.value.trim();
      go(onlyClause(key, carried, ops[0] ?? "eq", wanted === "" ? null : wanted));
    });
    body.append(form);
    return;
  }
  const form = document.createElement("form");
  form.className = "filter-range";

  const fields: Array<{ op: string; input: HTMLInputElement }> = [];
  for (const op of ops) {
    if (op !== "gte" && op !== "lte" && op !== "eq") continue;
    const wrap = document.createElement("label");
    wrap.className = "filter-range-field";
    const said = document.createElement("span");
    said.textContent = op === "gte" ? "from" : op === "lte" ? "to" : "exactly";
    const input = document.createElement("input");
    input.type = kind === "date" ? "date" : "number";
    if (kind === "num") input.step = "any";
    input.name = op;
    input.setAttribute("aria-label", `${key} ${said.textContent}`);
    // what the query already holds for this operator, so the control
    // opens showing that value rather than blank over a live filter
    for (const spelled of question().getAll("f")) {
      const parts = spelled.split(":");
      if (parts[0] === key && parts[1] === op) input.value = parts.slice(2).join(":");
    }
    if (carried === "scope" && op === ops[0]) {
      const value = [...now][0];
      if (value) input.value = value;
    }
    wrap.append(said, input);
    form.append(wrap);
    fields.push({ op, input });
  }

  const apply = document.createElement("button");
  apply.type = "submit";
  apply.textContent = "apply";
  form.append(apply);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    let asked = question();
    for (const { op, input } of fields) {
      // Each operator is its own clause, so clearing one box drops that
      // bound and leaves the other standing.
      const value = input.value.trim();
      if (carried === "scope") {
        asked = onlyClause(key, carried, op, value === "" ? null : value);
        continue;
      }
      const rest = asked.getAll("f").filter((one) => !one.startsWith(`${key}:${op}:`));
      asked.delete("f");
      for (const one of rest) asked.append("f", one);
      if (value !== "") asked.append("f", `${key}:${op}:${value}`);
    }
    go(asked);
  });

  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "filter-range-clear";
  clear.textContent = "clear";
  clear.addEventListener("click", () => go(onlyClause(key, carried, ops[0] ?? "eq", null)));
  form.append(clear);
  body.append(form);
}

/** Fetch and draw one dimension, once. */
async function fill(section: HTMLDetailsElement): Promise<void> {
  const body = findElement(section, "[data-filter-body]", HTMLElement);
  const key = section.dataset.filter;
  if (!body || !key) return;
  const carried = section.dataset.carried ?? "facet";
  const ops = (section.dataset.ops ?? "eq").split(",").filter(Boolean);

  if (section.dataset.listable !== "1") {
    drawRange(body, key, carried, section.dataset.valueKind ?? "int", ops);
    body.dataset.state = "ready";
    return;
  }

  // The body says what it is doing. Without it, "no values yet" and "not
  // asked yet" are the same empty element, and anything waiting on this
  // section -- a screen reader, a test -- reads the second as the first.
  body.dataset.state = "counting";
  body.replaceChildren();
  const waiting = document.createElement("p");
  waiting.className = "filter-note";
  waiting.textContent = "counting…";
  body.append(waiting);

  const asked = question();
  asked.set("key", key);
  try {
    const answered = await fetch(`/g/options?${asked.toString()}`, { headers: { accept: "application/json" } });
    if (!answered.ok) throw new Error(`${answered.status}`);
    drawList(body, (await answered.json()) as Options, carried, () => void fill(section));
    body.dataset.state = "ready";
  } catch {
    // Say which way it went. A section that silently stays on
    // "counting…" is indistinguishable from a slow one.
    body.replaceChildren();
    const failed = document.createElement("p");
    failed.className = "filter-note warn";
    failed.textContent = "could not count these";
    body.append(failed);
    body.dataset.state = "failed";
  }
}

export function mountFilters(root: HTMLElement): void {
  // `[data-filters-panel]`, never `[data-filters]`: the root carries
  // that as its open/closed state and would match first.
  const drawer = findElement(root, "[data-filters-panel]", HTMLElement);
  const open = findElement(root, "[data-filters-open]", HTMLElement);
  if (!drawer || !open) return;

  const show = (on: boolean, arranged = true) => {
    drawer.hidden = !on;
    root.dataset.filters = on ? "open" : "closed";
    open.setAttribute("aria-expanded", on ? "true" : "false");
    if (arranged) remember({ filters: on ? "open" : "closed" });
    if (!on) endSession();
  };

  // `hidden` is `boolean | "until-found"`, so "is it shut" is a
  // comparison against false rather than the truthiness of the property.
  open.addEventListener("click", () => show(drawer.hidden !== false));
  const close = findElement(drawer, "[data-filters-close]", HTMLElement);
  if (close) close.addEventListener("click", () => show(false));

  // Every section fetches its own values the first time it is opened,
  // and remembers whether it was open. Counting thirty dimensions to
  // draw a drawer nobody has looked in is thirty queries for nothing.
  for (const section of everyElement(drawer, "[data-filter]", HTMLDetailsElement)) {
    const key = section.dataset.filter ?? "";
    const said = panelState(`filter:${key}`);
    if (said) section.open = true;
    section.addEventListener("toggle", () => {
      rememberPanel(`filter:${key}`, section.open);
      if (section.open && !section.dataset.filled) {
        section.dataset.filled = "1";
        void fill(section);
      }
    });
    if (section.open) {
      section.dataset.filled = "1";
      void fill(section);
    }
  }

  // A chip and the filter that made it are the same thing, so clicking
  // one opens the other rather than being a label that does nothing.
  for (const chip of everyElement(root, "[data-chip-edit]", HTMLElement)) {
    chip.addEventListener("click", () => {
      const key = chip.dataset.chipEdit ?? "";
      const section = findElement(drawer, `[data-filter="${key}"]`, HTMLDetailsElement);
      show(true);
      if (!section) return;
      section.open = true;
      section.scrollIntoView({ block: "nearest" });
    });
  }

  // Clearing is a navigation to the query with nothing in it, and it
  // ends the editing session: the next filter someone applies starts a
  // fresh history entry rather than replacing this one.
  for (const clear of everyElement(root, "[data-filters-clear], [data-chips-clear]", HTMLElement)) {
    clear.addEventListener("click", endSession);
  }

  // `/` puts the caret in the search box, which is what `/` does
  // everywhere. Deliberately not a letter for the drawer: `f` has been
  // favourite since authored.ts claimed it, and the registry refuses a
  // second claim rather than letting one keystroke rate a picture AND
  // open a panel. The button is the way in, and it is always visible --
  // which was the requirement; the shortcut is a convenience on top.
  const search = findElement(root, '[data-ask] input[type="search"]', HTMLInputElement);
  if (search) {
    register([
      {
        key: "/",
        by: "gallery: search",
        run: () => {
          search.focus();
          search.select();
        },
      },
    ]);
  }

  // The state restored is the CHROME. Which filters are held is the
  // URL's, and was never stored.
  show(workspace().filters === "open", false);
}
