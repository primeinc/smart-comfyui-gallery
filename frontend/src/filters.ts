/**
 * The filter surface: what am I asking for.
 *
 * The gallery header used to carry the whole vocabulary the browser knew
 * about -- kind, rating and favorite, as three `<select>` elements --
 * which is three questions out of a registry of thirty, with no room for
 * a fourth before the header is taller than the photographs. This is the
 * door those went behind, and the room the other twenty-seven came out
 * into.
 *
 * Three decisions worth knowing.
 *
 * THE URL IS THE QUESTION. Nothing here holds filter state. A change
 * assembles a candidate URL and goes there; the server canonicalizes it
 * and renders chips, counts and grid from its own spelling, so reload,
 * Back, a bookmark and a pasted link are the same code path as a click.
 * The alternative -- filters living in JavaScript and the URL catching up
 * -- is how a shared link stops being the thing that was shared.
 *
 * WHAT IS REMEMBERED IS THE FURNITURE, NEVER THE QUESTION. Whether the
 * drawer is open and which sections are disclosed are how a person has
 * arranged their tools: workspace state, kept until they rearrange it.
 * The filters themselves never are. A filter that outlived its URL would
 * mean a bookmark that answers differently for two people.
 *
 * FILTERING IS ONE EDIT SESSION, NOT FOURTEEN. Clicking six values while
 * the drawer is open leaves ONE history entry, so Back means "the
 * question I had before I started filtering" rather than six presses of
 * undo-one-clause. The first change navigates; the rest replace.
 */

import { everyElement, findElement } from "./dom";
import { register } from "./keys";
import { panelState, remember, rememberPanel, workspace } from "./workspace";

/** The query parameters that are the QUESTION, not the position in it. */
const NOT_THE_QUESTION = new Set(["page"]);

/** Marks that this browsing session has begun editing the question. */
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

/** The question as it stands, ready to be changed. */
function question(): URLSearchParams {
  const held = new URLSearchParams(window.location.search);
  for (const name of NOT_THE_QUESTION) held.delete(name);
  return held;
}

/**
 * Go to a changed question.
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

/** The clauses the question holds for one dimension, as URL values. */
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

/**
 * Replace one BOUND of one discovered key, leaving the rest alone.
 *
 * `onlyClause` is too broad here: `param.num:gte:steps=30` and
 * `param.num:lte:steps=50` are one range and must coexist, and a bound
 * on `cfg` must survive editing the one on `steps`. So the clause
 * replaced is matched by key, operator AND parameter name.
 */
function toggledExact(key: string, op: string, param: string, value: string): URLSearchParams {
  const asked = question();
  const mine = `${key}:${op}:${param}=`;
  const rest = asked.getAll("f").filter((one) => !one.startsWith(mine));
  asked.delete("f");
  for (const one of rest) asked.append("f", one);
  if (value !== "") asked.append("f", `${mine}${value}`);
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
 * two kinds asks for a file that is two things and answers nothing --
 * that dimension is always `any`. A picture carries several LoRAs, so
 * both readings are real and the person picks; the stored choice is
 * per-dimension workspace state, because "all of these LoRAs" is an
 * arrangement somebody made, not part of the question's spelling.
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
  // the two readings are therefore different questions. Not offered on a
  // dimension a file has exactly one of, where "all" is a question that
  // answers nothing by construction.
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
        // Every clause this dimension already holds is respelled, so
        // the switch changes the QUESTION rather than only the next
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
  // rows; below a dozen the box is furniture in the way.
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
 * deserves, spelled with the operators the vocabulary says they allow.
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
      // like the tri-state the question actually has: yes, no, or not
      // asked at all.
      button.addEventListener("click", () => go(onlyClause(key, carried, ops[0] ?? "eq", on ? null : value)));
      pair.append(button);
    }
    body.append(pair);
    return;
  }

  // A field this application has no name of its own for: the key is
  // typed because there is no curated list of them, and the whole point
  // of the advanced door is asking about one nothing here anticipated.
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
    // what the question already says for this operator, so the control
    // opens showing the answer rather than blank over a live filter
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

/**
 * Above / below, for one discovered key that holds numbers.
 *
 * The same shape `drawRange` draws for a curated numeric dimension, but
 * the value it sends is a PAIR -- `steps=30` -- because the long tail is
 * rows rather than columns, so naming the field costs a value of its own
 * (db/facets.py `param.num`).
 */
function drawParamRange(body: HTMLElement, param: string, ops: string[]): void {
  for (const op of ops) {
    if (op === "eq") continue;
    const row = document.createElement("label");
    row.className = "filter-range";
    row.dataset.paramOp = op;
    const said = document.createElement("span");
    said.textContent = op === "gte" ? "at least" : "at most";
    const box = document.createElement("input");
    box.type = "number";
    box.step = "any";
    box.setAttribute("aria-label", `${param} ${said.textContent}`);
    const held = question()
      .getAll("f")
      .find((one) => one.startsWith(`param.num:${op}:${param}=`));
    if (held) box.value = held.slice(held.lastIndexOf("=") + 1);
    box.addEventListener("change", () => {
      const wanted = box.value.trim();
      go(toggledExact("param.num", op, param, wanted));
    });
    row.append(said, box);
    body.append(row);
  }
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

/** What `/g/fields/values` sends: one discovered key's values here. */
interface FieldValues {
  param: string;
  options: Option[];
  more: number;
}

/**
 * One discovered key's own control: its values, counted, in this answer.
 *
 * The half that turns "type sniffed format equals…" into picking `png`
 * off a list. A curated dimension has carried its values since the
 * drawer was built, because the vocabulary holds a statement per
 * dimension; the long tail never could, since there is no statement per
 * key -- there is one statement for every key (db/catalog.py `values`).
 *
 * The text box stays underneath, and is not a leftover. A value that is
 * not in this answer has no row here by design, and somebody who knows
 * exactly what they want should not have to find it in a list first.
 */
async function drawParamValues(body: HTMLElement, param: string, label: string): Promise<void> {
  const said = document.createElement("p");
  said.className = "filter-note";
  said.textContent = `${label} — counting…`;
  body.replaceChildren(said);
  body.dataset.state = "counting";
  body.dataset.param = param;

  const asked = question();
  asked.set("param", param);
  let told: FieldValues;
  try {
    const answered = await fetch(`/g/fields/values?${asked.toString()}`, { headers: { accept: "application/json" } });
    if (!answered.ok) throw new Error(`${answered.status}`);
    told = (await answered.json()) as FieldValues;
  } catch {
    said.className = "filter-note warn";
    said.textContent = `could not count ${label}`;
    body.dataset.state = "failed";
    return;
  }

  body.replaceChildren();
  const naming = document.createElement("p");
  naming.className = "filter-note";
  // The raw key, said out loud. This is the spelling somebody would
  // otherwise have had to know, and showing it is how they learn it.
  naming.textContent = `${label} · ${param}`;
  body.append(naming);

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
    // `any`, always: several values of ONE key OR. Asking for a file
    // whose `SniffedFormat` is both png and mp4 answers nothing by
    // construction -- the same reading `multi="any"` states for every
    // dimension a file has exactly one of.
    pick.addEventListener("click", () => {
      go(toggled("param.is", "facet", "any", `${param}=${one.value}`, !one.chosen));
    });
    row.append(pick);
    list.append(row);
  }
  body.append(list);

  if (told.options.length === 0) {
    const none = document.createElement("p");
    none.className = "filter-note";
    none.textContent = "nothing here holds a value for it";
    body.append(none);
  }
  if (told.more > 0) {
    const rest = document.createElement("p");
    rest.className = "filter-note";
    rest.textContent = `${counted(told.more)} more`;
    body.append(rest);
  }

  const form = document.createElement("form");
  form.className = "filter-range";
  const typed = document.createElement("input");
  typed.type = "text";
  typed.value = `${param}=`;
  typed.setAttribute("aria-label", `${label}, written key equals value`);
  const apply = document.createElement("button");
  apply.type = "submit";
  apply.textContent = "apply";
  form.append(typed, apply);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const wanted = typed.value.trim();
    go(onlyClause("param.is", "facet", "any", wanted === "" ? null : wanted));
  });
  body.append(form);
  body.dataset.state = "ready";
}

/** One field the catalog offered, as `/g/fields` sends it. */
interface CatalogField {
  key: string;
  param: string | null;
  label: string;
  group: string;
  value_kind: string;
  ops: string[];
  multi: string;
  note: string;
  curated: boolean;
  covered: number;
  values: number;
  repeats: number;
}

interface Catalog {
  fields: CatalogField[];
  more: number;
}

/** How long a pause in typing means "this is what I meant". */
const SETTLED_MS = 140;

/**
 * The Add-filter box: the application saying what it knows.
 *
 * The drawer already holds every curated dimension in a named section,
 * and the long tail behind an "advanced" heading whose control was a
 * text box placeheld `key=value`. Both assume you can find, or already
 * know, the name of the thing you want. This is the other direction --
 * you type what you half-remember and it answers with what it has,
 * curated and discovered in one list, ranked by what would actually cut
 * the answer you are looking at (db/catalog.py).
 *
 * Choosing a field does NOT apply a filter. It takes you to that
 * field's own control, which already knows its operators and can offer
 * its values -- so this is a way IN to the drawer rather than a second,
 * poorer copy of it. That is the whole reason the catalog says
 * `curated`: a fact we named has a section waiting, and a raw key is
 * asked through the long-tail door with its spelling already filled in,
 * which is the part nobody should have had to remember.
 */
function mountFind(drawer: HTMLElement, reveal: (key: string) => void): void {
  const box = findElement(drawer, "[data-filter-find]", HTMLElement);
  const field = box && findElement(box, "[data-filter-find-input]", HTMLInputElement);
  const list = box && findElement(box, "[data-filter-found]", HTMLElement);
  if (!box || !field || !list) return;

  let at = -1;
  let ticket = 0;
  let timer = 0;

  const rows = () => everyElement(list, "[data-field]", HTMLElement);

  const highlight = (wanted: number) => {
    const all = rows();
    if (all.length === 0) {
      at = -1;
      return;
    }
    // Wrap: a list you can walk off the end of makes somebody reach for
    // the mouse to get back to the top.
    at = (wanted + all.length) % all.length;
    for (const [index, row] of all.entries()) {
      row.setAttribute("aria-selected", String(index === at));
      if (index === at) row.scrollIntoView({ block: "nearest" });
    }
  };

  const shut = () => {
    list.hidden = true;
    list.replaceChildren();
    field.setAttribute("aria-expanded", "false");
    at = -1;
  };

  /** Take the field a row names: go to its control, ready to be used. */
  const take = (one: CatalogField) => {
    shut();
    field.value = "";
    reveal(one.key);
    if (one.param === null) return;
    const section = findElement(drawer, `[data-filter="${one.key}"]`, HTMLDetailsElement);
    const body = section && findElement(section, "[data-filter-body]", HTMLElement);
    if (!body) return;
    // A number-kinded key gets the comparisons a number has. `Steps` and
    // `CFG scale` are the obvious ones, and offering them a list of
    // every value somebody ever generated at -- when what they want is
    // "above 30" -- is a list nobody reads to the end of.
    //
    // The key is the catalog's, not a constant: a numeric field is
    // asked through `param.num`, which compares `fp.value_num`, and a
    // text one through `param.is`, which compares `fp.value_text`.
    if (one.key === "param.num") {
      body.replaceChildren();
      body.dataset.param = one.param;
      const said = document.createElement("p");
      said.className = "filter-note";
      said.dataset.paramSpelling = one.param;
      said.textContent = one.param;
      body.append(said);
      drawParamRange(body, one.param, one.ops);
      body.dataset.state = "ready";
      return;
    }
    // A discovered TEXT key: that section's own control is a text box
    // because there is no curated list of THESE. So the section is
    // rebuilt around the key that was chosen: its values, counted here,
    // with the raw spelling said out loud above them -- which is how
    // somebody learns the spelling instead of being required to know it.
    void drawParamValues(body, one.param, one.label);
  };

  const draw = (told: Catalog) => {
    list.replaceChildren();
    if (told.fields.length === 0) {
      const none = document.createElement("p");
      none.className = "filter-note";
      none.textContent = "nothing here answers to that";
      list.append(none);
    }
    for (const one of told.fields) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "filter-found-row";
      row.dataset.field = one.key;
      if (one.param !== null) row.dataset.param = one.param;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", "false");

      const name = document.createElement("span");
      name.className = "filter-found-label";
      name.textContent = one.label;
      const where = document.createElement("span");
      where.className = "filter-found-group";
      // What it costs to say: for a discovered key, how much of this
      // answer it can speak about at all. A person choosing between two
      // unfamiliar fields is choosing on exactly that.
      where.textContent = one.curated ? one.group : `${one.group} · ${one.covered}`;
      row.append(name, where);
      row.addEventListener("click", () => take(one));
      list.append(row);
    }
    if (told.more > 0) {
      // Never silently truncated: a cut list that does not say so reads
      // as a complete one, and then a field that IS here looks absent.
      const cut = document.createElement("p");
      cut.className = "filter-note";
      cut.textContent = `${told.more} more — keep typing`;
      list.append(cut);
    }
    list.hidden = false;
    field.setAttribute("aria-expanded", "true");
    highlight(0);
  };

  const look = async () => {
    const wanted = field.value.trim();
    const mine = ++ticket;
    const asked = question();
    asked.set("search", wanted);
    try {
      const answered = await fetch(`/g/fields?${asked.toString()}`, { headers: { accept: "application/json" } });
      if (!answered.ok) throw new Error(`${answered.status}`);
      const told = (await answered.json()) as Catalog;
      // A slower earlier answer must never overwrite a newer one: the
      // list would then show what was typed two keystrokes ago.
      if (mine === ticket) draw(told);
    } catch {
      if (mine === ticket) shut();
    }
  };

  field.addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => void look(), SETTLED_MS);
  });
  // Focusing an empty box opens the list at what is worth asking from
  // here -- which is the answer to "what CAN I filter by", and the
  // question somebody with an empty box actually has.
  field.addEventListener("focus", () => void look());

  field.addEventListener("keydown", (event) => {
    if (list.hidden) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      highlight(at + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      highlight(at - 1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      rows()[at]?.click();
    } else if (event.key === "Escape") {
      // The box first, the drawer second: Escape means "undo the
      // smallest thing I am doing", and closing the whole drawer
      // because a list was open is a surprise.
      event.preventDefault();
      event.stopPropagation();
      shut();
    }
  });

  // A click anywhere else is a dismissal. On the document, because the
  // list must also close when somebody goes back to the grid.
  document.addEventListener("click", (event) => {
    if (!list.hidden && event.target instanceof Node && !box.contains(event.target)) shut();
  });
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

  /**
   * Open one dimension's own control, wherever the ask came from.
   *
   * Shared by the chip that made a filter and by the Add-filter list,
   * because "take me to that field" is one behaviour: the drawer opens,
   * the section discloses, its values are counted the first time, and
   * it is scrolled to. Two copies of this drifted once already -- a
   * chip that opened a section which never filled reads as a control
   * that does nothing.
   */
  const reveal = (key: string) => {
    show(true);
    const section = findElement(drawer, `[data-filter="${key}"]`, HTMLDetailsElement);
    if (!section) return;
    section.open = true;
    if (!section.dataset.filled) {
      section.dataset.filled = "1";
      void fill(section);
    }
    section.scrollIntoView({ block: "nearest" });
  };

  // A chip and the filter that made it are the same thing, so clicking
  // one opens the other rather than being a label that does nothing.
  for (const chip of everyElement(root, "[data-chip-edit]", HTMLElement)) {
    chip.addEventListener("click", () => reveal(chip.dataset.chipEdit ?? ""));
  }

  mountFind(drawer, reveal);

  // Clearing is a navigation to the question with nothing in it, and it
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

  // The state restored is the FURNITURE. Which filters are held is the
  // URL's, and was never stored.
  show(workspace().filters === "open", false);
}
