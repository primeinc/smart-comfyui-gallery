/**
 * What the keyboard does, said on screen.
 *
 * Ten keys were reachable and none of them was written anywhere a person
 * could find: `c` keeps a picture for the compare tray, and the tray is
 * the whole feature — styled, tested, and openable only by somebody who
 * already knew the letter.
 *
 * The list is BUILT FROM THE REGISTRY (keys.ts `registered`), never
 * written out here. immich keeps its shortcut modal as a hand-maintained
 * array beside bindings that live elsewhere
 * (web/src/lib/modals/ShortcutsModal.svelte), which is a second copy free
 * to disagree with the first. Ours cannot: if a key is not registered it
 * does not appear, and if it moves the list moves with it.
 *
 * Read when the panel OPENS, not when this module mounts, so it does not
 * matter whether the viewer or the tray claimed its keys first. What it
 * shows is what is live on this surface at the moment somebody asked.
 */

import { panel } from "./ask";
import { register, registered } from "./keys";

/** How a key is spelled for a reader, not for `KeyboardEvent.key`. */
const SPELLED: Record<string, string> = {
  ArrowLeft: "←",
  ArrowRight: "→",
  ArrowUp: "↑",
  ArrowDown: "↓",
  " ": "Space",
  Escape: "Esc",
};

const spell = (key: string) => SPELLED[key] ?? (key.length === 1 ? key.toUpperCase() : key);

/**
 * The list, grouped by the surface that claimed each key.
 *
 * `by` is already written as "viewer: next" -- the surface, then what it
 * does -- so the grouping is read off the name rather than declared a
 * second time. A command with no prefix is a general one.
 */
function grouped(): Map<string, { key: string; does: string }[]> {
  const groups = new Map<string, { key: string; does: string }[]>();
  for (const { key, by } of registered()) {
    const cut = by.indexOf(":");
    const where = cut === -1 ? "everywhere" : by.slice(0, cut).trim();
    const does = cut === -1 ? by : by.slice(cut + 1).trim();
    const held = groups.get(where) ?? [];
    held.push({ key: spell(key), does });
    groups.set(where, held);
  }
  return groups;
}

function draw(body: HTMLElement): void {
  const groups = grouped();
  if (groups.size === 0) {
    const none = document.createElement("p");
    none.className = "muted";
    none.textContent = "nothing on this surface answers to a key.";
    body.append(none);
    return;
  }
  for (const [where, commands] of groups) {
    const section = document.createElement("section");
    section.className = "keys-group";
    const head = document.createElement("h3");
    head.textContent = where;
    section.append(head);
    const list = document.createElement("dl");
    list.className = "keys-list";
    for (const { key, does } of commands) {
      const term = document.createElement("dt");
      const cap = document.createElement("kbd");
      cap.textContent = key;
      term.append(cap);
      const said = document.createElement("dd");
      said.textContent = does;
      list.append(term, said);
    }
    section.append(list);
    body.append(section);
  }
}

/** Open it. Exported so a visible control can call it, not only a key. */
export function showShortcuts(): void {
  void panel("what the keyboard does", draw);
}

export function mountShortcuts(root: ParentNode): void {
  // `?` is the convention every application with shortcuts uses, immich
  // included. It reaches this module only when nothing is focused
  // (keys.ts refuses while an input or an open dialog has the keyboard),
  // so it cannot eat a question mark somebody is typing.
  register([{ key: "?", by: "what the keyboard does", run: showShortcuts }]);

  // And a control, because a key nobody can see is the problem this
  // module exists to fix. Every surface carries the shell, so this is on
  // every surface.
  for (const button of root.querySelectorAll("[data-shortcuts-open]")) {
    button.addEventListener("click", showShortcuts);
  }
}
