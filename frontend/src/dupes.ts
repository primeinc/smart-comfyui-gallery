/**
 * The duplicates page: pick a copy to look at, and say when one is not a
 * duplicate at all.
 *
 * Grouping is a guess. It compares the shape of a picture, so two photos
 * of one scene taken a second apart look the same to it and land in one
 * group. Saying "not a duplicate" splits them now and sticks: the next
 * run reads those answers before it groups anything (db/authored.py
 * `reject_duplicate`). A correction that lasted until the next run would
 * be a chore for ever.
 */
import { api, refusal } from "./api";
import { say } from "./ask";
import { difference, paint, said } from "./diff";
import { closestFrom, requireData } from "./dom";

(() => {
  const groups = document.querySelector("[data-dupe-groups]");
  if (!(groups instanceof HTMLElement)) return;

  // Clicking a copy shows it on the canvas beside the deck. The picture
  // is the comparison -- the numbers only corroborate it -- so switching
  // between copies has to cost one click and no navigation.
  groups.addEventListener("click", (event) => {
    const pick = closestFrom(event.target, "[data-dupe-pick]", HTMLButtonElement);
    if (!pick) return;
    const group = pick.closest("[data-dupe-group]");
    if (!(group instanceof HTMLElement)) return;
    const shown = group.querySelector("[data-dupe-shown]");
    const open = group.querySelector("[data-dupe-open]");
    const title = group.querySelector("[data-dupe-title]");
    if (shown instanceof HTMLImageElement) {
      shown.src = requireData(pick, "thumb");
      shown.alt = requireData(pick, "name");
    }
    if (open instanceof HTMLAnchorElement) open.href = requireData(pick, "href");
    if (title instanceof HTMLElement) title.textContent = requireData(pick, "name");
    // One pressed at a time: the deck is a choice, not a set of toggles.
    for (const other of group.querySelectorAll("[data-dupe-pick]")) {
      other.setAttribute("aria-pressed", String(other === pick));
    }
    void compare(group, pick);
  });

  // Photo or difference. The heat is always computed; this decides
  // whether the picture under it stays lit.
  groups.addEventListener("click", (event) => {
    const mode = closestFrom(event.target, "[data-dupe-mode]", HTMLButtonElement);
    if (!mode) return;
    const group = mode.closest("[data-dupe-group]");
    if (!(group instanceof HTMLElement)) return;
    const canvas = group.querySelector("[data-dupe-canvas]");
    if (!(canvas instanceof HTMLElement)) return;
    const wanted = requireData(mode, "dupeMode");
    canvas.dataset.mode = wanted;
    for (const other of group.querySelectorAll("[data-dupe-mode]")) {
      other.setAttribute("aria-pressed", String(other === mode));
    }
  });

  /**
   * Measure the picked copy against the one the sweep chose, and draw the
   * result over the picture.
   *
   * The chosen copy compared with itself is nothing, so the heat map and
   * its sentence are hidden there rather than drawn as an empty frame.
   */
  async function compare(group: HTMLElement, pick: HTMLElement): Promise<void> {
    const canvas = group.querySelector("[data-dupe-canvas]");
    const heat = group.querySelector("[data-dupe-heat]");
    const measure = group.querySelector("[data-dupe-measure]");
    if (!(canvas instanceof HTMLElement) || !(heat instanceof HTMLCanvasElement)) return;
    if (!(measure instanceof HTMLElement)) return;

    const readout = group.querySelector("[data-dupe-readout]");
    const figure = group.querySelector("[data-dupe-figure]");
    if (!(readout instanceof HTMLElement) || !(figure instanceof HTMLElement)) return;

    const best = requireData(canvas, "best");
    const shown = requireData(pick, "thumb");
    if (shown === best) {
      // A picture compared with itself is not a comparison. Say which one
      // this is rather than drawing an empty frame.
      canvas.dataset.mode = "photo";
      canvas.dataset.same = "";
      readout.hidden = false;
      figure.textContent = "★";
      measure.textContent = "the copy every other one is measured against";
      for (const m of group.querySelectorAll("[data-dupe-mode]")) {
        m.setAttribute("aria-pressed", String(requireData(m as HTMLElement, "dupeMode") === "photo"));
      }
      return;
    }
    delete canvas.dataset.same;
    readout.hidden = false;
    figure.textContent = "…";
    measure.textContent = "measuring";
    try {
      const found = await difference(best, shown);
      paint(heat, found.heat);
      figure.textContent = found.moved < 0.0005 ? "0%" : `${(found.moved * 100).toFixed(found.moved < 0.01 ? 2 : 0)}%`;
      measure.textContent = said(found);
    } catch (why) {
      // A picture that will not load, or one served from somewhere else:
      // say which rather than leaving the last answer on screen.
      figure.textContent = "—";
      measure.textContent = why instanceof Error ? why.message : "could not compare these";
    }
  }

  // Delegated: the page is server-rendered today and paged tomorrow, and
  // a listener per member would miss whatever arrives later.
  groups.addEventListener("click", async (event) => {
    const button = closestFrom(event.target, "[data-not-a-duplicate]", HTMLButtonElement);
    if (!button) return;
    const one = requireData(button, "notADuplicate");
    const other = requireData(button, "against");
    button.disabled = true;
    const held = await api.POST("/dupes/{slug}/not-a-duplicate", {
      params: { path: { slug: one } },
      body: { other },
    });
    if (held.error) {
      button.disabled = false;
      await say(refusal(held.error, "that was not recorded"));
      return;
    }
    // The member leaves the group it is no longer in. Its own row, not
    // the group: the rest of the group is unchanged by one member being
    // wrong about one comparison.
    const member = button.closest("[data-dupe-member]");
    if (member instanceof HTMLElement) member.remove();
  });
})();
