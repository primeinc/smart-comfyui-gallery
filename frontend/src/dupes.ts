/**
 * Disagreeing with a duplicate group.
 *
 * A perceptual group is a GUESS. pHash sees global low-frequency
 * composition, so two photographs of one scene a second apart are close
 * in it and land in one group -- and the page that shows them had no way
 * to say they are two pictures.
 *
 * Saying it takes them apart NOW and survives the next sweep, which
 * reads the verdicts back before it writes a group (db/authored.py
 * `reject_duplicate`). A correction that lasted only until the next run
 * would be a chore repeated for ever, which is the same reason denying a
 * person is a claim rather than a retraction.
 */
import { api, refusal } from "./api";
import { say } from "./ask";
import { closestFrom, requireData } from "./dom";

(() => {
  const groups = document.querySelector("[data-dupe-groups]");
  if (!(groups instanceof HTMLElement)) return;

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
