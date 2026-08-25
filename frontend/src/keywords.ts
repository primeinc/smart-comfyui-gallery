/**
 * Keeping the keyword vocabulary honest.
 *
 * A vocabulary is only worth having if one picture-idea gets one word,
 * and a year of typing produces "beach", "Beaches" and "beech" whether
 * anybody meant it to. The two gestures that fix that are here.
 *
 * Both writes answer with the WHOLE refreshed vocabulary rather than the
 * row they touched, and the list is redrawn from that answer. A fold
 * changes two rows -- one absorbs the other's pictures and one vanishes
 * -- so a page that patched the row it clicked would be reasoning about
 * what its own click must have done, which is exactly the thing the
 * authored strip already refuses to do.
 */
import { type Answered, answered, api } from "./api";
import { say } from "./ask";
import { closestFrom, requireData, requireElement } from "./dom";
import type { components } from "./generated/api";

type KeywordListed = components["schemas"]["KeywordListed"];

const draw = (list: HTMLElement, keywords: KeywordListed[]) => {
  list.replaceChildren(
    ...keywords.map((one) => {
      const row = document.createElement("li");
      row.className = "keyword-row";
      row.dataset.keyword = one.tag;
      row.dataset.pictures = String(one.pictures);

      const link = document.createElement("a");
      link.className = "keyword-name";
      link.href = `/g?${one.qs}`;
      link.textContent = one.label;

      const count = document.createElement("span");
      count.className = "keyword-count";
      count.textContent = `${one.pictures} picture${one.pictures === 1 ? "" : "s"}`;

      const form = document.createElement("form");
      form.className = "keyword-rename";
      form.dataset.rename = "";
      const box = document.createElement("input");
      box.type = "text";
      box.dataset.renameInput = "";
      box.maxLength = 100;
      box.autocomplete = "off";
      box.value = one.label;
      box.setAttribute("aria-label", `rename ${one.label}`);
      const go = document.createElement("button");
      go.type = "submit";
      go.textContent = "rename";
      form.append(box, go);

      const forget = document.createElement("button");
      forget.type = "button";
      forget.className = "keyword-forget";
      forget.dataset.forget = one.tag;
      forget.dataset.forgetPictures = String(one.pictures);
      forget.title = `take ${one.label} off all ${one.pictures}`;
      forget.textContent = "forget";

      row.append(link, count, form, forget);
      return row;
    }),
  );
};

const applied = async (list: HTMLElement, told: Answered<KeywordListed[]>) => {
  if (!told.ok) {
    await say(told.refusal);
    return;
  }
  draw(list, told.data);
};

(() => {
  const list = document.querySelector("[data-keywords]");
  if (!(list instanceof HTMLElement)) return;

  // Delegated, because the list is replaced wholesale after every write.
  list.addEventListener("submit", async (event) => {
    const form = closestFrom(event.target, "[data-rename]", HTMLElement);
    if (!form) return;
    event.preventDefault();
    const row = closestFrom(form, "[data-keyword]", HTMLElement);
    if (!row) return;
    const to = requireElement(form, "[data-rename-input]", HTMLInputElement).value.trim();
    // Unchanged is not a write. Submitting the value already in the box
    // is what pressing Enter to dismiss a field does, and it should cost
    // nothing rather than round-trip a rename onto itself.
    if (!to || to === row.querySelector(".keyword-name")?.textContent) return;
    await applied(
      list,
      answered(
        await api.POST("/keywords/rename", { body: { name: requireData(row, "keyword"), to } }),
        "the keyword could not be renamed",
      ),
    );
  });

  list.addEventListener("click", async (event) => {
    const button = closestFrom(event.target, "[data-forget]", HTMLElement);
    if (!button) return;
    const name = requireData(button, "forget");
    const pictures = Number(requireData(button, "forgetPictures"));
    // The only destroying gesture on this page, so it asks -- and the
    // question carries the NUMBER, because "forget beach" and "take
    // beach off 412 pictures" are different things to agree to.
    if (!window.confirm(`Take "${name}" off ${pictures} picture${pictures === 1 ? "" : "s"}? This cannot be undone.`)) {
      return;
    }
    await applied(
      list,
      answered(await api.POST("/keywords/forget", { body: { name, pictures } }), "the keyword could not be forgotten"),
    );
  });
})();
