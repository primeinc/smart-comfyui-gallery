/**
 * The generation recipe: readable, editable, and copyable.
 *
 * Two things this surface has to get right.
 *
 * The text is a SCRATCH COPY. Somebody looking at a generated picture
 * wants the prompt in order to take it somewhere else -- usually with a
 * word changed, a clause deleted, a LoRA dropped. So the fields are real
 * textareas: caret, selection, undo, Ctrl-A, delete the lot. And none of
 * it is saved. What a file says it is is a FACT, not a preference; the
 * next picture brings the file's own text back, and nothing recorded
 * moves. `revert` puts this picture's text back without walking away.
 *
 * Copying has to produce something USABLE. A "copy all" that omits LoRA
 * weights or the sampler is the complaint people have about every other
 * gallery's copy button -- it looks like it worked and the picture does
 * not come back. So the whole recipe is emitted in the shape the tools
 * that made it already read, and it copies what is ON SCREEN: an edited
 * prompt is the one you meant to take.
 */

import { everyElement, findElement } from "./dom";

/**
 * Grow a scratch field to its content, so nothing is hidden behind a
 * scrollbar.
 *
 * A field that is not laid out is left alone. `scrollHeight` of an
 * element inside a closed `<details>` -- or inside an inspector CSS has
 * set to `display: none` -- is zero, and writing that back collapses the
 * prompt to nothing the first time somebody discloses it. `clientWidth`
 * is the cheapest honest test for "the browser has placed this".
 */
function fit(field: HTMLTextAreaElement): void {
  if (field.clientWidth === 0) return;
  field.style.height = "auto";
  field.style.height = `${field.scrollHeight}px`;
}

/**
 * Put text on the clipboard and say so on the button that asked.
 *
 * The confirmation is the button's own label for a moment. A toast would
 * be a second thing to look at somewhere else on the screen for an
 * action whose whole point is that it was instant.
 */
async function copied(button: HTMLElement, text: string): Promise<void> {
  const was = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "copied";
    button.dataset.done = "true";
  } catch {
    // Denied permission, or an insecure origin. Say which way it went
    // rather than leaving a button that looks like it worked.
    button.textContent = "cannot copy";
  }
  setTimeout(() => {
    button.textContent = was;
    delete button.dataset.done;
  }, 1200);
}

/** The text of one named field as it currently reads on screen. */
function scratchOf(root: HTMLElement, named: string): string {
  const section = findElement(root, `[data-recipe-field="${named}"]`, HTMLElement);
  const field = section && findElement(section, "[data-scratch]", HTMLTextAreaElement);
  return field ? field.value : "";
}

/**
 * The whole recipe as text, in the shape the tools that read these
 * already use: prompt, then `Negative prompt:`, then one line of
 * `Key: value` pairs. A1111 wrote it, everything else learned to read
 * it, and a person pasting into any of them gets the picture back.
 */
function wholeRecipe(root: HTMLElement): string {
  const lines: string[] = [];
  const prompt = scratchOf(root, "prompt");
  if (prompt.trim()) lines.push(prompt.trim());
  const negative = scratchOf(root, "negative");
  if (negative.trim()) lines.push(`Negative prompt: ${negative.trim()}`);

  // The keys the READERS of this format match on, carried in the markup
  // beside the lowercase ones a person reads (see _media_inspector.html).
  const pairs: string[] = [];
  for (const value of everyElement(root, "[data-recipe-key]", HTMLElement)) {
    const key = value.dataset.recipeKey ?? "";
    const text = (value.dataset.recipeValue ?? value.textContent ?? "").trim();
    if (key && text) pairs.push(`${key}: ${text}`);
  }
  const checkpoint = findElement(root, "[data-recipe-checkpoint]", HTMLElement);
  if (checkpoint) pairs.push(`Model: ${(checkpoint.textContent ?? "").trim()}`);

  // Every LoRA WITH its weight. A name alone does not reproduce the
  // picture, and this is the field people report missing from other
  // galleries' copy buttons.
  //
  // Unless the prompt already carries it: A1111 and everything that
  // copied it write LoRAs INLINE as `<lora:name:weight>`, so a picture
  // made that way would otherwise be handed back its own tags twice --
  // once in the prompt and once in a trailing list -- and a paste of
  // that applies each LoRA a second time.
  const loras = [...everyElement(root, ".recipe-lora", HTMLElement)]
    .map((row) => {
      const name = row.querySelector("span:not(.recipe-label)")?.textContent?.trim() ?? "";
      const weight = row.querySelector(".recipe-weight")?.textContent?.trim();
      return { name, tag: weight ? `<lora:${name}:${weight}>` : `<lora:${name}>` };
    })
    .filter((lora) => lora.name && !prompt.includes(`<lora:${lora.name}`))
    .map((lora) => lora.tag);
  if (loras.length) pairs.push(`Loras: ${loras.join(" ")}`);

  if (pairs.length) lines.push(pairs.join(", "));
  return lines.join("\n");
}

/** Wire one recipe panel. Safe to call on a panel that has none. */
export function mountRecipe(root: HTMLElement): void {
  const panel = findElement(root, "[data-recipe]", HTMLElement);
  if (!panel) return;

  for (const section of everyElement(panel, "[data-recipe-field]", HTMLElement)) {
    const field = findElement(section, "[data-scratch]", HTMLTextAreaElement);
    const revert = findElement(section, "[data-revert]", HTMLElement);
    if (!field) continue;
    const original = field.value;

    fit(field);
    field.addEventListener("input", () => {
      fit(field);
      // The way back only appears once there is something to go back
      // from, so a panel nobody has touched carries no spare control.
      if (revert) revert.hidden = field.value === original;
    });

    if (revert) {
      revert.addEventListener("click", () => {
        field.value = original;
        fit(field);
        revert.hidden = true;
        field.focus();
      });
    }

    const copy = findElement(section, "[data-copy]", HTMLElement);
    if (copy) copy.addEventListener("click", () => void copied(copy, field.value));
  }

  const all = findElement(panel, "[data-copy-all]", HTMLElement);
  if (all) all.addEventListener("click", () => void copied(all, wholeRecipe(panel)));

  // The two ways this panel arrives on screen without any of its fields
  // being touched: the section is disclosed, or the whole inspector is
  // opened. Both are the first moment `fit` can measure anything, so both
  // re-run it -- otherwise a prompt discloses one line tall and stays
  // that way until somebody types in it.
  const refit = () => {
    for (const field of everyElement(panel, "[data-scratch]", HTMLTextAreaElement)) fit(field);
  };
  panel.addEventListener("toggle", refit);
  new MutationObserver(refit).observe(root, { attributeFilter: ["data-inspector"] });
  refit();
}
