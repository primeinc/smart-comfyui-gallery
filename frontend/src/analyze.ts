/**
 * The analysis panel: what this answer is made of.
 *
 * Almost nothing runs here, and that is the point. Every bar, share and
 * count is rendered by the server from the same membership the grid
 * reads, and every one of them is an ordinary link that adds a clause to
 * the question -- so refining an analysis is a navigation, works with
 * the middle mouse button, and survives with JavaScript switched off.
 *
 * What a browser is actually needed for is the clipboard.
 */

import { everyElement, findElement } from "./dom";
import { copied } from "./recipe";

/** Wire the analysis panel. Safe to call on a page that has none. */
export function mountAnalyze(root: HTMLElement): void {
  const panel = findElement(root, "[data-analyze]", HTMLElement);
  if (!panel) return;

  for (const button of everyElement(panel, "[data-copy-prompt]", HTMLElement)) {
    const use = button.closest("[data-prompt]");
    const text = use && findElement(use as HTMLElement, "[data-prompt-text]", HTMLElement);
    if (!text) continue;
    // textContent, not innerText: the prompt is rendered `white-space:
    // pre-wrap` and innerText would hand back the browser's idea of the
    // line breaks rather than the ones the file actually carries.
    button.addEventListener("click", () => void copied(button, text.textContent ?? ""));
  }
}
