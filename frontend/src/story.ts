// The story page: the same plan under another profile is one POST to
// /stories/renders -- content-addressed, so asking twice is one row --
// and the page moves to the render it names.
import { api, refusal } from "./api";
import { everyElement, findElement, requireData } from "./dom";
import type { components } from "./generated/api";

type RenderProfile = components["schemas"]["RenderRequest"]["profile"];

/**
 * The voices a render can speak in, proven rather than asserted.
 *
 * `data-story-profile-ask` is markup, so its value is a string at runtime;
 * sglint SG709 holds the Python Literal against the schema's CHECK, and
 * this holds the button against the Literal.
 */
const asProfile = (held: string): RenderProfile => {
  if (held !== "memory" && held !== "technical" && held !== "compact") {
    throw new Error(`the page offered the profile ${held}, which no render speaks`);
  }
  return held;
};

(() => {
  const main = findElement(document, "[data-story-render]", HTMLElement);
  if (!main) return;
  const status = findElement(document, "[data-story-status]", HTMLElement);
  const plan_id = Number(requireData(main, "storyPlan"));
  const locale = requireData(main, "storyLocale");

  for (const button of everyElement(document, "[data-story-profile-ask]", HTMLElement)) {
    button.addEventListener("click", async () => {
      const profile = asProfile(requireData(button, "storyProfileAsk"));
      if (status) status.textContent = `rendering ${profile}…`;
      const { data, error } = await api.POST("/stories/renders", { body: { plan_id, profile, locale } });
      if (!data) {
        if (status) status.textContent = refusal(error, "that render was refused");
        return;
      }
      window.location.href = `/stories/renders/${data.id}`;
    });
  }
})();
