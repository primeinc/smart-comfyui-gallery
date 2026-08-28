/**
 * A picture that will not load says what it is instead of breaking.
 *
 * The server already avoids pointing at a picture that cannot exist:
 * `thumbs.asset_url` answers None for a medium with no picture to take,
 * and the cell draws the kind instead. That is right, and it depends on
 * the row being right about the kind.
 *
 * A row can be wrong. An `.m4a` is ISO-BMFF, mimesniff calls it
 * video/mp4, and a folder of album tracks was stored as video -- so the
 * server minted thumbnail addresses in good faith and every one failed.
 * This is the second line, and it is about the symptom rather than about
 * m4a: any thumbnail that fails, for any reason the page cannot know,
 * degrades to the same grey label the server would have drawn.
 *
 * ONE LISTENER, on the document, in the capture phase. `error` does not
 * bubble, so a document listener sees it only on the way down -- which
 * is also why one listener covers images that do not exist yet: an
 * endless grid, a swapped fragment, a filmstrip remounted every step.
 *
 * ONE SWEEP as well, because a listener hears only what has not happened
 * yet. An image that failed before this module ran dispatched `error` to
 * nobody and will not mention it again, so it would keep the broken icon
 * for the life of the page. Not a corner case: a cached thumbnail fails
 * immediately, while this bundle is still being fetched.
 * `complete && naturalWidth === 0` is what a finished failure looks like.
 */

/** What a cell says when there is no picture: the shape the server uses. */
function label(kind: string | undefined): HTMLElement {
  const said = document.createElement("span");
  said.className = "cell-kind";
  said.dataset.cellKind = kind ?? "";
  said.dataset.brokenPicture = "";
  said.setAttribute("aria-hidden", "true");
  // The same two words the templates use, and the same fallback: a kind
  // nothing anticipated reads as a document rather than as a blank.
  said.textContent = kind === "audio" ? "audio" : "doc";
  return said;
}

/** Swap one failed thumbnail for the words it should have been.
 *
 * The caller establishes that it HAS failed -- an `error` event, or
 * `complete && naturalWidth === 0` for one that failed before the
 * listener existed. This looks at the src, not at the load state. */
function degrade(broken: HTMLImageElement): void {
  // Only pictures OF something in the library. A decorative image, a
  // background, an avatar that has its own fallback -- none of those
  // want a "doc" label dropped where they were.
  const src = broken.getAttribute("src") ?? "";
  if (!src.startsWith("/thumbs/") && !src.startsWith("/thumb/") && !src.startsWith("/preview/")) return;
  // Guard against a swap that itself fails: replaceWith detaches the
  // image, so a second error on the same node cannot arrive -- but a
  // page that re-renders could hand us one already replaced.
  if (!broken.isConnected) return;

  // The kind is on the cell, the row or the frame that holds it --
  // whichever this page uses -- and absent on surfaces that do not
  // carry one, where the label still reads honestly.
  const holder = broken.closest("[data-kind]");
  const kind = holder instanceof HTMLElement ? holder.dataset.kind : undefined;
  broken.replaceWith(label(kind));
}

export function mountPictures(): void {
  document.addEventListener(
    "error",
    (event) => {
      const broken = event.target;
      if (!(broken instanceof HTMLImageElement)) return;
      degrade(broken);
    },
    // The capture phase, because `error` on an <img> does not bubble.
    true,
  );

  // Whatever already failed while this bundle was on its way. `complete`
  // is true for a finished load AND for a finished failure; the two are
  // told apart by `naturalWidth`, which stays 0 when nothing decoded.
  // An image still in flight is not complete and belongs to the listener.
  for (const picture of document.querySelectorAll("img")) {
    if (picture.complete && picture.naturalWidth === 0) degrade(picture);
  }
}
