/**
 * What actually changed between two copies of one photo.
 *
 * The duplicates page can say "6 different files" from the hashes alone.
 * It could not say WHAT differs, and that is the question somebody has
 * when deciding which copy to keep: a re-save that changed nothing they
 * can see, or a crop that lost the left third.
 *
 * This reads the pixels. Both copies are drawn to an offscreen canvas at
 * one size, `getImageData` returns both, and the per-pixel difference is
 * mapped to a heat ramp: violet where a channel moved a little, orange
 * where it moved a lot. The same two colours the rest of the product
 * uses, so the ramp needs no legend.
 *
 * Canvas, and not `mix-blend-mode: difference`, for one reason: blending
 * draws a difference and cannot count one. The numbers under the picture
 * -- how much of the frame moved, and by how much at its worst -- are
 * the half that makes the picture worth showing, and only pixels you can
 * read back can produce them.
 *
 * Same-origin only. The thumbnails are served from this application
 * (`/thumbs/...`), so the canvas is never tainted and `getImageData` is
 * allowed. A cross-origin picture would throw a SecurityError, so the
 * caller is given the reason rather than a blank canvas.
 */

/** How far apart two channel values must be before a pixel counts as changed. */
const MOVED = 8;

/** The longest edge the comparison is done at. Two 1440px thumbnails is
 *  8M pixels of work for an answer that reads the same at 512. */
const SAMPLE = 512;

export interface Difference {
  /** Fraction of the frame whose colour moved at all, 0..1. */
  moved: number;
  /** The largest single-channel change anywhere, 0..255. */
  worst: number;
  /** Mean change across every pixel, 0..255. */
  mean: number;
  /** The heat map, ready to draw. */
  heat: ImageData;
}

const load = (src: string): Promise<HTMLImageElement> =>
  new Promise((ok, no) => {
    const img = new Image();
    img.decoding = "async";
    img.addEventListener("load", () => ok(img));
    img.addEventListener("error", () => no(new Error(`could not load ${src}`)));
    img.src = src;
  });

/** Both pictures at one size, so pixels can be compared position by position. */
function drawn(img: HTMLImageElement, w: number, h: number): ImageData {
  const board = document.createElement("canvas");
  board.width = w;
  board.height = h;
  const hand = board.getContext("2d", { willReadFrequently: true });
  if (!hand) throw new Error("no 2d context");
  // `cover`, the same fit the page shows them at: a letterboxed pair would
  // report the bars as identical and dilute the number.
  const scale = Math.max(w / img.naturalWidth, h / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  hand.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  return hand.getImageData(0, 0, w, h);
}

/**
 * Compare two pictures and return the heat map plus what it measured.
 *
 * The ramp is violet -> orange because those are the product's two
 * colours: a little movement reads as the quiet one, a lot as the loud
 * one. Unchanged pixels are left transparent so the heat sits over the
 * picture rather than replacing it.
 */
export async function difference(a: string, b: string): Promise<Difference> {
  const [one, two] = await Promise.all([load(a), load(b)]);
  const ratio = one.naturalWidth / one.naturalHeight;
  const w = Math.max(1, Math.round(ratio >= 1 ? SAMPLE : SAMPLE * ratio));
  const h = Math.max(1, Math.round(ratio >= 1 ? SAMPLE / ratio : SAMPLE));

  const left = drawn(one, w, h);
  const right = drawn(two, w, h);
  const heat = new ImageData(w, h);

  let moved = 0;
  let worst = 0;
  let total = 0;
  // Bound to locals: indexing a typed array is `number | undefined` under
  // this project's TypeScript settings, and `?? 0` is the honest reading
  // -- a byte past the end of the buffer is not a colour.
  const one8 = left.data;
  const two8 = right.data;
  for (let i = 0; i < one8.length; i += 4) {
    const dr = Math.abs((one8[i] ?? 0) - (two8[i] ?? 0));
    const dg = Math.abs((one8[i + 1] ?? 0) - (two8[i + 1] ?? 0));
    const db = Math.abs((one8[i + 2] ?? 0) - (two8[i + 2] ?? 0));
    const delta = Math.max(dr, dg, db);
    total += delta;
    if (delta > worst) worst = delta;
    if (delta < MOVED) continue;
    moved += 1;
    // 0..1 across the range that is left after the threshold, so a pixel
    // that only just counts is not drawn as loudly as one that changed
    // completely.
    const t = Math.min(1, (delta - MOVED) / (255 - MOVED));
    // violet #b79cff -> orange #f0913c
    heat.data[i] = Math.round(183 + (240 - 183) * t);
    heat.data[i + 1] = Math.round(156 + (145 - 156) * t);
    heat.data[i + 2] = Math.round(255 + (60 - 255) * t);
    heat.data[i + 3] = Math.round(90 + 165 * t);
  }

  const pixels = one8.length / 4;
  return { moved: moved / pixels, worst, mean: total / pixels, heat };
}

/** Draw a heat map into a visible canvas, sized for the screen it is on. */
export function paint(board: HTMLCanvasElement, heat: ImageData): void {
  board.width = heat.width;
  board.height = heat.height;
  const hand = board.getContext("2d");
  if (!hand) return;
  hand.putImageData(heat, 0, 0);
}

/** What the numbers mean, in a sentence rather than three figures. */
export function said(found: Difference): string {
  if (found.moved < 0.0005) return "identical to the eye — nothing moved";
  const part = found.moved < 0.01 ? `${(found.moved * 100).toFixed(2)}%` : `${Math.round(found.moved * 100)}%`;
  const how = found.worst > 160 ? "heavily" : found.worst > 60 ? "noticeably" : "slightly";
  return `${part} of the frame changed, ${how} (worst ${found.worst} of 255)`;
}
