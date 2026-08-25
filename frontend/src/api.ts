// The application's HTTP surface, as the browser sees it: one typed client
// over the contract the application generates from its own handlers.
//
// Same origin, so there is no base URL to configure -- these paths ARE the
// application's paths. What the generated `paths` type buys is that a typo in
// one, a method the route does not serve, a missing path parameter, a body
// field that is not in the model, or a response field nobody promised are all
// compile errors here rather than a 404 or an `undefined` a user finds first.
//
// `data` is present only on a 2XX and `error` only on a 4XX/5XX, so a call
// site has to say what it does about refusal before it can reach the response.
import createClient from "openapi-fetch";
import type { paths } from "./generated/api";

export const api = createClient<paths>();

/**
 * A response, or what the server said instead.
 *
 * openapi-fetch hands back `{ data, error }` where each is present only in
 * its own case, which a caller can quietly destructure down to `{ data }` --
 * and then a refused write is `undefined` and the click looks like it did
 * nothing. Narrowing on `ok` makes the refusal impossible to skip past: the
 * response is not reachable until the caller has dealt with the alternative.
 *
 * Deliberately not an exception. A server refusing a rating of nine is the
 * system working, not the program failing, and throwing would put ordinary
 * outcomes on the same path as bugs.
 */
export type Answered<T> = { readonly ok: true; readonly data: T } | { readonly ok: false; readonly refusal: string };

export function answered<T>(result: { data?: T | undefined; error?: unknown }, fallback: string): Answered<T> {
  if (result.data !== undefined) return { ok: true, data: result.data };
  return { ok: false, refusal: refusal(result.error, fallback) };
}

/**
 * What the server said about a refusal, or `fallback` if it said nothing.
 *
 * Litestar's error body carries `detail`. It is not part of any response
 * model, so this proves the shape at runtime rather than asserting it: an
 * error that turns out to be a string, a network failure, or a body that is
 * not JSON at all all land on the fallback instead of printing "undefined".
 */
export function refusal(error: unknown, fallback: string): string {
  if (typeof error === "object" && error !== null && "detail" in error && typeof error.detail === "string") {
    return error.detail;
  }
  return fallback;
}
