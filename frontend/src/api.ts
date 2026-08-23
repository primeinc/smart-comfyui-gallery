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
// site has to say what it does about refusal before it can reach the answer.
import createClient from "openapi-fetch";
import type { paths } from "./generated/api";

export const api = createClient<paths>();

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
