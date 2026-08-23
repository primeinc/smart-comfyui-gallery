// The socket's edge: a message, or nothing.
//
// `JSON.parse` returns `any`. A type annotation over it is a promise, not a
// proof -- the compiler stops asking questions and the first malformed frame
// becomes an `undefined` three functions away. Everything below exists so
// that promise is cashed exactly once, here, and the console downstream of
// `decodeFrame` only ever handles a value whose shape has been looked at.
//
// The frame types themselves are generated from the application's own
// contract (sg_web/console.py Frame, carried into the document by
// socket_frames()); nothing in this file restates them.
import type { components } from "./generated/api";

export type Event = components["schemas"]["Event"];
export type EventFrame = components["schemas"]["EventFrame"];
export type PendingFrame = components["schemas"]["PendingFrame"];
export type BacklogFrame = components["schemas"]["BacklogFrame"];

/** What arrives on /ws/events, discriminated on `frame`. */
export type Frame = EventFrame | PendingFrame | BacklogFrame;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const num = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const str = (value: unknown): value is string => typeof value === "string";
const numOrNull = (value: unknown): value is number | null => value === null || num(value);
const strOrNull = (value: unknown): value is string | null => value === null || str(value);
const dataOrNull = (value: unknown): value is Record<string, unknown> | null => value === null || isRecord(value);

/**
 * Everything a committed row and a live report share (sg_web/console.py
 * Reported), each field checked against what the console reads it as.
 *
 * `type` and `severity` are proven to be strings and no further. The console
 * uses both as strings -- `startsWith("phase.")`, a comparison, an attribute
 * -- so nothing downstream depends on the member being one of the seventeen;
 * closing that set is the server's CHECK constraint, held to the generated
 * union by sglint SG709, and this function does not restate it.
 */
function reported(held: Record<string, unknown>): boolean {
  return (
    num(held.job_id) &&
    num(held.at) &&
    str(held.type) &&
    numOrNull(held.item_id) &&
    strOrNull(held.phase) &&
    str(held.severity) &&
    strOrNull(held.message) &&
    dataOrNull(held.data) &&
    str(held.text) &&
    strOrNull(held.condition)
  );
}

function isEvent(value: unknown): value is Event {
  return isRecord(value) && reported(value) && num(value.id);
}

function isEventFrame(value: unknown): value is EventFrame {
  return isRecord(value) && value.frame === "event" && isEvent(value);
}

function isPendingFrame(value: unknown): value is PendingFrame {
  return isRecord(value) && value.frame === "pending" && reported(value);
}

function isBacklogFrame(value: unknown): value is BacklogFrame {
  return (
    isRecord(value) &&
    value.frame === "backlog" &&
    num(value.after) &&
    num(value.last_id) &&
    Array.isArray(value.events) &&
    value.events.every(isEvent)
  );
}

/**
 * One socket message as a frame, or null when it is not one.
 *
 * Null is the honest answer for a payload that is not text, is not JSON, or
 * carries a `frame` this build does not know: the caller decides what to do
 * about it, where a throw here would kill the socket over one bad row.
 */
export function decodeFrame(payload: unknown): Frame | null {
  if (typeof payload !== "string") return null;
  let held: unknown;
  try {
    held = JSON.parse(payload);
  } catch {
    return null;
  }
  if (isEventFrame(held)) return held;
  if (isPendingFrame(held)) return held;
  if (isBacklogFrame(held)) return held;
  return null;
}
