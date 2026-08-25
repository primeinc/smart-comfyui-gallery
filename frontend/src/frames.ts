// The socket's edge: a message, or nothing.
//
// `JSON.parse` returns `any`. A type annotation over it is a promise, not a
// proof -- the compiler stops asking questions and the first malformed frame
// becomes an `undefined` three functions away. Everything below exists so
// that promise is cashed exactly once, here.
//
// The frame types are generated from the application's own contract
// (sg_web/console.py Frame, carried into the document by socket_frames());
// nothing here restates them. What this file DOES restate is the difference
// between the contract and what a local decoder can prove about one message,
// which is the whole reason `Readable` exists below.
import type { components } from "./generated/api";

type Event = components["schemas"]["Event"];
type EventFrame = components["schemas"]["EventFrame"];
type PendingFrame = components["schemas"]["PendingFrame"];
type BacklogFrame = components["schemas"]["BacklogFrame"];

/**
 * A frame as this decoder can honestly hand it over.
 *
 * The contract closes `type` to seventeen members and `severity` to three.
 * Proving that here would mean writing those twenty strings into authored
 * TypeScript -- a second encoding of the vocabulary that db/ledger.py owns
 * and sglint SG709 holds to the schema's CHECK, which is exactly the
 * duplication this whole seam exists to remove. So the decoder proves what
 * it can, `typeof === "string"`, and says so in the type: a caller gets
 * every structural guarantee the contract makes and no claim about
 * membership that was never checked.
 *
 * The consequence is deliberate. An exhaustive `switch` over event types
 * will not compile against these, because this decoder cannot promise the
 * set is closed. Something that needs that promise must get the vocabulary
 * generated into runtime code from db/ledger.py, not copied by hand.
 */
type Readable<T extends { type: string; severity: string }> = Omit<T, "type" | "severity"> & {
  type: string;
  severity: string;
};

export type ReadableEvent = Readable<Event>;
export type ReadableEventFrame = Readable<EventFrame>;
export type ReadablePendingFrame = Readable<PendingFrame>;
export type ReadableBacklogFrame = Omit<BacklogFrame, "events"> & { events: ReadableEvent[] };

/** What arrives on /ws/events, discriminated on `frame`. */
export type Frame = ReadableEventFrame | ReadablePendingFrame | ReadableBacklogFrame;

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
 * Reported), each field checked against what it is declared to be.
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

function isEvent(value: unknown): value is ReadableEvent {
  return isRecord(value) && reported(value) && num(value.id);
}

function isEventFrame(value: unknown): value is ReadableEventFrame {
  return isRecord(value) && value.frame === "event" && isEvent(value);
}

function isPendingFrame(value: unknown): value is ReadablePendingFrame {
  return isRecord(value) && value.frame === "pending" && reported(value);
}

function isBacklogFrame(value: unknown): value is ReadableBacklogFrame {
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
 * Null means the transport is not speaking this protocol, and the caller
 * must treat it as a transport failure rather than as one skippable
 * message: an unreadable message is not evidence that the ledger rows
 * inside it may be discarded. A malformed `event` would eventually show up
 * as an id gap, but a malformed BACKLOG can carry hundreds of committed
 * rows and, if nothing newer is ever sent, leaves no gap to notice. The
 * durable ledger is what makes the right answer cheap: reconnect from the
 * last id held and every committed row arrives again.
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
