// The job feed's edge: a message, or nothing.
//
// The same discipline as frames.ts over /ws/events, and for the same reason:
// `JSON.parse` returns `any`, and a type annotation over it is a promise
// rather than a proof. What this file returns is what it checked.
//
// The frame types are generated from the application's contract
// (sg_web/app.py JobFrame, carried into the document by job_frames()).
// Nothing here restates them -- and in particular the thirteen job kinds and
// five job states are not written out in TypeScript, which is why the
// decoded types widen those two fields to `string`.
import type { components } from "./generated/api";

type JobListed = components["schemas"]["JobListed"];
type JobsSnapshotFrame = components["schemas"]["JobsSnapshotFrame"];
type JobDeltaFrame = components["schemas"]["JobDeltaFrame"];

/**
 * A frame as this decoder can honestly hand it over.
 *
 * `kind` and `state` are closed vocabularies in the contract, held to the
 * schema's CHECK by sglint SG709 through db/jobs.py. Proving membership here
 * would mean a second encoding of both in authored TypeScript -- the exact
 * duplication the generated contract exists to remove -- so the decoder
 * proves `typeof === "string"` and says so.
 */
type Readable<T extends { kind: string; state: string }> = Omit<T, "kind" | "state"> & {
  kind: string;
  state: string;
};

export type ReadableJob = Readable<JobListed>;
export type ReadableJobDelta = Readable<JobDeltaFrame>;
export type ReadableJobsSnapshot = Omit<JobsSnapshotFrame, "jobs"> & { jobs: ReadableJob[] };

/** What arrives on /ws/jobs, discriminated on `type`. */
export type JobFrame = ReadableJobsSnapshot | ReadableJobDelta;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const num = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const str = (value: unknown): value is string => typeof value === "string";
const bool = (value: unknown): value is boolean => typeof value === "boolean";
const numOrNull = (value: unknown): value is number | null => value === null || num(value);
const strOrNull = (value: unknown): value is string | null => value === null || str(value);

function isJob(value: unknown): value is ReadableJob {
  return (
    isRecord(value) &&
    num(value.id) &&
    str(value.kind) &&
    str(value.state) &&
    bool(value.cancel_requested) &&
    numOrNull(value.total) &&
    num(value.done_count) &&
    num(value.created_at) &&
    numOrNull(value.finished_at) &&
    strOrNull(value.derive)
  );
}

function isSnapshot(value: unknown): value is ReadableJobsSnapshot {
  return isRecord(value) && value.type === "snapshot" && Array.isArray(value.jobs) && value.jobs.every(isJob);
}

function isDelta(value: unknown): value is ReadableJobDelta {
  return (
    isRecord(value) &&
    value.type === "delta" &&
    num(value.job) &&
    str(value.kind) &&
    str(value.state) &&
    num(value.done) &&
    numOrNull(value.total) &&
    bool(value.cancel_requested) &&
    strOrNull(value.derive)
  );
}

/**
 * One socket message as a frame, or null when it is not one.
 *
 * Null is a transport failure, not a skippable message. The rows are the
 * system of record here as they are for the ledger, so the answer is
 * the same one: reconnect, and the feed opens with a fresh snapshot
 * read from them.
 */
export function decodeJobFrame(payload: unknown): JobFrame | null {
  if (typeof payload !== "string") return null;
  let held: unknown;
  try {
    held = JSON.parse(payload);
  } catch {
    return null;
  }
  if (isSnapshot(held)) return held;
  if (isDelta(held)) return held;
  return null;
}
