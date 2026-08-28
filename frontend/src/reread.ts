/**
 * "Read these files again", on the folder somebody is looking at.
 *
 * Re-reading is how this application corrects itself. Improving a parser
 * is a re-parse of the database -- the schema says so at `param_key` --
 * and the sniffer that decides a file's KIND is the part most likely to
 * improve, because it is the part that has to keep up with what
 * cameras, phones and generators actually write.
 *
 * The whole-library sweep already exists on the operations page and
 * costs what it costs. This is the one somebody actually runs: bounded
 * to the folder they are standing in, which is where they can SEE the
 * problem. A correction too expensive to apply is not a correction.
 */
import { api, refusal } from "./api";
import { say } from "./ask";
import { everyElement, requireData } from "./dom";

/** How often to ask a small job whether it is finished. */
const POLL_MS = 400;

export function mountReread(root: ParentNode): void {
  for (const button of everyElement(root, "[data-folder-reread]", HTMLButtonElement)) {
    button.addEventListener("click", async () => {
      const folder = requireData(button, "folderReread");
      button.disabled = true;
      const was = button.textContent;
      button.textContent = "queueing…";
      const { data, error, response } = await api.POST("/jobs/ingest", {
        params: { query: { everything: true, folder } },
      });
      // 204 is "nothing to do", which openapi-fetch reports as no data
      // and no error. Saying so is the point: a button that looked like
      // it failed when the answer was "already read" is why the state
      // is drawn from the response rather than from the click.
      if (!data && response.status !== 204) {
        button.disabled = false;
        button.textContent = was;
        await say(refusal(error, "the re-read was not queued"));
        return;
      }
      if (response.status === 204) {
        button.textContent = "already read";
        return;
      }
      // It IS a job -- the same per-file work the whole-library sweep
      // does, so it has to be resumable, cancellable and able to survive
      // a restart, and four thousand files inline would hold the request
      // open. But a job is not a reason to send somebody to another
      // page: a folder of twenty is done in a second, and "watch it in
      // operations" asks them to go and look for something that already
      // finished. So it reports HERE, and the operations console is
      // where it goes if they want the detail.
      const held = data as { id?: number } | undefined;
      const job = held?.id;
      if (job === undefined) {
        button.textContent = "queued";
        return;
      }
      for (;;) {
        const told = await api.GET("/jobs/{job_id}", { params: { path: { job_id: job } } });
        const state = told.data?.state;
        if (state === undefined) {
          button.textContent = "queued";
          return;
        }
        if (state === "done" || state === "failed" || state === "cancelled") {
          const failed = told.data?.failed_count ?? 0;
          button.textContent =
            state === "done" && !failed
              ? `read ${told.data?.done_count ?? 0} again`
              : `${state}${failed ? ` — ${failed} could not be read` : ""}`;
          return;
        }
        button.textContent = `reading… ${told.data?.done_count ?? 0} of ${told.data?.total ?? "?"}`;
        await new Promise((wake) => setTimeout(wake, POLL_MS));
      }
    });
  }
}
