// Selection: ephemeral browser state, anchored to ONE ResultSet answer.
//
//   { answer, selected: Set<entity uuid> }
//
// Selection belongs to an answer: it survives page swaps while the
// mounted grid's data-answer is unchanged (select on page 1, more on
// page 3, curate all of it at once), and clears the moment a different
// answer mounts -- a toolbar must never operate on files the current
// question no longer shows. Nothing here is durable, nothing rides the
// URL, and no membership or ordering is ever computed in the browser.
//
// Every action states ONE desired fact for the whole selection, sent as
// {answer, items, value} to a bulk route that proves the selection
// against the authoritative projection inside its write transaction.
// Settlement is the same answer-identity contract single writes use:
// an unchanged after-answer adopts the new currency and KEEPS the
// selection for the next operation; a changed one clears and redraws.
(() => {
  "use strict";

  const state = { answer: null, selected: new Set() };

  const grid = () => document.querySelector("[data-grid]");
  const bar = document.querySelector("[data-curate]");
  if (!bar) return;

  const draw = () => {
    bar.hidden = state.selected.size === 0;
    bar.querySelector("[data-curate-count]").textContent = `${state.selected.size} selected`;
    for (const shell of document.querySelectorAll("[data-selection-key]")) {
      const pick = shell.querySelector("[data-pick]");
      if (pick) pick.checked = state.selected.has(shell.dataset.selectionKey);
    }
  };

  const clear = () => {
    state.selected.clear();
    draw();
  };

  const sync = () => {
    const g = grid();
    if (!g) return;
    if (state.answer !== g.dataset.answer) {
      // A different answer mounted: the old selection named members of
      // a question that is no longer on screen.
      state.answer = g.dataset.answer;
      state.selected.clear();
    }
    draw();
  };

  const albums = bar.querySelector("[data-bulk-album]");
  const shelve = async () => {
    const asked = await fetch("/albums", { headers: { accept: "application/json" } });
    if (!asked.ok) return;
    const told = await asked.json();
    albums.replaceChildren(
      ...told
        .filter((one) => one.kind !== "smart") // rule-derived membership is not filed
        .map((one) => {
          const choice = document.createElement("option");
          choice.value = one.slug;
          choice.textContent = one.name;
          return choice;
        }),
    );
  };

  const tell = async (path, value) => {
    const answer = await fetch(`/g/selection/${path}${window.location.search}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ answer: state.answer, items: [...state.selected], value }),
    });
    if (answer.status === 409) {
      // The answer moved underneath the selection: nothing was written,
      // and the honest move is a whole redraw and a fresh selection.
      window.location.reload();
      return;
    }
    if (!answer.ok) {
      window.alert((await answer.json()).detail || "the selection could not be curated");
      return;
    }
    const told = await answer.json();
    const g = grid();
    if (told.after.answer === state.answer) {
      // The facts changed; the answer did not. Adopt the generation in
      // place and keep the selection mounted for the next operation.
      if (g) {
        g.dataset.currency = told.after.currency;
        g.dataset.answer = told.after.answer;
      }
      draw();
    } else {
      // The selected files left (or re-entered) this answer: the URL
      // owns what renders now.
      window.location.reload();
    }
  };

  document.addEventListener("change", (event) => {
    const pick = event.target.closest("[data-pick]");
    if (!pick) return;
    const shell = pick.closest("[data-selection-key]");
    if (!shell) return;
    if (pick.checked) state.selected.add(shell.dataset.selectionKey);
    else state.selected.delete(shell.dataset.selectionKey);
    if (state.selected.size === 1 && !albums.options.length) shelve();
    draw();
  });

  bar.addEventListener("click", (event) => {
    const favorite = event.target.closest("[data-bulk-favorite]");
    if (favorite) {
      tell("favorite", favorite.dataset.bulkFavorite === "1");
      return;
    }
    const stars = event.target.closest("[data-bulk-rate]");
    if (stars) {
      const n = +stars.dataset.bulkRate;
      tell("rating", n > 0 ? n : null);
      return;
    }
    const filed = event.target.closest("[data-bulk-file]");
    if (filed && albums.value) {
      tell(`collections/${albums.value}`, filed.dataset.bulkFile === "1");
      return;
    }
    if (event.target.closest("[data-curate-clear]")) clear();
  });

  document.body.addEventListener("htmx:afterSwap", sync);
  sync();
})();
