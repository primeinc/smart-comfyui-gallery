// The authored strip: favorite, rating and album membership for the
// media item on screen -- the same markup in the full page and the
// lightbox, because /i/{slug} has one state.
//
// Every write states the DESIRED FINAL FACT (favorite=true, rating=4,
// member=false), so a retry is harmless, and the strip redraws from the
// response's authoritative state, never from its own click.
//
// After a commit the library generation has moved (data_version bumps
// on every commit) while the mounted answer usually has not. The
// coherence check asks locate for the walked context's (currency,
// answer) pair: same answer -> adopt the new currency in place, so the
// next arrow does not 409 over an unchanged answer; different answer or
// no longer in it -> the mounted walk is really stale, redraw whole.
(() => {
  "use strict";

  const strip = (node) => node.closest("[data-authored]");

  const draw = (root, authored) => {
    root.querySelector("[data-fav]").setAttribute("aria-pressed", authored.favorite ? "true" : "false");
    const stars = root.querySelector("[data-stars]");
    stars.dataset.rating = authored.rating || 0;
    for (const star of stars.querySelectorAll("[data-rate]")) {
      const n = +star.dataset.rate;
      if (n > 0) star.setAttribute("aria-pressed", authored.rating && authored.rating >= n ? "true" : "false");
    }
    const albums = root.querySelector("[data-albums]");
    albums.replaceChildren(
      ...authored.collections.map((held) => {
        const link = document.createElement("a");
        link.href = `/t/${held.slug}`;
        link.textContent = held.name;
        return link;
      }),
    );
  };

  // The mounted result-set surfaces this item is being walked over:
  // the lightbox fragment and/or the gallery grid behind it.
  const mounted = () =>
    [document.querySelector("[data-lightbox]"), document.querySelector("[data-grid]")].filter(Boolean);

  const settle = async (root) => {
    const qs = root.dataset.qs;
    const surfaces = mounted();
    if (!surfaces.length) return;
    const asked = await fetch(`/g/locate/${root.dataset.slug}${qs ? `?${qs}` : ""}`);
    if (!asked.ok) {
      window.location.reload();
      return;
    }
    const told = await asked.json();
    const held = surfaces[0].dataset.answer || "";
    if (told.in_answer === false || (held && told.answer !== held)) {
      // The walked answer really changed -- the URL owns the state.
      window.location.reload();
      return;
    }
    for (const surface of surfaces) {
      surface.dataset.currency = told.currency;
      surface.dataset.answer = told.answer;
    }
  };

  const tell = async (root, path, value) => {
    const answer = await fetch(`/i/${root.dataset.slug}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value }),
    });
    if (!answer.ok) return null;
    const told = await answer.json();
    draw(root, told.authored);
    await settle(root);
    return told;
  };

  const choices = async (root) => {
    const box = root.querySelector("[data-album-choices]");
    if (!box.hidden) {
      box.hidden = true;
      return;
    }
    const asked = await fetch(`/i/${root.dataset.slug}/collection-choices`);
    if (!asked.ok) return;
    const told = await asked.json();
    box.replaceChildren(
      ...told.map((one) => {
        const row = document.createElement("label");
        const tick = document.createElement("input");
        tick.type = "checkbox";
        tick.checked = one.filed;
        tick.addEventListener("change", () => tell(root, `/collections/${one.slug}`, tick.checked));
        row.append(tick, ` ${one.name}`);
        return row;
      }),
    );
    if (!told.length) box.textContent = "no albums yet — make one on /albums";
    box.hidden = false;
  };

  document.addEventListener("click", (event) => {
    const root = strip(event.target);
    if (!root) return;
    const fav = event.target.closest("[data-fav]");
    if (fav) {
      tell(root, "/favorite", fav.getAttribute("aria-pressed") !== "true");
      return;
    }
    const star = event.target.closest("[data-rate]");
    if (star) {
      const n = +star.dataset.rate;
      tell(root, "/rating", n > 0 ? n : null);
      return;
    }
    if (event.target.closest("[data-album-picker]")) choices(root);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select") || event.ctrlKey || event.metaKey || event.altKey) return;
    const root = document.querySelector("[data-lightbox] [data-authored]") || document.querySelector("[data-authored]");
    if (!root) return;
    if (event.key === "f" || event.key === "F") {
      const fav = root.querySelector("[data-fav]");
      tell(root, "/favorite", fav.getAttribute("aria-pressed") !== "true");
    } else if (event.key >= "1" && event.key <= "5") {
      tell(root, "/rating", +event.key);
    } else if (event.key === "0") {
      tell(root, "/rating", null);
    } else if (event.key === "a" || event.key === "A") {
      choices(root);
    }
  });
})();
