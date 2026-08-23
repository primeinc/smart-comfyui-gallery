// Collection lifecycle, from the browser: the same one write adapter
// machines use, never a second opinion about the rules.
//
// Every control states the DESIRED FINAL STATE -- name is X, parent is
// Y, archived is true -- with the definition revision the page rendered
// at (data-rev). A stale revision answers 409, and this module's whole
// reaction is to reload: the server's CollectionView is authoritative
// and the browser never invents the resulting state. An empty input
// means CLEAR (sent as null); a fact the form does not carry is simply
// absent and stays unchanged.
(() => {
  const root = document.querySelector("[data-collection]");

  const told = async (method, path, body) => {
    const answer = await fetch(path, {
      method,
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!answer.ok) {
      const why = await answer.json().catch(() => ({}));
      window.alert(why.detail || "the collection did not accept that");
      if (answer.status === 409) window.location.reload();
      return null;
    }
    return answer.json();
  };

  const creating = document.querySelector("[data-new-collection]");
  if (creating) {
    creating.addEventListener("submit", async (event) => {
      event.preventDefault();
      const asked = new FormData(creating);
      const made = await told("POST", "/albums", {
        name: asked.get("name"),
        kind: asked.get("kind") || "album",
      });
      if (made) window.location.assign(`/t/${made.slug}`);
    });
  }

  if (!root) return;
  const slug = root.dataset.collection;
  const rev = +root.dataset.rev;

  const landed = (body) => {
    if (body) window.location.assign(`/t/${body.slug}`);
  };

  const editing = root.querySelector("[data-edit-definition]");
  if (editing) {
    editing.addEventListener("submit", async (event) => {
      event.preventDefault();
      const asked = new FormData(editing);
      landed(
        await told("PATCH", `/t/${slug}`, {
          expected_rev: rev,
          name: asked.get("name"),
          description: asked.get("description") || null,
          color: asked.get("color") || null,
          parent: asked.get("parent") || null,
        }),
      );
    });
  }

  const wire = (selector, ask) => {
    const control = root.querySelector(selector);
    if (control) {
      control.addEventListener("click", async () => landed(await ask(control)));
    }
  };

  wire("[data-archive]", () => told("PATCH", `/t/${slug}`, { expected_rev: rev, archived: true }));
  wire("[data-restore]", () => told("PATCH", `/t/${slug}`, { expected_rev: rev, archived: false }));
  wire("[data-convert]", (control) =>
    told("POST", `/t/${slug}/convert`, { expected_rev: rev, kind: control.dataset.convert }),
  );
  wire("[data-discard-rule]", () => {
    if (!window.confirm("discard this collection's rule and keep it as a plain album?")) return null;
    return told("POST", `/t/${slug}/convert`, { expected_rev: rev, kind: "album", discard_rule: true });
  });
})();
