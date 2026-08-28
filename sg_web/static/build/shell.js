"use strict";
(() => {
  // src/workspace.ts
  var VERSION = 1;
  var KEY = `sg.workspace.v${VERSION}`;
  function workspace() {
    try {
      const held = localStorage.getItem(KEY);
      if (!held) return {};
      const parsed = JSON.parse(held);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
      return parsed;
    } catch {
      return {};
    }
  }
  function remember(change) {
    try {
      localStorage.setItem(KEY, JSON.stringify({ ...workspace(), ...change }));
    } catch {
    }
  }

  // src/install.ts
  var standalone = () => navigator.standalone === true;
  var installed = () => window.matchMedia("not (display-mode: browser)").matches || standalone();
  var dismissed = () => workspace().installDismissed === true;
  var markDismissed = () => remember({ installDismissed: true });
  var isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent) || /macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1;
  function mountInstall() {
    const button2 = document.querySelector("[data-install]");
    const hint = document.querySelector("[data-install-ios]");
    if (installed() || dismissed()) return;
    let stashed = null;
    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      stashed = event;
      if (button2 instanceof HTMLElement) button2.hidden = false;
    });
    if (button2 instanceof HTMLElement) {
      button2.addEventListener("click", async () => {
        if (!stashed) return;
        stashed.prompt();
        const { outcome } = await stashed.userChoice;
        stashed = null;
        button2.hidden = true;
        if (outcome === "dismissed") markDismissed();
      });
    }
    window.addEventListener("appinstalled", () => {
      if (button2 instanceof HTMLElement) button2.hidden = true;
      if (hint instanceof HTMLElement) hint.hidden = true;
    });
    if (isIOS && !standalone() && hint instanceof HTMLElement) {
      hint.hidden = false;
      hint.querySelector("[data-dismiss]")?.addEventListener("click", () => {
        hint.hidden = true;
        markDismissed();
      });
    }
  }
  function mountServiceWorker() {
    if (!("serviceWorker" in navigator)) return;
    window.addEventListener("load", async () => {
      const reg = await navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" });
      reg.addEventListener("updatefound", () => {
        const next = reg.installing;
        if (!next) return;
        next.addEventListener("statechange", () => {
          if (next.state !== "installed" || !navigator.serviceWorker.controller) return;
          const notice = document.querySelector("[data-shell-notice]");
          if (!(notice instanceof HTMLElement)) return;
          notice.textContent = "a new version of the gallery is ready \u2014 ";
          const go = document.createElement("button");
          go.type = "button";
          go.className = "link";
          go.textContent = "reload";
          go.addEventListener("click", () => reg.waiting?.postMessage({ type: "SKIP_WAITING" }));
          notice.append(go);
        });
      });
    });
    const wasControlled = Boolean(navigator.serviceWorker.controller);
    let refreshing = false;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!wasControlled || refreshing) return;
      refreshing = true;
      location.reload();
    });
  }

  // src/dom.ts
  function requireElement(root, selector, type) {
    const found = root.querySelector(selector);
    if (!(found instanceof type)) {
      throw new Error(`expected ${selector} to be ${type.name}, found ${describe(found)}`);
    }
    return found;
  }
  function describe(found) {
    return found === null ? "nothing" : found.constructor.name;
  }

  // src/ask.ts
  var DISMISSED = "";
  var TAKEN = "ok";
  var button = (words, value, kind) => {
    const control = document.createElement("button");
    control.value = value;
    control.className = kind;
    control.textContent = words;
    return control;
  };
  async function ask(asked, build) {
    const box = document.createElement("dialog");
    box.className = "ask-box";
    box.innerHTML = `<form method="dialog" class="ask-form">
      <h2 class="ask-question"></h2>
      <p class="ask-detail" hidden></p>
      <div class="ask-body"></div>
      <div class="ask-feet"></div>
    </form>`;
    requireElement(box, ".ask-question", HTMLElement).textContent = asked.question;
    if (asked.detail !== void 0) {
      const line = requireElement(box, ".ask-detail", HTMLElement);
      line.textContent = asked.detail;
      line.hidden = false;
    }
    const read = build(requireElement(box, ".ask-body", HTMLElement), box);
    const feet = requireElement(box, ".ask-feet", HTMLElement);
    if (asked.submit !== null) {
      feet.append(button(asked.submit, TAKEN, asked.grave === true ? "ask-take is-grave" : "ask-take"));
    }
    if (asked.dismiss !== null) feet.append(button(asked.dismiss, DISMISSED, "ask-drop"));
    box.addEventListener("click", (event) => {
      const at = box.getBoundingClientRect();
      const inside = event.clientX >= at.left && event.clientX <= at.right && event.clientY >= at.top && event.clientY <= at.bottom;
      if (!inside && event.detail > 0) box.close(DISMISSED);
    });
    const answer = new Promise((settle) => {
      box.addEventListener(
        "close",
        () => {
          const taken = box.returnValue !== DISMISSED ? read() : null;
          box.remove();
          settle(taken);
        },
        { once: true }
      );
    });
    document.body.append(box);
    box.showModal();
    return answer;
  }
  async function panel(title, fill, dismiss = "close") {
    await ask({ question: title, submit: null, dismiss }, (body) => {
      fill(body);
      return () => void 0;
    });
  }

  // src/keys.ts
  var claimed = /* @__PURE__ */ new Map();
  var spelled = (key) => key.length === 1 ? key.toLowerCase() : key;
  function register(commands) {
    const wanted = /* @__PURE__ */ new Map();
    for (const command of commands) {
      const key = spelled(command.key);
      const mine = wanted.get(key);
      if (mine) throw new Error(`${command.by} and ${mine.by} both claim "${key}" in one registration`);
      const already = claimed.get(key);
      if (already) throw new Error(`${command.by} claims "${key}", which ${already.by} already answers to`);
      wanted.set(key, command);
    }
    for (const [key, command] of wanted) claimed.set(key, command);
    return () => {
      for (const command of commands) {
        const key = spelled(command.key);
        if (claimed.get(key) === command) claimed.delete(key);
      }
    };
  }
  function registered() {
    return [...claimed.entries()].map(([key, command]) => ({ key, by: command.by })).sort((a, b) => a.by.localeCompare(b.by));
  }
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof Element && target.closest("input, textarea, select, [contenteditable], dialog[open]")) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const command = claimed.get(spelled(event.key));
    if (!command) return;
    event.preventDefault();
    command.run();
  });

  // src/shortcuts.ts
  var SPELLED = {
    ArrowLeft: "\u2190",
    ArrowRight: "\u2192",
    ArrowUp: "\u2191",
    ArrowDown: "\u2193",
    " ": "Space",
    Escape: "Esc"
  };
  var spell = (key) => SPELLED[key] ?? (key.length === 1 ? key.toUpperCase() : key);
  function grouped() {
    const groups = /* @__PURE__ */ new Map();
    for (const { key, by } of registered()) {
      const cut = by.indexOf(":");
      const where = cut === -1 ? "everywhere" : by.slice(0, cut).trim();
      const does = cut === -1 ? by : by.slice(cut + 1).trim();
      const held = groups.get(where) ?? [];
      held.push({ key: spell(key), does });
      groups.set(where, held);
    }
    return groups;
  }
  function draw(body) {
    const groups = grouped();
    if (groups.size === 0) {
      const none = document.createElement("p");
      none.className = "muted";
      none.textContent = "nothing on this surface answers to a key.";
      body.append(none);
      return;
    }
    for (const [where, commands] of groups) {
      const section = document.createElement("section");
      section.className = "keys-group";
      const head = document.createElement("h3");
      head.textContent = where;
      section.append(head);
      const list = document.createElement("dl");
      list.className = "keys-list";
      for (const { key, does } of commands) {
        const term = document.createElement("dt");
        const cap = document.createElement("kbd");
        cap.textContent = key;
        term.append(cap);
        const said = document.createElement("dd");
        said.textContent = does;
        list.append(term, said);
      }
      section.append(list);
      body.append(section);
    }
  }
  function showShortcuts() {
    void panel("what the keyboard does", draw);
  }
  function mountShortcuts(root) {
    register([{ key: "?", by: "what the keyboard does", run: showShortcuts }]);
    for (const button2 of root.querySelectorAll("[data-shortcuts-open]")) {
      button2.addEventListener("click", showShortcuts);
    }
  }

  // src/entries/shell.ts
  mountInstall();
  mountServiceWorker();
  mountShortcuts(document);
})();
//# sourceMappingURL=shell.js.map
