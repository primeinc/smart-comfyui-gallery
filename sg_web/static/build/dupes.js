"use strict";
(() => {
  // ../node_modules/openapi-fetch/dist/index.mjs
  var PATH_PARAM_RE = /\{[^{}]+\}/g;
  var supportsRequestInitExt = () => {
    return typeof process === "object" && Number.parseInt(process?.versions?.node?.substring(0, 2)) >= 18 && process.versions.undici;
  };
  function randomID() {
    return Math.random().toString(36).slice(2, 11);
  }
  function createClient(clientOptions) {
    let {
      baseUrl = "",
      Request: CustomRequest = globalThis.Request,
      fetch: baseFetch = globalThis.fetch,
      querySerializer: globalQuerySerializer,
      bodySerializer: globalBodySerializer,
      pathSerializer: globalPathSerializer,
      headers: baseHeaders,
      requestInitExt = void 0,
      ...baseOptions
    } = { ...clientOptions };
    requestInitExt = supportsRequestInitExt() ? requestInitExt : void 0;
    baseUrl = removeTrailingSlash(baseUrl);
    const globalMiddlewares = [];
    async function coreFetch(schemaPath, fetchOptions) {
      const {
        baseUrl: localBaseUrl,
        fetch = baseFetch,
        Request = CustomRequest,
        headers,
        params = {},
        parseAs = "json",
        querySerializer: requestQuerySerializer,
        bodySerializer = globalBodySerializer ?? defaultBodySerializer,
        pathSerializer: requestPathSerializer,
        body,
        middleware: requestMiddlewares = [],
        ...init
      } = fetchOptions || {};
      let finalBaseUrl = baseUrl;
      if (localBaseUrl) {
        finalBaseUrl = removeTrailingSlash(localBaseUrl) ?? baseUrl;
      }
      let querySerializer = typeof globalQuerySerializer === "function" ? globalQuerySerializer : createQuerySerializer(globalQuerySerializer);
      if (requestQuerySerializer) {
        querySerializer = typeof requestQuerySerializer === "function" ? requestQuerySerializer : createQuerySerializer({
          ...typeof globalQuerySerializer === "object" ? globalQuerySerializer : {},
          ...requestQuerySerializer
        });
      }
      const pathSerializer = requestPathSerializer || globalPathSerializer || defaultPathSerializer;
      const serializedBody = body === void 0 ? void 0 : bodySerializer(
        body,
        // Note: we declare mergeHeaders() both here and below because it’s a bit of a chicken-or-egg situation:
        // bodySerializer() needs all headers so we aren’t dropping ones set by the user, however,
        // the result of this ALSO sets the lowest-priority content-type header. So we re-merge below,
        // setting the content-type at the very beginning to be overwritten.
        // Lastly, based on the way headers work, it’s not a simple “present-or-not” check becauase null intentionally un-sets headers.
        mergeHeaders(baseHeaders, headers, params.header)
      );
      const finalHeaders = mergeHeaders(
        // with no body, we should not to set Content-Type
        serializedBody === void 0 || // if serialized body is FormData; browser will correctly set Content-Type & boundary expression
        serializedBody instanceof FormData ? {} : {
          "Content-Type": "application/json"
        },
        baseHeaders,
        headers,
        params.header
      );
      const finalMiddlewares = [...globalMiddlewares, ...requestMiddlewares];
      const requestInit = {
        redirect: "follow",
        ...baseOptions,
        ...init,
        body: serializedBody,
        headers: finalHeaders
      };
      let id;
      let options;
      let request = new Request(
        createFinalURL(schemaPath, { baseUrl: finalBaseUrl, params, querySerializer, pathSerializer }),
        requestInit
      );
      let response;
      for (const key in init) {
        if (!(key in request)) {
          request[key] = init[key];
        }
      }
      if (finalMiddlewares.length) {
        id = randomID();
        options = Object.freeze({
          baseUrl: finalBaseUrl,
          fetch,
          parseAs,
          querySerializer,
          bodySerializer,
          pathSerializer
        });
        for (const m of finalMiddlewares) {
          if (m && typeof m === "object" && typeof m.onRequest === "function") {
            const result = await m.onRequest({
              request,
              schemaPath,
              params,
              options,
              id
            });
            if (result) {
              if (result instanceof Request) {
                request = result;
              } else if (result instanceof Response) {
                response = result;
                break;
              } else {
                throw new Error("onRequest: must return new Request() or Response() when modifying the request");
              }
            }
          }
        }
      }
      if (!response) {
        try {
          response = await fetch(request, requestInitExt);
        } catch (error2) {
          let errorAfterMiddleware = error2;
          if (finalMiddlewares.length) {
            for (let i = finalMiddlewares.length - 1; i >= 0; i--) {
              const m = finalMiddlewares[i];
              if (m && typeof m === "object" && typeof m.onError === "function") {
                const result = await m.onError({
                  request,
                  error: errorAfterMiddleware,
                  schemaPath,
                  params,
                  options,
                  id
                });
                if (result) {
                  if (result instanceof Response) {
                    errorAfterMiddleware = void 0;
                    response = result;
                    break;
                  }
                  if (result instanceof Error) {
                    errorAfterMiddleware = result;
                    continue;
                  }
                  throw new Error("onError: must return new Response() or instance of Error");
                }
              }
            }
          }
          if (errorAfterMiddleware) {
            throw errorAfterMiddleware;
          }
        }
        if (finalMiddlewares.length) {
          for (let i = finalMiddlewares.length - 1; i >= 0; i--) {
            const m = finalMiddlewares[i];
            if (m && typeof m === "object" && typeof m.onResponse === "function") {
              const result = await m.onResponse({
                request,
                response,
                schemaPath,
                params,
                options,
                id
              });
              if (result) {
                if (!(result instanceof Response)) {
                  throw new Error("onResponse: must return new Response() when modifying the response");
                }
                response = result;
              }
            }
          }
        }
      }
      const contentLength = response.headers.get("Content-Length");
      if (response.status === 204 || request.method === "HEAD" || contentLength === "0" && !response.headers.get("Transfer-Encoding")?.includes("chunked")) {
        return response.ok ? { data: void 0, response } : { error: void 0, response };
      }
      if (response.ok) {
        const getResponseData = async () => {
          if (parseAs === "stream") {
            return response.body;
          }
          if (parseAs === "json" && !contentLength) {
            const raw = await response.text();
            return raw ? JSON.parse(raw) : void 0;
          }
          return await response[parseAs]();
        };
        return { data: await getResponseData(), response };
      }
      let error = await response.text();
      try {
        error = JSON.parse(error);
      } catch {
      }
      return { error, response };
    }
    return {
      request(method, url, init) {
        return coreFetch(url, { ...init, method: method.toUpperCase() });
      },
      /** Call a GET endpoint */
      GET(url, init) {
        return coreFetch(url, { ...init, method: "GET" });
      },
      /** Call a PUT endpoint */
      PUT(url, init) {
        return coreFetch(url, { ...init, method: "PUT" });
      },
      /** Call a POST endpoint */
      POST(url, init) {
        return coreFetch(url, { ...init, method: "POST" });
      },
      /** Call a DELETE endpoint */
      DELETE(url, init) {
        return coreFetch(url, { ...init, method: "DELETE" });
      },
      /** Call a OPTIONS endpoint */
      OPTIONS(url, init) {
        return coreFetch(url, { ...init, method: "OPTIONS" });
      },
      /** Call a HEAD endpoint */
      HEAD(url, init) {
        return coreFetch(url, { ...init, method: "HEAD" });
      },
      /** Call a PATCH endpoint */
      PATCH(url, init) {
        return coreFetch(url, { ...init, method: "PATCH" });
      },
      /** Call a TRACE endpoint */
      TRACE(url, init) {
        return coreFetch(url, { ...init, method: "TRACE" });
      },
      /** Register middleware */
      use(...middleware) {
        for (const m of middleware) {
          if (!m) {
            continue;
          }
          if (typeof m !== "object" || !("onRequest" in m || "onResponse" in m || "onError" in m)) {
            throw new Error("Middleware must be an object with one of `onRequest()`, `onResponse() or `onError()`");
          }
          globalMiddlewares.push(m);
        }
      },
      /** Unregister middleware */
      eject(...middleware) {
        for (const m of middleware) {
          const i = globalMiddlewares.indexOf(m);
          if (i !== -1) {
            globalMiddlewares.splice(i, 1);
          }
        }
      }
    };
  }
  function serializePrimitiveParam(name, value, options) {
    if (value === void 0 || value === null) {
      return "";
    }
    if (typeof value === "object") {
      throw new Error(
        "Deeply-nested arrays/objects aren\u2019t supported. Provide your own `querySerializer()` to handle these."
      );
    }
    return `${name}=${options?.allowReserved === true ? value : encodeURIComponent(value)}`;
  }
  function serializeObjectParam(name, value, options) {
    if (!value || typeof value !== "object") {
      return "";
    }
    const values = [];
    const joiner = {
      simple: ",",
      label: ".",
      matrix: ";"
    }[options.style] || "&";
    if (options.style !== "deepObject" && options.explode === false) {
      for (const k in value) {
        values.push(k, options.allowReserved === true ? value[k] : encodeURIComponent(value[k]));
      }
      const final2 = values.join(",");
      switch (options.style) {
        case "form": {
          return `${name}=${final2}`;
        }
        case "label": {
          return `.${final2}`;
        }
        case "matrix": {
          return `;${name}=${final2}`;
        }
        default: {
          return final2;
        }
      }
    }
    for (const k in value) {
      const finalName = options.style === "deepObject" ? `${name}[${k}]` : k;
      values.push(serializePrimitiveParam(finalName, value[k], options));
    }
    const final = values.join(joiner);
    return options.style === "label" || options.style === "matrix" ? `${joiner}${final}` : final;
  }
  function serializeArrayParam(name, value, options) {
    if (!Array.isArray(value)) {
      return "";
    }
    if (options.explode === false) {
      const joiner2 = { form: ",", spaceDelimited: "%20", pipeDelimited: "|" }[options.style] || ",";
      const final = (options.allowReserved === true ? value : value.map((v) => encodeURIComponent(v))).join(joiner2);
      switch (options.style) {
        case "simple": {
          return final;
        }
        case "label": {
          return `.${final}`;
        }
        case "matrix": {
          return `;${name}=${final}`;
        }
        // case "spaceDelimited":
        // case "pipeDelimited":
        default: {
          return `${name}=${final}`;
        }
      }
    }
    const joiner = { simple: ",", label: ".", matrix: ";" }[options.style] || "&";
    const values = [];
    for (const v of value) {
      if (options.style === "simple" || options.style === "label") {
        values.push(options.allowReserved === true ? v : encodeURIComponent(v));
      } else {
        values.push(serializePrimitiveParam(name, v, options));
      }
    }
    return options.style === "label" || options.style === "matrix" ? `${joiner}${values.join(joiner)}` : values.join(joiner);
  }
  function createQuerySerializer(options) {
    return function querySerializer(queryParams) {
      const search = [];
      if (queryParams && typeof queryParams === "object") {
        for (const name in queryParams) {
          const value = queryParams[name];
          if (value === void 0 || value === null) {
            continue;
          }
          if (Array.isArray(value)) {
            if (value.length === 0) {
              continue;
            }
            search.push(
              serializeArrayParam(name, value, {
                style: "form",
                explode: true,
                ...options?.array,
                allowReserved: options?.allowReserved || false
              })
            );
            continue;
          }
          if (typeof value === "object") {
            search.push(
              serializeObjectParam(name, value, {
                style: "deepObject",
                explode: true,
                ...options?.object,
                allowReserved: options?.allowReserved || false
              })
            );
            continue;
          }
          search.push(serializePrimitiveParam(name, value, options));
        }
      }
      return search.join("&");
    };
  }
  function defaultPathSerializer(pathname, pathParams) {
    let nextURL = pathname;
    for (const match of pathname.match(PATH_PARAM_RE) ?? []) {
      let name = match.substring(1, match.length - 1);
      let explode = false;
      let style = "simple";
      if (name.endsWith("*")) {
        explode = true;
        name = name.substring(0, name.length - 1);
      }
      if (name.startsWith(".")) {
        style = "label";
        name = name.substring(1);
      } else if (name.startsWith(";")) {
        style = "matrix";
        name = name.substring(1);
      }
      if (!pathParams || pathParams[name] === void 0 || pathParams[name] === null) {
        continue;
      }
      const value = pathParams[name];
      if (Array.isArray(value)) {
        nextURL = nextURL.replace(match, serializeArrayParam(name, value, { style, explode }));
        continue;
      }
      if (typeof value === "object") {
        nextURL = nextURL.replace(match, serializeObjectParam(name, value, { style, explode }));
        continue;
      }
      if (style === "matrix") {
        nextURL = nextURL.replace(match, `;${serializePrimitiveParam(name, value)}`);
        continue;
      }
      nextURL = nextURL.replace(match, style === "label" ? `.${encodeURIComponent(value)}` : encodeURIComponent(value));
    }
    return nextURL;
  }
  function defaultBodySerializer(body, headers) {
    if (body instanceof FormData) {
      return body;
    }
    if (headers) {
      const contentType = headers.get instanceof Function ? headers.get("Content-Type") ?? headers.get("content-type") : headers["Content-Type"] ?? headers["content-type"];
      if (contentType === "application/x-www-form-urlencoded") {
        return new URLSearchParams(body).toString();
      }
    }
    return JSON.stringify(body);
  }
  function createFinalURL(pathname, options) {
    let finalURL = `${options.baseUrl}${pathname}`;
    if (options.params?.path) {
      finalURL = options.pathSerializer(finalURL, options.params.path);
    }
    let search = options.querySerializer(options.params.query ?? {});
    if (search.startsWith("?")) {
      search = search.substring(1);
    }
    if (search) {
      finalURL += `?${search}`;
    }
    return finalURL;
  }
  function mergeHeaders(...allHeaders) {
    const finalHeaders = new Headers();
    for (const h of allHeaders) {
      if (!h || typeof h !== "object") {
        continue;
      }
      const iterator = h instanceof Headers ? h.entries() : Object.entries(h);
      for (const [k, v] of iterator) {
        if (v === null) {
          finalHeaders.delete(k);
        } else if (Array.isArray(v)) {
          for (const v2 of v) {
            finalHeaders.append(k, v2);
          }
        } else if (v !== void 0) {
          finalHeaders.set(k, v);
        }
      }
    }
    return finalHeaders;
  }
  function removeTrailingSlash(url) {
    if (url.endsWith("/")) {
      return url.substring(0, url.length - 1);
    }
    return url;
  }

  // src/api.ts
  var api = createClient();
  function refusal(error, fallback) {
    if (typeof error === "object" && error !== null && "detail" in error && typeof error.detail === "string") {
      return error.detail;
    }
    return fallback;
  }

  // src/dom.ts
  function requireElement(root, selector, type) {
    const found = root.querySelector(selector);
    if (!(found instanceof type)) {
      throw new Error(`expected ${selector} to be ${type.name}, found ${describe(found)}`);
    }
    return found;
  }
  function findElement(root, selector, type) {
    const found = root.querySelector(selector);
    return found instanceof type ? found : null;
  }
  function everyElement(root, selector, type) {
    return [...root.querySelectorAll(selector)].filter((node) => node instanceof type);
  }
  function closestFrom(target, selector, type) {
    if (!(target instanceof Element)) return null;
    const found = target.closest(selector);
    return found instanceof type ? found : null;
  }
  function requireData(node, key) {
    const held = node.dataset[key];
    if (held === void 0) {
      throw new Error(`expected a data-${key} on ${node.tagName.toLowerCase()}`);
    }
    return held;
  }
  function describe(found) {
    return found === null ? "nothing" : found.constructor.name;
  }

  // src/ask.ts
  var DISMISSED = "";
  var TAKEN = "ok";
  var framed = (question, submit, dismiss, said) => ({
    question,
    submit: said.submit !== void 0 ? said.submit : submit,
    dismiss: said.dismiss !== void 0 ? said.dismiss : dismiss,
    ...said.detail !== void 0 ? { detail: said.detail } : {},
    ...said.grave !== void 0 ? { grave: said.grave } : {}
  });
  var button = (words, value, kind) => {
    const control = document.createElement("button");
    control.value = value;
    control.className = kind;
    control.textContent = words;
    return control;
  };
  async function ask(asked, build2) {
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
    const read2 = build2(requireElement(box, ".ask-body", HTMLElement), box);
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
          const taken = box.returnValue !== DISMISSED ? read2() : null;
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
  async function say(message, framing = {}) {
    await ask(framed(message, "ok", null, framing), () => () => void 0);
  }

  // src/dupes.ts
  (() => {
    const groups = document.querySelector("[data-dupe-groups]");
    if (!(groups instanceof HTMLElement)) return;
    groups.addEventListener("click", async (event) => {
      const button2 = closestFrom(event.target, "[data-not-a-duplicate]", HTMLButtonElement);
      if (!button2) return;
      const one = requireData(button2, "notADuplicate");
      const other = requireData(button2, "against");
      button2.disabled = true;
      const held = await api.POST("/dupes/{slug}/not-a-duplicate", {
        params: { path: { slug: one } },
        body: { other }
      });
      if (held.error) {
        button2.disabled = false;
        await say(refusal(held.error, "that was not recorded"));
        return;
      }
      const member = button2.closest("[data-dupe-member]");
      if (member instanceof HTMLElement) member.remove();
    });
  })();

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
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof Element && target.closest("input, textarea, select, [contenteditable], dialog[open]")) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const command = claimed.get(spelled(event.key));
    if (!command) return;
    event.preventDefault();
    command.run();
  });

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

  // src/zoom.ts
  var TRAY_MAX_SCALE = 16;

  // src/compare.ts
  var MOST = 8;
  function kept() {
    const held = workspace().compare;
    return Array.isArray(held) ? held.filter((one) => one && typeof one.slug === "string") : [];
  }
  function keep(held) {
    remember({ compare: held.slice(-MOST) });
  }
  function current(root) {
    const lightbox = findElement(root, "[data-lightbox][data-slug]", HTMLElement);
    if (lightbox) {
      const named = findElement(lightbox, "[data-viewer][data-slug]", HTMLElement) ?? lightbox;
      const slug = named.dataset.slug ?? lightbox.dataset.slug;
      if (slug) return { slug, name: named.dataset.name ?? slug, kind: named.dataset.kind ?? "" };
    }
    const viewer = findElement(root, "[data-viewer][data-slug]", HTMLElement);
    if (viewer?.dataset.slug) {
      return {
        slug: viewer.dataset.slug,
        name: viewer.dataset.name ?? viewer.dataset.slug,
        kind: viewer.dataset.kind ?? ""
      };
    }
    const under = root.querySelector("a.cell[data-slug]:hover");
    const focused = document.activeElement?.closest?.("a.cell[data-slug]") ?? null;
    return read(under ?? focused);
  }
  function read(cell) {
    if (!cell?.dataset.slug) return null;
    const shown = cell.querySelector("img");
    return {
      slug: cell.dataset.slug,
      name: shown?.getAttribute("alt") || cell.dataset.slug,
      kind: cell.dataset.kind ?? "",
      thumb: shown?.getAttribute("src") ?? ""
    };
  }
  function picked(root) {
    const found = [];
    for (const box of root.querySelectorAll("[data-pick]")) {
      if (!(box instanceof HTMLInputElement) || !box.checked) continue;
      const one = read(box.closest(".cell-shell")?.querySelector("a.cell[data-slug]") ?? null);
      if (one) found.push(one);
    }
    return found;
  }
  function playable(one) {
    if (one.kind === "video" || one.kind === "animated_image") {
      const clip = document.createElement("video");
      clip.src = `/media/${one.slug}`;
      clip.poster = `/preview/${one.slug}`;
      clip.controls = true;
      clip.loop = true;
      clip.playsInline = true;
      clip.setAttribute("aria-label", one.name);
      return clip;
    }
    if (one.kind === "audio") {
      const sound = document.createElement("audio");
      sound.src = `/media/${one.slug}`;
      sound.controls = true;
      sound.setAttribute("aria-label", one.name);
      return sound;
    }
    const shown = document.createElement("img");
    shown.src = `/preview/${one.slug}`;
    shown.alt = one.name;
    return shown;
  }
  function letter(at) {
    return String.fromCharCode(65 + at % 26);
  }
  function showComparison(held) {
    const old = document.querySelector("[data-compare-view]");
    if (old) old.remove();
    if (held.length < 2) return;
    const sheet = document.createElement("div");
    sheet.className = "compare-view";
    sheet.dataset.compareView = "";
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-label", "comparing");
    const bar = document.createElement("header");
    bar.className = "compare-view-bar";
    const said = document.createElement("span");
    said.className = "compare-view-said";
    const zoom = document.createElement("button");
    zoom.type = "button";
    zoom.className = "compare-zoom";
    zoom.dataset.compareZoom = "";
    zoom.title = "back to fit";
    zoom.textContent = "fit";
    zoom.addEventListener("click", () => zoomTo(1, 0.5, 0.5));
    const close = document.createElement("button");
    close.type = "button";
    close.className = "compare-view-close";
    close.dataset.compareViewClose = "";
    close.setAttribute("aria-label", "stop comparing");
    close.textContent = "\xD7";
    const modes = document.createElement("div");
    modes.className = "compare-modes";
    modes.setAttribute("role", "group");
    modes.setAttribute("aria-label", "how to compare");
    bar.append(said, modes, zoom, close);
    const strip = document.createElement("div");
    strip.className = "compare-view-strip";
    for (const [at2, one] of held.entries()) {
      const column = document.createElement("figure");
      column.className = "compare-column";
      column.dataset.compareColumn = one.slug;
      column.dataset.at = String(at2);
      column.dataset.letter = letter(at2);
      const frame = document.createElement("div");
      frame.className = "compare-frame";
      const shown = playable(one);
      frame.append(shown);
      const label2 = document.createElement("figcaption");
      const named = document.createElement("b");
      named.className = "compare-letter";
      named.textContent = letter(at2);
      const link = document.createElement("a");
      link.href = `/i/${one.slug}`;
      link.textContent = one.name;
      label2.append(named, link);
      column.append(frame, label2);
      strip.append(column);
    }
    sheet.append(bar, strip);
    document.body.append(sheet);
    const glass = { scale: 1, x: 0.5, y: 0.5 };
    const magnify = () => {
      for (const column of everyElement(strip, "[data-compare-column]", HTMLElement)) {
        const shown = column.querySelector(".compare-frame > *");
        if (!shown) continue;
        shown.style.transformOrigin = `${glass.x * 100}% ${glass.y * 100}%`;
        shown.style.transform = glass.scale === 1 ? "" : `scale(${glass.scale})`;
      }
      strip.dataset.zoomed = String(glass.scale !== 1);
      zoom.textContent = glass.scale === 1 ? "fit" : `${Math.round(glass.scale * 100)}%`;
      zoom.setAttribute("aria-label", glass.scale === 1 ? "fit" : `zoomed to ${Math.round(glass.scale * 100)}%`);
    };
    const clamp = (n) => Math.min(1, Math.max(0, n));
    const zoomTo = (scale, x, y) => {
      glass.scale = Math.min(TRAY_MAX_SCALE, Math.max(1, scale));
      if (glass.scale === 1) {
        glass.x = 0.5;
        glass.y = 0.5;
      } else {
        glass.x = clamp(x);
        glass.y = clamp(y);
      }
      magnify();
    };
    const fractionIn = (frame, event) => {
      const box = frame.getBoundingClientRect();
      return { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
    };
    strip.addEventListener(
      "wheel",
      (event) => {
        const frame = closestFrom(event.target, ".compare-frame", HTMLElement);
        if (!frame) return;
        event.preventDefault();
        const where = fractionIn(frame, event);
        zoomTo(glass.scale * (event.deltaY < 0 ? 1.15 : 1 / 1.15), where.x, where.y);
      },
      { passive: false }
    );
    let dragging = null;
    strip.addEventListener("pointerdown", (event) => {
      if (glass.scale === 1) return;
      const frame = closestFrom(event.target, ".compare-frame", HTMLElement);
      if (!frame) return;
      dragging = { x: event.clientX, y: event.clientY, frame };
      frame.setPointerCapture(event.pointerId);
    });
    strip.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const box = dragging.frame.getBoundingClientRect();
      glass.x = clamp(glass.x - (event.clientX - dragging.x) / box.width / glass.scale);
      glass.y = clamp(glass.y - (event.clientY - dragging.y) / box.height / glass.scale);
      dragging = { ...dragging, x: event.clientX, y: event.clientY };
      magnify();
    });
    const letGo = () => {
      dragging = null;
    };
    strip.addEventListener("pointerup", letGo);
    strip.addEventListener("pointercancel", letGo);
    strip.addEventListener("dblclick", () => zoomTo(1, 0.5, 0.5));
    let mode = workspace().compareMode === "flip" ? "flip" : "side";
    let at = 0;
    const columns = () => everyElement(strip, "[data-compare-column]", HTMLElement);
    const paint = () => {
      sheet.dataset.mode = mode;
      const all = columns();
      at = (at % all.length + all.length) % all.length;
      for (const [index, column] of all.entries()) {
        column.hidden = mode === "flip" && index !== at;
        column.dataset.showing = String(mode === "side" || index === at);
      }
      const one = held[at];
      said.textContent = mode === "side" ? `${held.length} side by side` : `${letter(at)} of ${held.length} \xB7 ${one ? one.name : ""}`;
      for (const button2 of everyElement(modes, "[data-compare-mode]", HTMLElement)) {
        button2.setAttribute("aria-pressed", String(button2.dataset.compareMode === mode));
      }
    };
    for (const [name, words, why] of [
      ["side", "side by side", "every one at once: how do these differ"],
      ["flip", "flip", "one at a time in the same place: did this change"]
    ]) {
      const button2 = document.createElement("button");
      button2.type = "button";
      button2.className = "compare-mode";
      button2.dataset.compareMode = name;
      button2.title = why;
      button2.textContent = words;
      button2.addEventListener("click", () => {
        mode = name;
        remember({ compareMode: name });
        paint();
      });
      modes.append(button2);
    }
    const step = (by) => {
      at += by;
      if (mode !== "flip") {
        mode = "flip";
        remember({ compareMode: "flip" });
      }
      paint();
    };
    paint();
    const dismiss = () => sheet.remove();
    close.addEventListener("click", dismiss);
    sheet.addEventListener("click", (event) => {
      if (event.target === sheet) dismiss();
    });
    sheet.tabIndex = -1;
    sheet.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        dismiss();
        return;
      }
      if (event.key === " " || event.key === "f" || event.key === "F") {
        event.preventDefault();
        step(1);
        return;
      }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        event.preventDefault();
        step(1);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        event.preventDefault();
        step(-1);
      }
    });
    sheet.focus();
  }
  function drawTray(tray) {
    const held = kept();
    const open = workspace().tray !== "closed";
    tray.hidden = held.length === 0;
    tray.dataset.tray = open ? "open" : "closed";
    const count = findElement(tray, "[data-compare-count]", HTMLElement);
    if (count) count.textContent = String(held.length);
    const compare = findElement(tray, "[data-compare-open]", HTMLButtonElement);
    if (compare) compare.disabled = held.length < 2;
    const list = findElement(tray, "[data-compare-items]", HTMLElement);
    if (!list) return;
    list.replaceChildren();
    for (const [at, one] of held.entries()) {
      const item = document.createElement("li");
      item.className = "tray-item";
      item.draggable = true;
      item.dataset.compareSlug = one.slug;
      item.dataset.at = String(at);
      const shown = document.createElement("img");
      shown.src = one.thumb ?? `/thumb/${one.slug}`;
      shown.alt = one.name;
      shown.title = one.name;
      const drop = document.createElement("button");
      drop.type = "button";
      drop.className = "tray-drop";
      drop.dataset.compareRemove = one.slug;
      drop.setAttribute("aria-label", `stop keeping ${one.name}`);
      drop.textContent = "\xD7";
      item.append(shown, drop);
      list.append(item);
    }
    for (const drop of everyElement(list, "[data-compare-remove]", HTMLElement)) {
      drop.addEventListener("click", () => {
        keep(kept().filter((one) => one.slug !== drop.dataset.compareRemove));
        drawTray(tray);
      });
    }
    let from = null;
    for (const item of everyElement(list, "[data-compare-slug]", HTMLElement)) {
      item.addEventListener("dragstart", (event) => {
        from = Number(item.dataset.at);
        event.dataTransfer?.setData("text/plain", item.dataset.compareSlug ?? "");
        item.dataset.dragging = "true";
      });
      item.addEventListener("dragend", () => {
        delete item.dataset.dragging;
      });
      item.addEventListener("dragover", (event) => event.preventDefault());
      item.addEventListener("drop", (event) => {
        event.preventDefault();
        const to = Number(item.dataset.at);
        if (from === null || from === to) return;
        const order = kept();
        const [moved] = order.splice(from, 1);
        if (moved) order.splice(to, 0, moved);
        keep(order);
        from = null;
        drawTray(tray);
      });
    }
  }
  function build() {
    const tray = document.createElement("aside");
    tray.className = "tray";
    tray.dataset.compareTray = "";
    tray.hidden = true;
    tray.setAttribute("aria-label", "kept to compare");
    tray.innerHTML = [
      '<header class="tray-bar">',
      '<button type="button" class="tray-tab" data-compare-collapse aria-label="show or hide what is kept">',
      "kept <b data-compare-count>0</b>",
      "</button>",
      '<button type="button" class="tray-act" data-compare-open>compare</button>',
      '<button type="button" class="tray-act" data-compare-clear>clear</button>',
      "</header>",
      '<ol class="tray-items" data-compare-items></ol>'
    ].join("");
    return tray;
  }
  function mountCompare(root) {
    if (document.querySelector("[data-compare-tray]")) return;
    const tray = build();
    document.body.append(tray);
    const collapse = findElement(tray, "[data-compare-collapse]", HTMLElement);
    if (collapse) {
      collapse.addEventListener("click", () => {
        remember({ tray: workspace().tray === "closed" ? "open" : "closed" });
        drawTray(tray);
      });
    }
    const open = findElement(tray, "[data-compare-open]", HTMLElement);
    if (open) open.addEventListener("click", () => showComparison(kept()));
    const clear = findElement(tray, "[data-compare-clear]", HTMLElement);
    if (clear) {
      clear.addEventListener("click", () => {
        keep([]);
        drawTray(tray);
      });
    }
    const add = () => {
      const one = current(root);
      if (!one) return;
      const held = kept();
      keep(held.some((each) => each.slug === one.slug) ? held.filter((each) => each.slug !== one.slug) : [...held, one]);
      remember({ tray: "open" });
      drawTray(tray);
    };
    register([{ key: "c", by: "compare: keep this", run: add }]);
    for (const button2 of root.querySelectorAll("[data-compare-selection]")) {
      button2.addEventListener("click", () => {
        const chosen = picked(root);
        if (chosen.length === 0) return;
        const held = kept();
        const fresh = chosen.filter((one) => !held.some((each) => each.slug === one.slug));
        keep([...held, ...fresh]);
        remember({ tray: "open" });
        drawTray(tray);
      });
    }
    drawTray(tray);
  }

  // src/compare-mount.ts
  mountCompare(document.body);

  // src/pictures.ts
  function label(kind) {
    const said = document.createElement("span");
    said.className = "cell-kind";
    said.dataset.cellKind = kind ?? "";
    said.dataset.brokenPicture = "";
    said.setAttribute("aria-hidden", "true");
    said.textContent = kind === "audio" ? "audio" : "doc";
    return said;
  }
  function degrade(broken) {
    const src = broken.getAttribute("src") ?? "";
    if (!src.startsWith("/thumbs/") && !src.startsWith("/thumb/") && !src.startsWith("/preview/")) return;
    if (!broken.isConnected) return;
    const holder = broken.closest("[data-kind]");
    const kind = holder instanceof HTMLElement ? holder.dataset.kind : void 0;
    broken.replaceWith(label(kind));
  }
  function mountPictures() {
    document.addEventListener(
      "error",
      (event) => {
        const broken = event.target;
        if (!(broken instanceof HTMLImageElement)) return;
        degrade(broken);
      },
      // The capture phase, because `error` on an <img> does not bubble.
      true
    );
    for (const picture of document.querySelectorAll("img")) {
      if (picture.complete && picture.naturalWidth === 0) degrade(picture);
    }
  }

  // src/pictures-mount.ts
  mountPictures();
})();
//# sourceMappingURL=dupes.js.map
