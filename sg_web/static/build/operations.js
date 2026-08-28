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
        fetch: fetch2 = baseFetch,
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
          fetch: fetch2,
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
          response = await fetch2(request, requestInitExt);
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

  // src/frames.ts
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  var num = (value) => typeof value === "number" && Number.isFinite(value);
  var str = (value) => typeof value === "string";
  var numOrNull = (value) => value === null || num(value);
  var strOrNull = (value) => value === null || str(value);
  var dataOrNull = (value) => value === null || isRecord(value);
  function reported(held) {
    return num(held.job_id) && num(held.at) && str(held.type) && numOrNull(held.item_id) && strOrNull(held.phase) && str(held.severity) && strOrNull(held.message) && dataOrNull(held.data) && str(held.text) && strOrNull(held.condition);
  }
  function isEvent(value) {
    return isRecord(value) && reported(value) && num(value.id);
  }
  function isEventFrame(value) {
    return isRecord(value) && value.frame === "event" && isEvent(value);
  }
  function isPendingFrame(value) {
    return isRecord(value) && value.frame === "pending" && reported(value);
  }
  function isBacklogFrame(value) {
    return isRecord(value) && value.frame === "backlog" && num(value.after) && num(value.last_id) && Array.isArray(value.events) && value.events.every(isEvent);
  }
  function decodeFrame(payload) {
    if (typeof payload !== "string") return null;
    let held;
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

  // src/operations.ts
  (() => {
    const root = requireElement(document, "[data-console]", HTMLElement);
    const ROW_H = 24;
    const OVERSCAN = 12;
    const TAPE_COLD = 500;
    const TAPE_PAGE = 2e3;
    const RENDER_DEBOUNCE_MS = 400;
    const held = [];
    const ids = /* @__PURE__ */ new Set();
    const head = Number(requireData(root, "lastEventId"));
    let lastId = head;
    let firstId = Number.POSITIVE_INFINITY;
    const pendingByJob = /* @__PURE__ */ new Map();
    let paused = false;
    let heldWhilePaused = 0;
    let selectedJob = null;
    let selectedEvent = null;
    let socket = null;
    let retry = 0;
    let resuming = 0;
    let lastFrameAt = null;
    const filter = { type: "", severity: "", job: "" };
    let view = [];
    const transport = requireElement(root, "[data-health-transport]", HTMLElement);
    const transportState = requireElement(root, "[data-transport-state]", HTMLElement);
    const transportLast = requireElement(root, "[data-transport-last]", HTMLElement);
    const transportAge = requireElement(root, "[data-transport-age]", HTMLElement);
    const matrixRows = requireElement(root, "[data-matrix-rows]", HTMLOListElement);
    const inspectorBody = requireElement(root, "[data-inspector-body]", HTMLElement);
    const inspectorHint = requireElement(root, "[data-inspector-hint]", HTMLElement);
    const scroller = requireElement(root, "[data-tape-scroll]", HTMLElement);
    const spacer = requireElement(root, "[data-tape-spacer]", HTMLElement);
    const rows = requireElement(root, "[data-tape-rows]", HTMLOListElement);
    const rawBody = requireElement(root, "[data-tape-raw-body]", HTMLPreElement);
    const countEl = requireElement(root, "[data-tape-count]", HTMLElement);
    const heldEl = requireElement(root, "[data-tape-held]", HTMLElement);
    const pauseBtn = requireElement(root, "[data-tape-pause]", HTMLButtonElement);
    const follow = requireElement(root, "[data-tape-autoscroll]", HTMLInputElement);
    const jobFilter = requireElement(root, "[data-tape-filter-job]", HTMLInputElement);
    const pad = (n, w = 2) => String(n).padStart(w, "0");
    function clock(epoch) {
      const d = new Date(epoch * 1e3);
      return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
    }
    function seconds(v) {
      if (v == null) return "\u2014";
      if (v < 60) return `${v.toFixed(1)}s`;
      if (v < 3600) return `${Math.floor(v / 60)}m ${pad(Math.floor(v % 60))}s`;
      return `${Math.floor(v / 3600)}h ${pad(Math.floor(v % 3600 / 60))}m`;
    }
    function el(tag, attrs, text) {
      const node = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs ?? {})) {
        if (v === false || v == null) continue;
        node.setAttribute(k, v === true ? "" : String(v));
      }
      if (text != null) node.textContent = text;
      return node;
    }
    function ingest(event) {
      if (ids.has(event.id)) return false;
      ids.add(event.id);
      const newest = held.at(-1);
      if (newest !== void 0 && event.id < newest.id) {
        let i = held.length;
        while (i > 0) {
          const before = held[i - 1];
          if (before === void 0 || before.id <= event.id) break;
          i--;
        }
        held.splice(i, 0, event);
      } else {
        held.push(event);
      }
      if (event.id > lastId) lastId = event.id;
      if (event.id < firstId) firstId = event.id;
      if (settles(event.type)) pendingByJob.delete(event.job_id);
      return true;
    }
    function settles(type) {
      return !type.startsWith("phase.") && type !== "item.observed";
    }
    function passes(e) {
      if (filter.type && !e.type.startsWith(filter.type)) return false;
      if (filter.severity === "warning" && e.severity === "info") return false;
      if (filter.severity === "error" && e.severity !== "error") return false;
      if (filter.job && String(e.job_id) !== filter.job) return false;
      return true;
    }
    function gaps() {
      let found = 0;
      let previous = null;
      for (const e of held) {
        if (previous !== null && e.id !== previous + 1) found++;
        previous = e.id;
      }
      return found;
    }
    const unfiltered = () => !filter.type && !filter.severity && !filter.job;
    function rebuildView() {
      view = held.filter(passes);
      const skipped = gaps();
      heldEl.hidden = skipped === 0;
      if (skipped) heldEl.textContent = `${skipped} gap(s) in the held ids \u2014 click a dashed row to fetch`;
      countEl.textContent = `${view.length} of ${held.length} shown${paused ? ` \xB7 paused, ${heldWhilePaused} new held` : ""}`;
      root.dataset.held = String(held.length);
      root.dataset.lastEventId = String(lastId);
      root.dataset.gaps = String(skipped);
    }
    function rowFor(e, isHead) {
      const li = el("li", {
        class: "tape-row",
        "data-event": e.id,
        "data-type": e.type,
        "data-severity": e.severity,
        "data-job": e.job_id,
        "data-condition": e.condition,
        "data-head": isHead || null,
        "aria-selected": selectedEvent === e.id ? "true" : "false",
        role: "option"
      });
      li.appendChild(el("span", { class: "tape-id" }, `#${e.id}`));
      li.appendChild(el("span", { class: "tape-at" }, clock(e.at)));
      li.appendChild(el("span", { class: "tape-type" }, e.type));
      li.appendChild(el("span", { class: "tape-job" }, `job ${e.job_id}${e.item_id != null ? ` \xB7 ${e.item_id}` : ""}`));
      li.appendChild(el("span", { class: "tape-text", title: e.text }, e.text));
      li.addEventListener("click", () => select(e));
      return li;
    }
    function paint() {
      if (paused) return;
      const total = view.length;
      spacer.style.height = `${total * ROW_H}px`;
      const top = scroller.scrollTop;
      const first = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
      const last = Math.min(total, Math.ceil((top + scroller.clientHeight) / ROW_H) + OVERSCAN);
      rows.style.transform = `translateY(${first * ROW_H}px)`;
      rows.textContent = "";
      const headId = held.at(-1)?.id;
      let previous = first > 0 ? view[first - 1] : void 0;
      for (const e of view.slice(first, last)) {
        if (previous !== void 0 && unfiltered() && e.id !== previous.id + 1) {
          const after = previous;
          const gap = el(
            "li",
            { class: "tape-gap", role: "button", tabindex: "0" },
            `\u2500\u2500 ${e.id - after.id - 1} event(s) not held between #${after.id} and #${e.id} \u2014 fetch \u2500\u2500`
          );
          gap.addEventListener("click", () => void fill(after.id, e.id));
          rows.appendChild(gap);
        }
        rows.appendChild(rowFor(e, e.id === headId));
        previous = e;
      }
    }
    function repaint(scrollToEnd) {
      rebuildView();
      if (paused) return;
      paint();
      if (scrollToEnd && follow.checked) scroller.scrollTop = scroller.scrollHeight;
    }
    function select(e) {
      selectedEvent = e.id;
      rawBody.textContent = JSON.stringify(e, null, 2);
      for (const li of everyElement(rows, "[data-event]", HTMLLIElement)) {
        li.setAttribute("aria-selected", li.dataset.event === String(e.id) ? "true" : "false");
      }
    }
    scroller.addEventListener("scroll", () => {
      if (!paused) paint();
    });
    window.addEventListener("resize", () => {
      if (!paused) paint();
    });
    pauseBtn.addEventListener("click", () => {
      paused = !paused;
      pauseBtn.setAttribute("aria-pressed", String(paused));
      pauseBtn.textContent = paused ? "resume" : "pause";
      if (paused) {
        rebuildView();
      } else {
        heldWhilePaused = 0;
        repaint(true);
      }
    });
    requireElement(root, "[data-tape-filter-type]", HTMLSelectElement).addEventListener("change", (ev) => {
      if (ev.currentTarget instanceof HTMLSelectElement) filter.type = ev.currentTarget.value;
      repaint(true);
    });
    requireElement(root, "[data-tape-filter-severity]", HTMLSelectElement).addEventListener("change", (ev) => {
      if (ev.currentTarget instanceof HTMLSelectElement) filter.severity = ev.currentTarget.value;
      repaint(true);
    });
    jobFilter.addEventListener("input", () => {
      filter.job = jobFilter.value.trim();
      repaint(true);
    });
    requireElement(root, "[data-tape-earlier]", HTMLButtonElement).addEventListener("click", () => void earlier());
    async function fill(after, before) {
      let cursor = after;
      while (cursor < before - 1) {
        const { data } = await api.GET("/operations/events", { params: { query: { after: cursor, limit: TAPE_PAGE } } });
        if (!data) return;
        let advanced = false;
        for (const e of data.events) {
          if (e.id >= before) break;
          ingest(e);
          cursor = e.id;
          advanced = true;
        }
        if (!advanced) break;
      }
      repaint(false);
    }
    async function earlier() {
      if (!Number.isFinite(firstId)) return;
      const { data } = await api.GET("/operations/events/before", {
        params: { query: { before: firstId, limit: TAPE_COLD } }
      });
      if (!data) return;
      const keep = scroller.scrollHeight - scroller.scrollTop;
      for (const e of data.events) ingest(e);
      repaint(false);
      scroller.scrollTop = scroller.scrollHeight - keep;
    }
    let overviewTimer = null;
    function refreshOverviewSoon() {
      if (overviewTimer !== null) return;
      overviewTimer = window.setTimeout(() => {
        overviewTimer = null;
        void loadOverview();
      }, RENDER_DEBOUNCE_MS);
    }
    async function loadOverview() {
      const { data } = await api.GET("/operations/overview");
      if (!data) return;
      paintHealth(data.overview);
      paintMatrix(data.matrix, data.collections);
    }
    function paintHealth(o) {
      const say = (selector, text) => {
        const node = findElement(root, selector, HTMLElement);
        if (node) node.textContent = text;
      };
      const heartbeat = o.worker.heartbeat_age != null ? `${o.worker.heartbeat_age.toFixed(1)}s ago` : "none";
      const stalled = !o.worker.enabled && o.queue.queued > 0;
      const condition = o.worker.working ? "working" : stalled ? "stalled" : o.worker.enabled ? "idle" : "off";
      const workerCell = findElement(root, "[data-health-worker]", HTMLElement);
      if (workerCell) workerCell.dataset.workerCondition = condition;
      say(
        "[data-worker-state]",
        stalled ? `disabled \u2014 ${o.queue.queued} queued, nothing will run` : `${o.worker.enabled ? "enabled" : "disabled"} \xB7 ${o.worker.working ? "working" : "idle"} \xB7 thread ${o.worker.thread_alive ? "alive" : "not running"}`
      );
      say(
        "[data-worker-raw]",
        `${o.worker.thread || "no thread"} \xB7 ${o.worker.owners.length ? o.worker.owners.join(", ") : "no owner"} \xB7 heartbeat ${heartbeat}`
      );
      say("[data-queue-state]", `${o.queue.queued} queued \xB7 ${o.queue.running} running`);
      const oldest = o.queue.oldest_queued_age != null ? `${Math.round(o.queue.oldest_queued_age)}s` : "\u2014";
      const settled = Object.entries(o.queue.settled_24h).map(([state, n]) => `${n} ${state}`).join(", ") || "nothing";
      say("[data-queue-raw]", `oldest queued ${oldest} \xB7 settled 24h ${settled}`);
      say("[data-ledger-state]", `${o.ledger.events.toLocaleString()} events`);
      say("[data-ledger-raw]", `head #${o.ledger.last_id} \xB7 job_event \xB7 never sampled`);
      say("[data-coverage-files]", String(o.coverage.files));
      for (const node of everyElement(document, "[data-missing]", HTMLElement)) {
        const n = o.coverage.missing[requireData(node, "missing")];
        if (n != null) node.textContent = `${n} missing`;
      }
    }
    function paintMatrix(jobs, collections) {
      matrixRows.textContent = "";
      const grouped = /* @__PURE__ */ new Set();
      for (const group of collections) for (const id of group.steps) grouped.add(id);
      const byId = new Map(jobs.map((j) => [j.id, j]));
      for (const group of collections) {
        const holder = el("li", { class: "matrix-collection", "data-matrix-collection": group.name });
        const fold = el("details", {});
        if (group.state === "running" || group.state === "failed") fold.open = true;
        const head2 = el("summary", { class: "matrix-row", "data-collection-state": group.state });
        head2.appendChild(el("span", { class: "matrix-id" }, `${group.steps.length} steps`));
        const kind = el("span", { class: "matrix-kind" });
        kind.appendChild(el("span", { class: "v" }, group.name));
        kind.appendChild(el("code", { class: "raw" }, `${group.settled}/${group.steps.length} settled`));
        head2.appendChild(kind);
        head2.appendChild(el("span", { class: "matrix-state", "data-state": group.state }, group.state));
        const bar = el("progress", { class: "matrix-progress" });
        if (group.total) {
          bar.value = group.done;
          bar.max = group.total;
        }
        head2.appendChild(bar);
        head2.appendChild(
          el("code", { class: "matrix-count" }, `${group.done}${group.total != null ? `/${group.total}` : ""}`)
        );
        if (group.state === "running" || group.state === "queued") {
          const stop = el(
            "button",
            {
              type: "button",
              class: "matrix-stop",
              "data-stop-collection": group.name,
              title: "stop this collection: queued steps end now, a running one stops at its next item"
            },
            "stop"
          );
          stop.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            stop.disabled = true;
            await fetch(`/operations/collections/${encodeURIComponent(group.name)}/stop`, { method: "POST" });
          });
          head2.appendChild(stop);
        }
        fold.appendChild(head2);
        const steps = el("ol", { class: "matrix matrix-steps" });
        for (const id of group.steps) {
          const step = byId.get(id);
          if (step) steps.appendChild(matrixRow(step));
        }
        fold.appendChild(steps);
        holder.appendChild(fold);
        matrixRows.appendChild(holder);
      }
      for (const j of jobs) {
        if (grouped.has(j.id)) continue;
        matrixRows.appendChild(matrixRow(j));
      }
      wireMatrix();
    }
    function matrixRow(j) {
      {
        const cancelling = j.derived.cancellation === "requested";
        const li = el("li", {
          class: "matrix-row",
          "data-matrix-job": j.id,
          "data-state": j.state,
          "data-cancelling": cancelling || null,
          tabindex: "0",
          role: "button",
          "aria-current": selectedJob === j.id ? "true" : null
        });
        li.appendChild(el("span", { class: "matrix-id" }, `#${j.id}`));
        const kind = el("span", { class: "matrix-kind" });
        kind.appendChild(el("span", { class: "v" }, j.what || j.kind.replace(/_/g, " ")));
        kind.appendChild(el("code", { class: "raw" }, j.kind));
        li.appendChild(kind);
        li.appendChild(el("span", { class: "matrix-state", "data-state": j.state }, cancelling ? "cancelling" : j.state));
        const bar = el("progress", { class: "matrix-progress" });
        if (j.total) {
          bar.value = j.done_count;
          bar.max = j.total;
        }
        li.appendChild(bar);
        li.appendChild(
          el(
            "code",
            { class: "matrix-count" },
            `${j.done_count}${j.total != null ? `/${j.total}` : ""}${j.failed_count ? ` \xB7 ${j.failed_count} failed` : ""}`
          )
        );
        li.appendChild(
          el("code", { class: "matrix-exec" }, `a${j.attempt} f${j.fence ?? ""}${j.owner ? ` \xB7 ${j.owner}` : ""}`)
        );
        const live = pendingByJob.get(j.id) ?? j.live;
        if (live && j.state === "running") {
          li.appendChild(el("span", { class: "matrix-live", "data-matrix-live": "" }, liveWords(live)));
        }
        return li;
      }
    }
    function liveWords(live) {
      return `${live.phase || live.type}${live.item_id != null ? ` \xB7 item ${live.item_id}` : ""}`;
    }
    function wireMatrix() {
      for (const li of everyElement(matrixRows, "[data-matrix-job]", HTMLLIElement)) {
        const jobId = Number(requireData(li, "matrixJob"));
        li.onclick = () => choose(jobId);
        li.onkeydown = (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            choose(jobId);
          }
        };
      }
    }
    inspectorBody.addEventListener("click", (ev) => {
      const load = closestFrom(ev.target, "[data-items-load], [data-items-more]", HTMLAnchorElement);
      if (load) {
        ev.preventDefault();
        void loadItems(load);
        return;
      }
      const tapeFilter = closestFrom(ev.target, "[data-tape-job-filter]", HTMLElement);
      if (tapeFilter) {
        ev.preventDefault();
        jobFilter.value = requireData(tapeFilter, "tapeJobFilter");
        filter.job = jobFilter.value;
        repaint(true);
        scroller.scrollIntoView({ block: "start" });
      }
    });
    async function loadItems(link) {
      const slot = findElement(inspectorBody, "[data-items-slot]", HTMLElement);
      if (!slot) return;
      const r = await fetch(link.href, { headers: { accept: "text/html" } });
      if (!r.ok) {
        slot.textContent = `${r.status}`;
        return;
      }
      const fragment = await r.text();
      if (link.hasAttribute("data-items-more")) {
        link.remove();
        slot.insertAdjacentHTML("beforeend", fragment);
      } else {
        slot.innerHTML = fragment;
      }
    }
    let inspectorTimer = null;
    async function loadInspector() {
      const job = selectedJob;
      if (job == null) return;
      const r = await fetch(`/operations/job/${job}`, { headers: { accept: "text/html" } });
      if (!r.ok) {
        inspectorBody.textContent = "";
        inspectorBody.appendChild(el("p", { class: "empty" }, `job ${job}: ${r.status}`));
        return;
      }
      inspectorBody.innerHTML = await r.text();
      window.htmx?.process(inspectorBody);
      for (const node of everyElement(inspectorBody, "time[data-epoch]", HTMLTimeElement)) {
        const epoch = Number(requireData(node, "epoch"));
        const d = new Date(epoch * 1e3);
        node.textContent = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${clock(epoch)}`;
        node.title = `epoch ${epoch}`;
      }
      inspectorHint.textContent = `job #${job} \xB7 refreshed ${clock(Date.now() / 1e3)}`;
      paintPending();
    }
    function refreshInspectorSoon() {
      if (inspectorTimer !== null) return;
      inspectorTimer = window.setTimeout(() => {
        inspectorTimer = null;
        void loadInspector();
      }, 350);
    }
    function choose(jobId) {
      selectedJob = jobId;
      for (const li of everyElement(matrixRows, "[data-matrix-job]", HTMLLIElement)) {
        li.setAttribute("aria-current", Number(li.dataset.matrixJob) === jobId ? "true" : "false");
      }
      void loadInspector();
    }
    function paintPending() {
      const slot = findElement(inspectorBody, "[data-current-phase]", HTMLElement);
      if (!slot) return;
      const p = selectedJob != null ? pendingByJob.get(selectedJob) : void 0;
      if (!p) return;
      slot.textContent = "";
      slot.appendChild(el("span", { class: "v" }, p.phase || p.message || p.type));
      slot.appendChild(document.createTextNode(" "));
      slot.appendChild(el("code", { class: "raw" }, `${p.type} \xB7 ${p.message || ""} \xB7 live, not yet in the ledger`));
    }
    function setTransport(state, text) {
      transport.dataset.transport = state;
      transportState.textContent = text;
    }
    function connect(after) {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const live = new WebSocket(`${proto}://${location.host}/ws/events?after=${after}`);
      socket = live;
      let unreadable = false;
      setTransport(retry ? "reconnecting" : "connecting", retry ? `reconnecting (${retry})` : "connecting");
      live.onopen = () => {
        retry = 0;
        setTransport("connected", "connected");
        refreshOverviewSoon();
      };
      live.onmessage = (msg) => {
        const frame = decodeFrame(msg.data);
        if (frame === null) {
          unreadable = true;
          root.dataset.unreadableFrames = String(Number(root.dataset.unreadableFrames ?? 0) + 1);
          live.close();
          return;
        }
        lastFrameAt = Date.now();
        receive(frame);
      };
      live.onclose = () => {
        setTransport(
          unreadable ? "error" : "disconnected",
          unreadable ? `unreadable frame; resuming from #${lastId}` : "disconnected"
        );
        retry += 1;
        resuming = window.setTimeout(() => connect(lastId), Math.min(4e3, 250 * 2 ** Math.min(retry, 4)));
      };
      live.onerror = () => live.close();
    }
    function receive(frame) {
      if (frame.frame === "backlog") {
        let added = 0;
        for (const e of frame.events) if (ingest(e)) added++;
        if (paused) heldWhilePaused += added;
        repaint(true);
        return;
      }
      if (frame.frame === "pending") {
        pendingByJob.set(frame.job_id, frame);
        if (frame.job_id === selectedJob) paintPending();
        const row = findElement(matrixRows, `[data-matrix-job="${frame.job_id}"]`, HTMLLIElement);
        if (row) {
          let slot = findElement(row, "[data-matrix-live]", HTMLElement);
          if (!slot) {
            slot = el("span", { class: "matrix-live", "data-matrix-live": "" });
            row.appendChild(slot);
          }
          slot.textContent = liveWords(frame);
        }
        return;
      }
      const before = held.at(-1)?.id ?? lastId;
      if (!ingest(frame)) return;
      if (paused) heldWhilePaused++;
      if (frame.id > before + 1 && before > 0) void fill(before, frame.id);
      repaint(true);
      transportLast.textContent = String(lastId);
      if (frame.job_id === selectedJob) refreshInspectorSoon();
      if (settles(frame.type)) refreshOverviewSoon();
    }
    requireElement(root, "[data-transport-reconnect]", HTMLButtonElement).addEventListener("click", () => {
      if (socket && socket.readyState <= 1) {
        socket.close();
        return;
      }
      window.clearTimeout(resuming);
      retry = 0;
      connect(lastId);
    });
    window.setInterval(() => {
      transportLast.textContent = String(lastId);
      transportAge.textContent = lastFrameAt ? `${((Date.now() - lastFrameAt) / 1e3).toFixed(1)}s since last frame` : "no frame yet";
      for (const node of everyElement(inspectorBody, "[data-age-of]", HTMLElement)) {
        node.textContent = `${(Date.now() / 1e3 - Number(requireData(node, "ageOf"))).toFixed(1)}s ago`;
      }
      for (const node of everyElement(inspectorBody, "[data-lease-until]", HTMLElement)) {
        const left = Number(requireData(node, "leaseUntil")) - Date.now() / 1e3;
        node.textContent = left >= 0 ? `expires in ${seconds(left)}` : `expired ${seconds(-left)} ago \xB7 reclaimable`;
        node.classList.toggle("warn", left < 0);
      }
      for (const node of everyElement(inspectorBody, "[data-elapsed-from]", HTMLElement)) {
        const from = Number(requireData(node, "elapsedFrom"));
        if (!from || node.dataset.elapsedTo) continue;
        node.textContent = seconds(Date.now() / 1e3 - from);
      }
    }, 1e3);
    async function cold() {
      if (head <= 0) return;
      const { data } = await api.GET("/operations/events/before", {
        params: { query: { before: head + 1, limit: TAPE_COLD } }
      });
      if (!data) return;
      for (const e of data.events) ingest(e);
      repaint(true);
    }
    wireMatrix();
    repaint(true);
    Promise.allSettled([loadOverview(), cold()]).then(() => connect(head), console.error);
  })();
})();
//# sourceMappingURL=operations.js.map
