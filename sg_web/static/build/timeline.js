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
  function refusal(error, fallback) {
    if (typeof error === "object" && error !== null && "detail" in error && typeof error.detail === "string") {
      return error.detail;
    }
    return fallback;
  }

  // src/dom.ts
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

  // src/jobframes.ts
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  var num = (value) => typeof value === "number" && Number.isFinite(value);
  var str = (value) => typeof value === "string";
  var bool = (value) => typeof value === "boolean";
  var numOrNull = (value) => value === null || num(value);
  var strOrNull = (value) => value === null || str(value);
  function isJob(value) {
    return isRecord(value) && num(value.id) && str(value.kind) && str(value.state) && bool(value.cancel_requested) && numOrNull(value.total) && num(value.done_count) && num(value.created_at) && numOrNull(value.finished_at) && strOrNull(value.derive);
  }
  function isSnapshot(value) {
    return isRecord(value) && value.type === "snapshot" && Array.isArray(value.jobs) && value.jobs.every(isJob);
  }
  function isDelta(value) {
    return isRecord(value) && value.type === "delta" && num(value.job) && str(value.kind) && str(value.state) && num(value.done) && numOrNull(value.total) && bool(value.cancel_requested) && strOrNull(value.derive);
  }
  function decodeJobFrame(payload) {
    if (typeof payload !== "string") return null;
    let held;
    try {
      held = JSON.parse(payload);
    } catch {
      return null;
    }
    if (isSnapshot(held)) return held;
    if (isDelta(held)) return held;
    return null;
  }

  // src/timeline.ts
  var INVALIDATES = /* @__PURE__ */ new Set([
    "scan",
    "context",
    "events",
    "detect_faces",
    "cluster_faces",
    "story_plan"
  ]);
  var SETTLED = /* @__PURE__ */ new Set(["done", "failed", "cancelled"]);
  (() => {
    const swap = findElement(document, "#timeline-swap", HTMLElement);
    if (!swap) return;
    const NARROWEST = 3600;
    const W = 1e3;
    const surface = () => findElement(swap, "[data-surface]", HTMLElement);
    const read = () => {
      const s = surface();
      if (!s || s.dataset.extentStart === void 0) return null;
      return {
        start: Number(s.dataset.windowStart),
        end: Number(s.dataset.windowEnd),
        extentStart: Number(s.dataset.extentStart),
        extentEnd: Number(s.dataset.extentEnd),
        scope: s.dataset.scopeQs ?? ""
      };
    };
    const urlFor = (start, end, snap = false) => {
      const qs = new URLSearchParams(read()?.scope ?? "");
      qs.set("start", String(start));
      qs.set("end", String(end));
      if (snap) qs.set("snap", "true");
      return `/timeline?${qs}`;
    };
    const scopeOf = () => {
      const qs = new URLSearchParams(read()?.scope ?? "");
      const rating = qs.get("rating_min");
      return {
        folder: qs.get("folder"),
        album: qs.get("album"),
        person: qs.get("person"),
        artifact: qs.get("artifact"),
        kind: qs.get("kind"),
        favorite: qs.get("favorite"),
        rating_min: rating === null ? null : Number(rating),
        f: qs.getAll("f")
      };
    };
    let drag = null;
    let generation = 0;
    const settled = (mine) => {
      if (mine === generation) delete swap.dataset.loading;
    };
    const move = async (url, push) => {
      const mine = ++generation;
      swap.dataset.loading = "";
      const answer = await fetch(url, { headers: { "hx-request": "true", accept: "text/html" } });
      if (mine !== generation) return;
      if (!answer.ok) {
        const why = await answer.json().catch(() => null);
        const note = findElement(swap, "[data-note]", HTMLElement);
        if (note) note.textContent = refusal(why, answer.statusText);
        settled(mine);
        return;
      }
      const body = await answer.text();
      if (mine !== generation) return;
      const held = findElement(swap, "[data-strip]", HTMLElement);
      if (held) held.dataset.settling = "";
      swap.innerHTML = body;
      if (drag) {
        const fresh = findElement(swap, "[data-overview]", SVGSVGElement);
        if (fresh && fresh !== drag.overview) fresh.replaceWith(drag.overview);
      }
      thin();
      if (push === true) history.pushState({ url }, "", url);
      else if (push === false) history.replaceState({ url }, "", url);
      settled(mine);
    };
    const revalidate = () => void move(location.pathname + location.search, null);
    const thin = () => {
      const row = findElement(swap, "[data-samples]", HTMLElement);
      if (!row) return;
      const width = row.getBoundingClientRect().width || 1;
      let edge = Number.NEGATIVE_INFINITY;
      for (const a of everyElement(row, ".surface-sample", HTMLElement)) {
        const left = Number.parseFloat(a.style.left) / 100 * width;
        if (left < edge) {
          a.hidden = true;
          continue;
        }
        a.hidden = false;
        edge = left + 42;
      }
    };
    thin();
    window.addEventListener("resize", thin);
    const LIVE_MS = 120;
    let liveAt = 0;
    let liveTimer = 0;
    const live = (start, end, snap = false) => {
      const now = performance.now();
      clearTimeout(liveTimer);
      const run = () => {
        liveAt = performance.now();
        void move(urlFor(Math.round(start), Math.round(end), snap), false);
      };
      if (now - liveAt >= LIVE_MS) run();
      else liveTimer = window.setTimeout(run, LIVE_MS - (now - liveAt));
    };
    (() => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const open = () => {
        const feed = new WebSocket(`${proto}://${location.host}/ws/jobs`);
        feed.onmessage = (msg) => {
          const frame = decodeJobFrame(msg.data);
          if (frame === null) {
            feed.close();
            return;
          }
          if (frame.type === "snapshot") {
            revalidate();
            return;
          }
          if (SETTLED.has(frame.state) && INVALIDATES.has(frame.kind)) revalidate();
        };
        feed.onclose = () => window.setTimeout(open, 2e3);
        feed.onerror = () => feed.close();
      };
      open();
    })();
    window.addEventListener("popstate", (e) => {
      const held = e.state;
      const url = typeof held === "object" && held !== null && "url" in held && typeof held.url === "string" ? held.url : location.pathname + location.search;
      void move(url, false);
    });
    swap.addEventListener("click", (e) => {
      const a = closestFrom(e.target, "[data-preset], [data-bin-window], [data-month-window]", HTMLAnchorElement);
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      void move(a.getAttribute("href") ?? location.pathname + location.search, true);
    });
    const REACH = 0.025;
    const masses = () => {
      const out = [];
      for (const bar of everyElement(swap, ".overview-bar[data-pictures]", SVGRectElement)) {
        const n = Number(bar.dataset.pictures);
        if (n > 0) out.push({ at: Number(bar.dataset.at), end: Number(bar.dataset.end), weight: Math.sqrt(n) });
      }
      return out;
    };
    const pull = (held, t, field = masses()) => {
      const reach = REACH * (held.extentEnd - held.extentStart);
      let force = 0;
      let toward = 0;
      for (const m of field) {
        const d = t < m.at ? m.at - t : t > m.end ? t - m.end : 0;
        if (d === 0 || d > reach) continue;
        const w = m.weight * (1 - d / reach) ** 2;
        force += w;
        toward += w * (t < m.at ? m.at : m.end);
      }
      if (!force) return t;
      const heaviest = Math.max(...field.map((m) => m.weight));
      const grip = Math.min(1, force / heaviest);
      return t + (toward / force - t) * grip;
    };
    const ox = (held, t) => (t - held.extentStart) / Math.max(1, held.extentEnd - held.extentStart) * W;
    const ot = (held, x) => held.extentStart + Math.min(W, Math.max(0, x)) / W * (held.extentEnd - held.extentStart);
    const overviewX = (box, clientX) => (clientX - box.left) / (box.width || 1) * W;
    const placeBrush = (overview, held, start, end) => {
      const x0 = ox(held, start);
      const x1 = ox(held, end);
      const body = findElement(overview, "[data-brush]", SVGRectElement);
      const from = findElement(overview, '[data-brush-edge="start"]', SVGRectElement);
      const to = findElement(overview, '[data-brush-edge="end"]', SVGRectElement);
      if (body) {
        body.setAttribute("x", String(x0));
        body.setAttribute("width", String(Math.max(2, x1 - x0)));
      }
      if (from) from.setAttribute("x", String(x0 - 3));
      if (to) to.setAttribute("x", String(x1 - 3));
    };
    const handAt = (held, event) => {
      if (event.clientX >= held.box.left && event.clientX <= held.box.right) held.last = event.clientX;
      return held.last;
    };
    const dragged = (state, event) => {
      const { held, mode, at } = state;
      const x = overviewX(state.box, handAt(state, event));
      const dt = ot(held, x) - ot(held, at);
      const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
      let start = held.start;
      let end = held.end;
      const field = masses();
      if (mode === "move") {
        const width = end - start;
        end = pull(held, Math.min(Math.max(held.extentStart + width, end + dt), held.extentEnd), field);
        end = Math.min(held.extentEnd, Math.max(held.extentStart + width, end));
        start = end - width;
      } else if (mode === "start") {
        start = Math.max(held.extentStart, Math.min(pull(held, start + dt, field), end - narrowest));
      } else if (mode === "end") {
        end = Math.min(held.extentEnd, Math.max(pull(held, end + dt, field), start + narrowest));
      } else {
        const a = pull(held, ot(held, at), field);
        const b = pull(held, ot(held, x), field);
        start = Math.max(held.extentStart, Math.min(a, b));
        end = Math.min(held.extentEnd, Math.max(a, b, start + narrowest));
      }
      return { start, end };
    };
    swap.addEventListener("pointerdown", (event) => {
      const overview = closestFrom(event.target, "[data-overview]", SVGSVGElement);
      const held = read();
      if (!overview || !held) return;
      const box = overview.getBoundingClientRect();
      const x = overviewX(box, event.clientX);
      const x0 = ox(held, held.start);
      const x1 = ox(held, held.end);
      const grip = 8;
      let mode = "new";
      if (Math.abs(x - x0) <= grip) mode = "start";
      else if (Math.abs(x - x1) <= grip) mode = "end";
      else if (x > x0 && x < x1) mode = "move";
      drag = { overview, box, held, mode, at: x, last: event.clientX };
      overview.setPointerCapture(event.pointerId);
      overview.dataset.dragging = mode;
      event.preventDefault();
    });
    swap.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const { start, end } = dragged(drag, event);
      placeBrush(drag.overview, drag.held, start, end);
      live(start, end, true);
    });
    const release = (event) => {
      if (!drag) return;
      const { start, end } = dragged(drag, event);
      delete drag.overview.dataset.dragging;
      drag = null;
      clearTimeout(liveTimer);
      void move(urlFor(Math.round(start), Math.round(end), true), true).then(() => {
        const held = read();
        if (!held) return;
        const url = urlFor(Math.round(held.start), Math.round(held.end));
        history.replaceState({ url }, "", url);
      });
    };
    let pan = null;
    swap.addEventListener("pointerdown", (event) => {
      const axis = closestFrom(event.target, "[data-strip]", HTMLElement);
      const held = read();
      if (!axis || !held || event.button !== 0) return;
      pan = {
        axis,
        px: axis.getBoundingClientRect().width || 1,
        x: event.clientX,
        start: held.start,
        end: held.end,
        moved: false,
        held
      };
    });
    swap.addEventListener("pointermove", (event) => {
      if (!pan) return;
      if (!pan.moved && Math.abs(event.clientX - pan.x) < 4) return;
      if (!pan.moved) {
        pan.moved = true;
        pan.axis.dataset.dragging = "";
        pan.axis.setPointerCapture(event.pointerId);
      }
      const width = pan.end - pan.start;
      const dt = (pan.x - event.clientX) / pan.px * width;
      let end = pull(pan.held, pan.end + dt);
      end = Math.min(pan.held.extentEnd, Math.max(pan.held.extentStart + width, end));
      live(end - width, end, true);
    });
    let panned = false;
    swap.addEventListener(
      "click",
      (e) => {
        if (panned && closestFrom(e.target, "[data-strip]", Element)) {
          e.preventDefault();
          e.stopImmediatePropagation();
        }
        panned = false;
      },
      true
    );
    const unpan = () => {
      if (!pan) return;
      const was = pan;
      pan = null;
      panned = was.moved;
      if (!was.moved) return;
      delete was.axis.dataset.dragging;
      clearTimeout(liveTimer);
      const held = read();
      if (held) void move(urlFor(Math.round(held.start), Math.round(held.end)), true);
    };
    window.addEventListener("pointerup", unpan);
    window.addEventListener("pointercancel", unpan);
    swap.addEventListener(
      "wheel",
      (e) => {
        const stage = closestFrom(e.target, "[data-strip], [data-overview]", Element);
        const held = read();
        if (!stage || !held || !(e.ctrlKey || e.metaKey || e.shiftKey)) return;
        e.preventDefault();
        const width = held.end - held.start;
        const box = stage.getBoundingClientRect();
        const at = held.start + (e.clientX - box.left) / (box.width || 1) * width;
        let start;
        let end;
        if (e.shiftKey) {
          const step = (e.deltaY > 0 ? 1 : -1) * width / 5;
          start = held.start + step;
          end = held.end + step;
        } else {
          const factor = e.deltaY > 0 ? 1.25 : 0.8;
          start = at - (at - held.start) * factor;
          end = at + (held.end - at) * factor;
        }
        const narrowest = Math.min(NARROWEST, held.extentEnd - held.extentStart);
        if (end - start < narrowest) {
          start = at - narrowest / 2;
          end = at + narrowest / 2;
        }
        start = Math.max(held.extentStart, start);
        end = Math.min(held.extentEnd, Math.max(end, start + narrowest));
        live(start, end);
      },
      { passive: false }
    );
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    swap.addEventListener("keydown", (e) => {
      if (!closestFrom(e.target, "[data-overview]", Element)) return;
      const held = read();
      if (!held) return;
      const width = held.end - held.start;
      const step = width / 4;
      const go = (s, t) => {
        e.preventDefault();
        void move(urlFor(Math.round(s), Math.round(t)), true);
      };
      if (e.key === "ArrowLeft")
        go(Math.max(held.extentStart, held.start - step), Math.max(held.extentStart + width, held.end - step));
      if (e.key === "ArrowRight")
        go(Math.min(held.extentEnd - width, held.start + step), Math.min(held.extentEnd, held.end + step));
      if (e.key === "+" || e.key === "=") go(held.start + width / 4, held.end - width / 4);
      if (e.key === "-")
        go(Math.max(held.extentStart, held.start - width / 2), Math.min(held.extentEnd, held.end + width / 2));
    });
    const segmentAt = (x, y) => {
      for (const el of document.elementsFromPoint(x, y)) {
        const seg = el.closest(".segment");
        if (seg instanceof HTMLElement) return seg;
      }
      return null;
    };
    const nearestWithPictures = (seg, y) => {
      if (Number(seg.dataset.pictures) > 0) return seg;
      let best = seg;
      let nearest = Number.POSITIVE_INFINITY;
      for (const other of everyElement(swap, ".segment", HTMLElement)) {
        if (!(Number(other.dataset.pictures) > 0)) continue;
        const box = other.getBoundingClientRect();
        const d = y < box.top ? box.top - y : y > box.bottom ? y - box.bottom : 0;
        if (d < nearest) {
          nearest = d;
          best = other;
        }
      }
      return best;
    };
    const TILE = 30;
    const fillSegments = () => {
      for (const seg of everyElement(swap, ".segment.held", HTMLElement)) {
        const strip = findElement(seg, "[data-segment-strip]", HTMLElement);
        if (!strip || strip.dataset.filled) continue;
        const box = seg.getBoundingClientRect();
        const cols = Math.max(1, Math.floor(box.width / TILE));
        const rows = Math.max(1, Math.floor(box.height / (TILE + 1)));
        strip.style.setProperty("--cols", String(cols));
        strip.style.setProperty("--tile", `${TILE}px`);
        const n = Math.min(400, cols * rows);
        strip.dataset.filled = String(n);
        void api.GET("/timeline/spread", {
          params: {
            query: {
              ...scopeOf(),
              start: Number(requireData(seg, "at")),
              end: Number(requireData(seg, "end")),
              n
            }
          }
        }).then(({ data }) => {
          if (data === void 0 || !strip.isConnected) return;
          strip.replaceChildren(
            ...data.pictures.map((p) => {
              const img = document.createElement("img");
              img.src = `/thumb/${p.slug}`;
              img.alt = "";
              img.loading = "lazy";
              img.draggable = false;
              img.dataset.moment = String(p.moment);
              return img;
            })
          );
        }, console.error);
      }
    };
    fillSegments();
    new MutationObserver(fillSegments).observe(swap, { childList: true });
    window.addEventListener("resize", () => {
      for (const s of everyElement(swap, "[data-segment-strip]", HTMLElement)) delete s.dataset.filled;
      fillSegments();
    });
    const rankAt = (seg, y) => {
      const box = seg.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (y - box.top) / (box.height || 1)));
      const n = Number(seg.dataset.pictures);
      return Math.min(n - 1, Math.max(0, Math.round((1 - f) * (n - 1))));
    };
    let asking = 0;
    const nth = async (seg, y) => {
      const mine = ++asking;
      const { data } = await api.GET("/timeline/nth", {
        params: {
          query: {
            ...scopeOf(),
            start: Number(requireData(seg, "at")),
            end: Number(requireData(seg, "end")),
            k: rankAt(seg, y)
          }
        }
      });
      if (mine !== asking || data === void 0) return null;
      return data;
    };
    const peek = (seg, y) => {
      const card = findElement(swap, "[data-scrubber-peek]", HTMLElement);
      if (!card) return;
      for (const was of everyElement(swap, ".segment-strip img.under", HTMLElement)) was.classList.remove("under");
      if (!seg) {
        card.hidden = true;
        return;
      }
      const rail = findElement(swap, "[data-scrubber]", HTMLElement);
      const img = findElement(card, "img", HTMLImageElement);
      const label = findElement(card, ".scrubber-peek-label", HTMLElement);
      const count = findElement(card, ".scrubber-peek-count", HTMLElement);
      if (!rail || !img || !label || !count) return;
      const box = rail.getBoundingClientRect();
      card.hidden = false;
      card.style.top = `${Math.min(box.height - 60, Math.max(40, y - box.top))}px`;
      if (!Number(seg.dataset.pictures)) {
        img.removeAttribute("src");
        img.hidden = true;
        label.textContent = seg.dataset.label ?? "";
        count.textContent = "nothing";
        return;
      }
      void nth(seg, y).then((told) => {
        if (!told) return;
        img.src = `/thumb/${told.slug}`;
        img.hidden = false;
        label.textContent = told.spelled;
        count.textContent = `${(told.k + 1).toLocaleString()} of ${told.of.toLocaleString()}`;
        let best = null;
        let nearest = Number.POSITIVE_INFINITY;
        for (const tile of everyElement(seg, ".segment-strip img[data-moment]", HTMLElement)) {
          const d = Math.abs(Number(tile.dataset.moment) - told.moment);
          if (d < nearest) {
            nearest = d;
            best = tile;
          }
        }
        if (best) best.classList.add("under");
      }, console.error);
    };
    let scrub = null;
    let scrubbed = false;
    swap.addEventListener(
      "click",
      (e) => {
        if (scrubbed && closestFrom(e.target, "[data-scrubber]", Element)) {
          e.preventDefault();
          e.stopImmediatePropagation();
        }
        scrubbed = false;
      },
      true
    );
    swap.addEventListener("pointerdown", (event) => {
      const rail = closestFrom(event.target, "[data-scrubber]", HTMLElement);
      const held = read();
      if (!rail || !held || event.button !== 0) return;
      scrub = { held, rail, pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
      event.preventDefault();
    });
    swap.addEventListener("pointermove", (event) => {
      const rail = closestFrom(event.target, "[data-scrubber]", HTMLElement);
      const seg = segmentAt(event.clientX, event.clientY);
      if (rail || scrub) peek(seg, event.clientY);
      const state = scrub;
      if (!state) return;
      if (!state.moved && Math.abs(event.clientY - state.y) < 3) return;
      if (!state.moved) {
        state.moved = true;
        const holding = state.rail.isConnected ? state.rail : findElement(swap, "[data-scrubber]", HTMLElement);
        if (holding) {
          holding.setPointerCapture(state.pointer);
          holding.dataset.dragging = "";
        }
      }
      if (!seg) return;
      const width = state.held.end - state.held.start;
      const target = nearestWithPictures(seg, event.clientY);
      const held = state.held;
      const land = (t) => {
        const end = Math.min(held.extentEnd, Math.max(held.extentStart + width, t));
        live(end - width, end, true);
      };
      if (target !== seg) {
        land(Number(requireData(target, "end")) - 1);
        return;
      }
      void nth(seg, event.clientY).then((told) => {
        if (told && scrub) land(told.moment + 1);
      }, console.error);
    });
    const unscrub = () => {
      if (!scrub) return;
      const was = scrub;
      scrub = null;
      scrubbed = was.moved;
      for (const rail of everyElement(swap, "[data-scrubber]", HTMLElement)) delete rail.dataset.dragging;
      if (!was.moved) return;
      clearTimeout(liveTimer);
      const held = read();
      if (held) void move(urlFor(Math.round(held.start), Math.round(held.end)), true);
    };
    window.addEventListener("pointerup", unscrub);
    window.addEventListener("pointercancel", unscrub);
    swap.addEventListener(
      "pointerleave",
      (e) => {
        if (!scrub && closestFrom(e.target, "[data-scrubber]", Element)) peek(null, 0);
      },
      true
    );
    const ROW = { least: 120, most: 520, fallback: 200, key: "timeline.row" };
    const rowOf = () => {
      try {
        return Number(localStorage.getItem(ROW.key)) || ROW.fallback;
      } catch {
        return ROW.fallback;
      }
    };
    const sizeRows = (px) => {
      const row = Math.min(ROW.most, Math.max(ROW.least, Math.round(px)));
      const s = surface();
      if (s) s.style.setProperty("--row", `${row}px`);
      try {
        localStorage.setItem(ROW.key, String(row));
      } catch {
      }
    };
    const sized = () => {
      const s = surface();
      if (s) s.style.setProperty("--row", `${rowOf()}px`);
    };
    sized();
    new MutationObserver(sized).observe(swap, { childList: true });
    swap.addEventListener(
      "wheel",
      (e) => {
        if (!(e.ctrlKey || e.metaKey) || !closestFrom(e.target, "[data-sessions]", Element)) return;
        e.preventDefault();
        sizeRows(rowOf() * (e.deltaY > 0 ? 0.9 : 1.1));
      },
      { passive: false }
    );
    let pinch = null;
    const apart = (touches) => {
      const a = touches.item(0);
      const b = touches.item(1);
      if (a === null || b === null) return null;
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    };
    swap.addEventListener(
      "touchstart",
      (e) => {
        if (e.touches.length !== 2 || !closestFrom(e.target, "[data-sessions]", Element)) return;
        const distance = apart(e.touches);
        if (distance !== null) pinch = { distance, row: rowOf() };
      },
      { passive: true }
    );
    swap.addEventListener(
      "touchmove",
      (e) => {
        if (!pinch || e.touches.length !== 2) return;
        const distance = apart(e.touches);
        if (distance === null) return;
        e.preventDefault();
        sizeRows(pinch.row * (distance / pinch.distance));
      },
      { passive: false }
    );
    swap.addEventListener("touchend", () => {
      pinch = null;
    });
  })();
})();
//# sourceMappingURL=timeline.js.map
