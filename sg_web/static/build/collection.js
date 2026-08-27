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
  var framed = (question, submit, dismiss, said2) => ({
    question,
    submit: said2.submit !== void 0 ? said2.submit : submit,
    dismiss: said2.dismiss !== void 0 ? said2.dismiss : dismiss,
    ...said2.detail !== void 0 ? { detail: said2.detail } : {},
    ...said2.grave !== void 0 ? { grave: said2.grave } : {}
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
    const read = build2(requireElement(box, ".ask-body", HTMLElement), box);
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
  async function say(message, framing = {}) {
    await ask(framed(message, "ok", null, framing), () => () => void 0);
  }
  async function askYesNo(question, framing = {}) {
    return await ask(framed(question, "yes", "no", framing), () => () => true) === true;
  }

  // src/collection.ts
  var landed = async (result) => {
    if (result.data) {
      window.location.assign(`/t/${result.data.slug}`);
      return;
    }
    await say(refusal(result.error, "the collection did not accept that"));
    if (result.response.status === 409) window.location.reload();
  };
  var said = (form, name) => {
    const held = new FormData(form).get(name);
    return typeof held === "string" && held.trim() ? held : null;
  };
  var asListedKind = (held) => {
    if (held !== "album" && held !== "flag") {
      throw new Error(`data-convert offered ${held}, which is not a listed kind`);
    }
    return held;
  };
  (() => {
    const creating = findElement(document, "[data-new-collection]", HTMLFormElement);
    creating?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const name = said(creating, "name");
      if (!name) return;
      const kind = said(creating, "kind");
      const { data, error } = await api.POST("/albums", {
        body: { name, kind: kind === null ? "album" : asListedKind(kind) }
      });
      if (!data) {
        await say(refusal(error, "the collection could not be created"));
        return;
      }
      window.location.assign(`/t/${data.slug}`);
    });
    const root = findElement(document, "[data-collection]", HTMLElement);
    if (!root) return;
    const slug = requireData(root, "collection");
    const expected_rev = Number(requireData(root, "rev"));
    const editing = findElement(root, "[data-edit-definition]", HTMLFormElement);
    editing?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const name = said(editing, "name");
      if (!name) return;
      await landed(
        await api.PATCH("/t/{slug}", {
          params: { path: { slug } },
          body: {
            expected_rev,
            name,
            description: said(editing, "description"),
            color: said(editing, "color"),
            parent: said(editing, "parent")
          }
        })
      );
    });
    const onClick = (selector, ask2) => {
      const control = findElement(root, selector, HTMLElement);
      control?.addEventListener("click", () => void ask2(control));
    };
    const archived = async (value) => {
      await landed(await api.PATCH("/t/{slug}", { params: { path: { slug } }, body: { expected_rev, archived: value } }));
    };
    onClick("[data-archive]", () => archived(true));
    onClick("[data-restore]", () => archived(false));
    onClick("[data-convert]", async (control) => {
      const wanted = requireData(control, "convert");
      const body = wanted === "smart" ? { kind: "smart", expected_rev } : { kind: asListedKind(wanted), expected_rev, discard_rule: false };
      await landed(await api.POST("/t/{slug}/convert", { params: { path: { slug } }, body }));
    });
    onClick("[data-discard-rule]", async () => {
      const sure = await askYesNo("discard this collection's rule?", {
        detail: "it keeps every file it holds right now and stops following the question",
        submit: "discard the rule",
        dismiss: "keep it",
        grave: true
      });
      if (!sure) return;
      await landed(
        await api.POST("/t/{slug}/convert", {
          params: { path: { slug } },
          body: { kind: "album", expected_rev, discard_rule: true }
        })
      );
    });
  })();

  // src/reread.ts
  var POLL_MS = 400;
  function mountReread(root) {
    for (const button2 of everyElement(root, "[data-folder-reread]", HTMLButtonElement)) {
      button2.addEventListener("click", async () => {
        const folder = requireData(button2, "folderReread");
        button2.disabled = true;
        const was = button2.textContent;
        button2.textContent = "queueing\u2026";
        const { data, error, response } = await api.POST("/jobs/ingest", {
          params: { query: { everything: true, folder } }
        });
        if (!data && response.status !== 204) {
          button2.disabled = false;
          button2.textContent = was;
          await say(refusal(error, "the re-read was not queued"));
          return;
        }
        if (response.status === 204) {
          button2.textContent = "already read";
          return;
        }
        const held = data;
        const job = held?.id;
        if (job === void 0) {
          button2.textContent = "queued";
          return;
        }
        for (; ; ) {
          const told = await api.GET("/jobs/{job_id}", { params: { path: { job_id: job } } });
          const state = told.data?.state;
          if (state === void 0) {
            button2.textContent = "queued";
            return;
          }
          if (state === "done" || state === "failed" || state === "cancelled") {
            const failed = told.data?.failed_count ?? 0;
            button2.textContent = state === "done" && !failed ? `read ${told.data?.done_count ?? 0} again` : `${state}${failed ? ` \u2014 ${failed} could not be read` : ""}`;
            return;
          }
          button2.textContent = `reading\u2026 ${told.data?.done_count ?? 0} of ${told.data?.total ?? "?"}`;
          await new Promise((wake) => setTimeout(wake, POLL_MS));
        }
      });
    }
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
    const RECONNECT_MS = 2e3;
    const TILES_MOST = 400;
    const surface = () => findElement(swap, "[data-surface]", HTMLElement);
    const bands = () => {
      const held = surface()?.dataset.axis;
      if (!held) return [];
      try {
        const told = JSON.parse(held);
        return Array.isArray(told) ? told : [];
      } catch {
        return [];
      }
    };
    const timeAtFraction = (fraction, start, end) => {
      const held = bands();
      const x = Math.min(W, Math.max(0, fraction * W));
      if (!held.length) return start + x / W * (end - start);
      for (const one of held) {
        if (x < one.x1) {
          const drawn = one.x1 - one.x0;
          if (drawn <= 0) return one.t0;
          return one.t0 + (x - one.x0) / drawn * (one.t1 - one.t0);
        }
      }
      return end;
    };
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
        feed.onclose = () => window.setTimeout(open, RECONNECT_MS);
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
      const box = pan.axis.getBoundingClientRect();
      const was = timeAtFraction((pan.x - box.left) / pan.px, pan.start, pan.end);
      const now = timeAtFraction((event.clientX - box.left) / pan.px, pan.start, pan.end);
      const dt = was - now;
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
        const at = stage.matches("[data-strip]") || stage.closest("[data-strip]") ? timeAtFraction((e.clientX - box.left) / (box.width || 1), held.start, held.end) : held.start + (e.clientX - box.left) / (box.width || 1) * width;
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
        const n = Math.min(TILES_MOST, cols * rows);
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
              img.src = p.thumb ?? "";
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
      const label2 = findElement(card, ".scrubber-peek-label", HTMLElement);
      const count = findElement(card, ".scrubber-peek-count", HTMLElement);
      if (!rail || !img || !label2 || !count) return;
      const box = rail.getBoundingClientRect();
      card.hidden = false;
      card.style.top = `${Math.min(box.height - 60, Math.max(40, y - box.top))}px`;
      if (!Number(seg.dataset.pictures)) {
        img.removeAttribute("src");
        img.hidden = true;
        label2.textContent = seg.dataset.label ?? "";
        count.textContent = "nothing";
        return;
      }
      void nth(seg, y).then((told) => {
        if (!told) return;
        img.src = told.thumb ?? "";
        img.hidden = false;
        label2.textContent = told.spelled;
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
  mountReread(document);

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
    const cell = under ?? focused;
    if (cell?.dataset.slug) {
      const shown = cell.querySelector("img");
      return {
        slug: cell.dataset.slug,
        name: shown?.getAttribute("alt") || cell.dataset.slug,
        kind: cell.dataset.kind ?? "",
        thumb: shown?.getAttribute("src") ?? ""
      };
    }
    return null;
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
    const said2 = document.createElement("span");
    said2.className = "compare-view-said";
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
    bar.append(said2, modes, zoom, close);
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
      said2.textContent = mode === "side" ? `${held.length} side by side` : `${letter(at)} of ${held.length} \xB7 ${one ? one.name : ""}`;
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
    drawTray(tray);
  }

  // src/compare-mount.ts
  mountCompare(document.body);

  // src/pictures.ts
  function label(kind) {
    const said2 = document.createElement("span");
    said2.className = "cell-kind";
    said2.dataset.cellKind = kind ?? "";
    said2.dataset.brokenPicture = "";
    said2.setAttribute("aria-hidden", "true");
    said2.textContent = kind === "audio" ? "audio" : "doc";
    return said2;
  }
  function mountPictures() {
    document.addEventListener(
      "error",
      (event) => {
        const broken = event.target;
        if (!(broken instanceof HTMLImageElement)) return;
        const src = broken.getAttribute("src") ?? "";
        if (!src.startsWith("/thumbs/") && !src.startsWith("/thumb/") && !src.startsWith("/preview/")) return;
        if (!broken.isConnected) return;
        const holder = broken.closest("[data-kind]");
        const kind = holder instanceof HTMLElement ? holder.dataset.kind : void 0;
        broken.replaceWith(label(kind));
      },
      // The capture phase, because `error` on an <img> does not bubble.
      true
    );
  }

  // src/pictures-mount.ts
  mountPictures();
})();
//# sourceMappingURL=collection.js.map
