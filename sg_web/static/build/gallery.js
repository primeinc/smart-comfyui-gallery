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
  function answered(result, fallback) {
    if (result.data !== void 0) return { ok: true, data: result.data };
    return { ok: false, refusal: refusal(result.error, fallback) };
  }
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
    if (target instanceof Element && target.closest("input, textarea, select, [contenteditable]")) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const command = claimed.get(spelled(event.key));
    if (!command) return;
    event.preventDefault();
    command.run();
  });

  // src/overlay.ts
  function isPlainClick(event, link) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && link?.getAttribute("target") !== "_blank";
  }
  function addressableOverlay(spec) {
    const root = findElement(document, spec.root, HTMLElement);
    if (!root) return null;
    root.tabIndex = -1;
    let flight = 0;
    let opener = null;
    const underlay = (frozen) => {
      for (const el of document.body.children) {
        if (el !== root && el.tagName !== "SCRIPT" && el instanceof HTMLElement) el.inert = frozen;
      }
    };
    const open = async (href, mode) => {
      const ticket = ++flight;
      let mended = false;
      while (true) {
        const headers = { "HX-Request": "true" };
        const expected = spec.generation ? spec.generation() : null;
        if (expected) headers["X-SG-Expect"] = expected;
        let answer;
        try {
          answer = await fetch(href, { headers });
        } catch {
          if (ticket !== flight) return;
          window.location.assign(href);
          return;
        }
        if (ticket !== flight) return;
        if (!answer.ok) {
          if (answer.status === 409 && spec.recover && !mended) {
            let proven = false;
            try {
              proven = await spec.recover();
            } catch {
              proven = false;
            }
            if (ticket !== flight) return;
            if (proven) {
              mended = true;
              continue;
            }
          }
          window.location.assign(href);
          return;
        }
        let fragment;
        try {
          fragment = await answer.text();
        } catch {
          if (ticket !== flight) return;
          window.location.assign(href);
          return;
        }
        if (ticket !== flight) return;
        if (expected) {
          const got = /data-currency="([^"]*)"/.exec(fragment);
          if (!got?.[1] || got[1] !== expected) {
            window.location.assign(href);
            return;
          }
        }
        root.innerHTML = fragment;
        spec.mounted?.(root);
        if (root.hidden) {
          root.hidden = false;
          underlay(true);
        }
        if (mode === "push") history.pushState({ sgOverlay: true }, "", href);
        else if (mode === "replace") history.replaceState({ sgOverlay: true }, "", href);
        root.focus();
        return;
      }
    };
    const close = () => {
      flight += 1;
      root.hidden = true;
      root.replaceChildren();
      spec.mounted?.(null);
      underlay(false);
      if (opener?.isConnected) opener.focus();
      opener = null;
    };
    document.addEventListener("click", (event) => {
      const trigger = closestFrom(event.target, spec.trigger, HTMLElement);
      if (trigger) {
        if (!isPlainClick(event, trigger)) return;
        const href = trigger.getAttribute("href");
        if (!href) return;
        event.preventDefault();
        opener = trigger;
        void open(href, "push");
        return;
      }
      if (event.target === root || closestFrom(event.target, "[data-close]", Element)) {
        event.preventDefault();
        history.back();
      }
    });
    register([
      {
        key: "Escape",
        by: `overlay: ${spec.pathPrefix}`,
        run: () => {
          if (root.hidden) return;
          if (spec.dismiss?.()) return;
          history.back();
        }
      }
    ]);
    window.addEventListener("popstate", () => {
      if (window.location.pathname.startsWith(spec.pathPrefix)) void open(window.location.href, "none");
      else if (!root.hidden) close();
    });
    return { root, open, close };
  }

  // src/recipe.ts
  function fit(field) {
    if (field.clientWidth === 0) return;
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
  }
  async function copied(button, text) {
    const was = button.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "copied";
      button.dataset.done = "true";
    } catch {
      button.textContent = "cannot copy";
    }
    setTimeout(() => {
      button.textContent = was;
      delete button.dataset.done;
    }, 1200);
  }
  function scratchOf(root, named) {
    const section = findElement(root, `[data-recipe-field="${named}"]`, HTMLElement);
    const field = section && findElement(section, "[data-scratch]", HTMLTextAreaElement);
    return field ? field.value : "";
  }
  function wholeRecipe(root) {
    const lines = [];
    const prompt = scratchOf(root, "prompt");
    if (prompt.trim()) lines.push(prompt.trim());
    const negative = scratchOf(root, "negative");
    if (negative.trim()) lines.push(`Negative prompt: ${negative.trim()}`);
    const pairs = [];
    for (const value of everyElement(root, "[data-recipe-key]", HTMLElement)) {
      const key = value.dataset.recipeKey ?? "";
      const text = (value.dataset.recipeValue ?? value.textContent ?? "").trim();
      if (key && text) pairs.push(`${key}: ${text}`);
    }
    const checkpoint = findElement(root, "[data-recipe-checkpoint]", HTMLElement);
    if (checkpoint) pairs.push(`Model: ${(checkpoint.textContent ?? "").trim()}`);
    const loras = [...everyElement(root, ".recipe-lora", HTMLElement)].map((row) => {
      const name = row.querySelector("span:not(.recipe-label)")?.textContent?.trim() ?? "";
      const weight = row.querySelector(".recipe-weight")?.textContent?.trim();
      return { name, tag: weight ? `<lora:${name}:${weight}>` : `<lora:${name}>` };
    }).filter((lora) => lora.name && !prompt.includes(`<lora:${lora.name}`)).map((lora) => lora.tag);
    if (loras.length) pairs.push(`Loras: ${loras.join(" ")}`);
    if (pairs.length) lines.push(pairs.join(", "));
    return lines.join("\n");
  }
  function mountRecipe(root) {
    const panel = findElement(root, "[data-recipe]", HTMLElement);
    if (!panel) return;
    for (const section of everyElement(panel, "[data-recipe-field]", HTMLElement)) {
      const field = findElement(section, "[data-scratch]", HTMLTextAreaElement);
      const revert = findElement(section, "[data-revert]", HTMLElement);
      if (!field) continue;
      const original = field.value;
      fit(field);
      field.addEventListener("input", () => {
        fit(field);
        if (revert) revert.hidden = field.value === original;
      });
      if (revert) {
        revert.addEventListener("click", () => {
          field.value = original;
          fit(field);
          revert.hidden = true;
          field.focus();
        });
      }
      const copy = findElement(section, "[data-copy]", HTMLElement);
      if (copy) copy.addEventListener("click", () => void copied(copy, field.value));
    }
    const all = findElement(panel, "[data-copy-all]", HTMLElement);
    if (all) all.addEventListener("click", () => void copied(all, wholeRecipe(panel)));
    const refit = () => {
      for (const field of everyElement(panel, "[data-scratch]", HTMLTextAreaElement)) fit(field);
    };
    panel.addEventListener("toggle", refit);
    new MutationObserver(refit).observe(root, { attributeFilter: ["data-inspector"] });
    refit();
  }

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
  function panelState(name) {
    return workspace().panels?.[name];
  }
  function rememberPanel(name, open) {
    remember({ panels: { ...workspace().panels ?? {}, [name]: open } });
  }

  // src/viewer.ts
  var FIT = { framing: "fit", scale: 1, x: 0, y: 0 };
  function isStill(stage) {
    return stage.kind === "image";
  }
  function actualScale(source, fitted) {
    const wanted = source.width / (window.devicePixelRatio || 1);
    return fitted.width > 0 ? wanted / fitted.width : 1;
  }
  function fillScale(fitted, box) {
    if (fitted.width <= 0 || fitted.height <= 0) return 1;
    return Math.max(box.width / fitted.width, box.height / fitted.height);
  }
  var MIN_SCALE = 1;
  var MAX_SCALE = 40;
  var IDLE_MS = 2200;
  var ABSORBED = 1.35;
  function mountViewer(root, walk) {
    const stageBox = findElement(root, "[data-stage]", HTMLElement);
    if (!stageBox) return null;
    const stage = JSON.parse(requireData(stageBox, "stage"));
    const media = findElement(stageBox, "[data-stage-media]", HTMLElement);
    const still = isStill(stage) ? stage : null;
    let look = { ...FIT };
    let promoted = false;
    let idle = 0;
    const bound = [];
    const onElement = (target, type, listener, options) => {
      target.addEventListener(type, listener, options);
      bound.push(() => target.removeEventListener(type, listener));
    };
    const onDocument = (type, listener) => {
      document.addEventListener(type, listener);
      bound.push(() => document.removeEventListener(type, listener));
    };
    const fitted = () => {
      const rect = (media ?? stageBox).getBoundingClientRect();
      return { width: rect.width / look.scale, height: rect.height / look.scale };
    };
    const tethered = (x, y, scale) => {
      const size = fitted();
      const box = stageBox.getBoundingClientRect();
      const room = (picture, stage2) => Math.max(0, (picture * scale - stage2) / 2);
      const across = room(size.width, box.width);
      const down = room(size.height, box.height);
      return {
        x: Math.min(across, Math.max(-across, x)),
        y: Math.min(down, Math.max(-down, y))
      };
    };
    const paint = () => {
      if (!media) return;
      media.style.transform = `translate(${look.x}px, ${look.y}px) scale(${look.scale})`;
      stageBox.dataset.framing = look.framing;
      root.dataset.absorbed = look.scale > ABSORBED ? "true" : "false";
      root.dataset.zoom = String(Math.round(look.scale * 100));
    };
    const promote = () => {
      if (!still || promoted || !still.promotable || !still.shown) return;
      const wanted = fitted().width * look.scale * (window.devicePixelRatio || 1);
      if (wanted <= still.shown.width) return;
      promoted = true;
      const full = new Image();
      full.src = still.original;
      void full.decode().then(() => {
        const img = findElement(stageBox, "img[data-stage-media]", HTMLImageElement);
        if (img) img.src = still.original;
        stageBox.dataset.quality = "original";
      }).catch(() => {
        promoted = false;
      });
    };
    const clamp = (scale) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
    const frame = (framing) => {
      if (!still) return;
      const box = stageBox.getBoundingClientRect();
      const size = fitted();
      const scale = framing === "fit" ? 1 : framing === "fill" ? fillScale(size, box) : still.source ? actualScale(still.source, size) : 1;
      look = { framing, scale: clamp(scale), x: 0, y: 0 };
      paint();
      promote();
    };
    const zoomAbout = (factor, clientX, clientY) => {
      if (!still) return;
      const box = stageBox.getBoundingClientRect();
      const next = clamp(look.scale * factor);
      if (next === look.scale) return;
      const px = clientX - (box.left + box.width / 2);
      const py = clientY - (box.top + box.height / 2);
      const ratio = next / look.scale;
      const held = tethered(px - (px - look.x) * ratio, py - (py - look.y) * ratio, next);
      look = { framing: "free", scale: next, ...held };
      paint();
      promote();
    };
    const resettle = () => {
      if (!still) return;
      if (look.framing === "fit" || look.framing === "fill" || look.framing === "actual") {
        frame(look.framing);
        return;
      }
      look = { ...look, ...tethered(look.x, look.y, look.scale) };
      paint();
    };
    const watching = new ResizeObserver(() => resettle());
    watching.observe(stageBox);
    bound.push(() => watching.disconnect());
    const wake = () => {
      if (root.dataset.chrome === "focus") return;
      root.dataset.chrome = "visible";
      window.clearTimeout(idle);
      idle = window.setTimeout(() => {
        if (root.dataset.chrome === "visible") root.dataset.chrome = "resting";
      }, IDLE_MS);
    };
    const focus = () => {
      window.clearTimeout(idle);
      root.dataset.chrome = root.dataset.chrome === "focus" ? "visible" : "focus";
      if (root.dataset.chrome === "visible") wake();
    };
    const inspector = findElement(root, "[data-inspector-panel]", HTMLElement);
    const showInspector = (open, arranged = true) => {
      if (!inspector) return;
      root.dataset.inspector = open ? "open" : "closed";
      for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
        button.setAttribute("aria-expanded", String(open));
      }
      if (arranged) remember({ inspector: open ? "open" : "closed" });
    };
    const panel = (named) => {
      if (!inspector) return;
      for (const section of everyElement(inspector, "[data-panel]", HTMLDetailsElement)) {
        if (section.dataset.panel !== named) continue;
        section.open = true;
        section.scrollIntoView({ block: "nearest" });
      }
    };
    const walksOnWheel = () => root.dataset.wheelModifier === "alt";
    const NOTCH = 90;
    const QUIET_MS = 260;
    let rolled = 0;
    let spent = false;
    let lastWheel = Number.NEGATIVE_INFINITY;
    const stepped = (by) => {
      if (by === 0) return;
      const now = performance.now();
      const reversed = rolled !== 0 && Math.sign(by) !== Math.sign(rolled);
      if (now - lastWheel > QUIET_MS || reversed) {
        rolled = 0;
        spent = false;
      }
      lastWheel = now;
      rolled += by;
      if (spent || Math.abs(rolled) < NOTCH) return;
      const step = findElement(root, `[data-nav="${rolled > 0 ? "next" : "previous"}"]`, HTMLAnchorElement);
      spent = true;
      if (step) walk(step.href);
    };
    onElement(
      stageBox,
      "wheel",
      (event) => {
        const plain = !event.altKey && !event.ctrlKey && !event.shiftKey && !event.metaKey;
        const walking = event.altKey && !event.ctrlKey && !event.shiftKey && !event.metaKey && walksOnWheel();
        if (!plain && !walking) return;
        event.preventDefault();
        const pixels = (delta) => event.deltaMode === 0 ? delta : delta * 16;
        if (walking) {
          stepped(pixels(event.deltaY || event.deltaX));
          return;
        }
        zoomAbout(Math.exp(-pixels(event.deltaY) / 400), event.clientX, event.clientY);
      },
      // not passive: the page must not scroll out from under a zoom
      { passive: false }
    );
    if (still && media) {
      onElement(stageBox, "dblclick", (event) => {
        event.preventDefault();
        frame(look.framing === "actual" ? "fit" : "actual");
      });
      let dragging = null;
      let from = { x: 0, y: 0, ox: 0, oy: 0 };
      onElement(stageBox, "pointerdown", (event) => {
        if (event.button !== 0 || look.scale <= 1) return;
        stageBox.setPointerCapture(event.pointerId);
        dragging = event.pointerId;
        from = { x: event.clientX, y: event.clientY, ox: look.x, oy: look.y };
        stageBox.dataset.panning = "true";
      });
      onElement(stageBox, "pointermove", (event) => {
        if (dragging !== event.pointerId) return;
        const held = tethered(from.ox + (event.clientX - from.x), from.oy + (event.clientY - from.y), look.scale);
        look = { framing: "free", scale: look.scale, ...held };
        paint();
      });
      const release = (event) => {
        if (dragging !== event.pointerId) return;
        dragging = null;
        delete stageBox.dataset.panning;
        if (stageBox.hasPointerCapture(event.pointerId)) stageBox.releasePointerCapture(event.pointerId);
      };
      onElement(stageBox, "pointerup", release);
      onElement(stageBox, "pointercancel", release);
    }
    const stepping = (wanted) => () => {
      const step = findElement(root, `[data-nav="${wanted}"]`, HTMLAnchorElement);
      if (step) walk(step.href);
    };
    const middle = (by) => () => zoomAbout(by, window.innerWidth / 2, window.innerHeight / 2);
    bound.push(
      register([
        { key: "z", by: "viewer: fit/actual", run: () => frame(look.framing === "actual" ? "fit" : "actual") },
        { key: "l", by: "viewer: focus", run: focus },
        { key: "i", by: "viewer: inspector", run: () => showInspector(root.dataset.inspector !== "open") },
        { key: "+", by: "viewer: zoom in", run: middle(1.3) },
        { key: "=", by: "viewer: zoom in", run: middle(1.3) },
        { key: "-", by: "viewer: zoom out", run: middle(1 / 1.3) },
        { key: "ArrowRight", by: "viewer: next", run: stepping("next") },
        { key: "ArrowLeft", by: "viewer: previous", run: stepping("previous") }
      ])
    );
    onDocument("pointermove", wake);
    const strip2 = findElement(root, "[data-filmstrip-track]", HTMLElement);
    if (strip2) {
      const here = findElement(strip2, "[data-filmstrip-item][aria-current='true']", HTMLElement);
      here?.scrollIntoView({ block: "nearest", inline: "center" });
      onElement(strip2, "click", (event) => {
        const near = closestFrom(event.target, "[data-filmstrip-item]", HTMLAnchorElement);
        if (!near || !isPlainClick(event, near)) return;
        event.preventDefault();
        walk(near.href);
      });
    }
    for (const button of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
      onElement(button, "click", () => showInspector(root.dataset.inspector !== "open"));
    }
    for (const button of everyElement(root, "[data-panel-open]", HTMLElement)) {
      onElement(button, "click", () => {
        showInspector(true);
        panel(requireData(button, "panelOpen"));
      });
    }
    for (const button of everyElement(root, "[data-focus]", HTMLElement)) {
      onElement(button, "click", focus);
    }
    if (inspector) {
      const kept = workspace();
      const generated = inspector.querySelector("[data-panel='creation']") !== null && root.dataset.made === "generated";
      showInspector(kept.inspector ? kept.inspector === "open" : false, false);
      for (const section of everyElement(inspector, "[data-panel]", HTMLDetailsElement)) {
        const named = section.dataset.panel ?? "";
        const said = panelState(named);
        section.open = said ?? (generated ? named === "creation" : named === "about");
        onElement(section, "toggle", () => rememberPanel(named, section.open));
      }
    }
    mountRecipe(root);
    root.dataset.inspector = root.dataset.inspector ?? "closed";
    root.dataset.chrome = "visible";
    stageBox.dataset.quality = "preview";
    paint();
    wake();
    return {
      /**
       * Escape unwinds what the viewer is doing before it means "leave".
       *
       * The ladder, outermost state first: a picture pushed off centre comes
       * back, a zoomed picture fits, an open inspector closes, hidden chrome
       * returns. Only a viewer with nothing left to undo hands Escape to its
       * container -- which is why the container asks rather than owning the
       * key: one shared shell, one dismissal, and no competing listener.
       */
      unwind: () => {
        if (look.scale > 1 || look.x !== 0 || look.y !== 0) {
          frame("fit");
          return true;
        }
        if (root.dataset.inspector === "open") {
          showInspector(false);
          return true;
        }
        if (root.dataset.chrome === "focus") {
          focus();
          return true;
        }
        return false;
      },
      release: () => {
        window.clearTimeout(idle);
        for (const off of bound) off();
        bound.length = 0;
      }
    };
  }

  // src/gallery.ts
  var asked = (spelled2, take) => {
    const question = new URLSearchParams(spelled2);
    const one = (name) => question.get(name);
    const counted = (name) => {
      const held = question.get(name);
      return held === null ? null : Number(held);
    };
    return {
      take,
      folder: one("folder"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted("rating_min"),
      q: one("q"),
      sort: one("sort"),
      f: question.getAll("f")
    };
  };
  (() => {
    const ask = findElement(document, "[data-ask]", HTMLFormElement);
    if (ask) {
      const fields = () => [
        ...everyElement(ask, "input", HTMLInputElement),
        ...everyElement(ask, "select", HTMLSelectElement)
      ];
      ask.addEventListener("submit", () => {
        const phrase = requireElement(ask, '[name="q"]', HTMLInputElement);
        const sort = requireElement(ask, '[name="sort"]', HTMLSelectElement);
        if (phrase.value.trim()) sort.value = "similarity";
        else if (sort.value === "similarity") sort.value = "newest";
        for (const field of fields()) {
          if (!field.value.trim()) field.disabled = true;
        }
      });
      window.addEventListener("pageshow", () => {
        for (const field of fields()) field.disabled = false;
      });
    }
    const grid = () => findElement(document, "[data-grid]", HTMLElement);
    const spelling = () => {
      const mounted2 = grid();
      const held = mounted2 ? requireData(mounted2, "qbase").replace(/&$/, "") : window.location.search;
      const question = new URLSearchParams(held);
      question.delete("page");
      question.delete("size");
      return question.toString();
    };
    const cutoff = (spelled2) => {
      if (!new URLSearchParams(spelled2).get("q")) return null;
      const held = window.prompt("how many top results belong to it?", "100");
      return held === null ? void 0 : Number(held);
    };
    const saver = findElement(document, "[data-save-smart]", HTMLElement);
    saver?.addEventListener("click", async () => {
      const spelled2 = spelling();
      const name = window.prompt("name this smart collection");
      if (!name) return;
      const take = cutoff(spelled2);
      if (take === void 0) return;
      const { data, error } = await api.POST("/albums/smart", { body: { name, ...asked(spelled2, take) } });
      if (!data) {
        window.alert(refusal(error, "the view could not be saved"));
        return;
      }
      window.location.assign(`/t/${data.slug}`);
    });
    const replacer = findElement(document, "[data-replace-smart]", HTMLElement);
    replacer?.addEventListener("click", async () => {
      const shelf = await api.GET("/albums", { headers: { accept: "application/json" } });
      const smarts = (shelf.data ?? []).filter((held) => held.kind === "smart");
      const first = smarts[0];
      if (first === void 0) {
        window.alert("no smart collection exists yet -- save the view as a new one instead");
        return;
      }
      const named = window.prompt(
        `replace the rule of which smart collection?
${smarts.map((held) => held.slug).join(", ")}`,
        first.slug
      );
      if (!named) return;
      const current = await api.GET("/t/{slug}", {
        params: { path: { slug: named } },
        headers: { accept: "application/json" }
      });
      if (!current.data) {
        window.alert(`no collection at /t/${named}`);
        return;
      }
      const spelled2 = spelling();
      const take = cutoff(spelled2);
      if (take === void 0) return;
      const { data, error } = await api.PUT("/t/{slug}/rule", {
        params: { path: { slug: named } },
        body: { expected_rev: current.data.definition_rev, ...asked(spelled2, take) }
      });
      if (!data) {
        window.alert(refusal(error, "the rule could not be replaced"));
        return;
      }
      window.location.assign(`/t/${data.slug}`);
    });
    const rail = findElement(document, "[data-rail]", HTMLElement);
    if (!rail) return;
    const thumb = requireElement(rail, "[data-rail-thumb]", HTMLElement);
    const pop = requireElement(rail, "[data-rail-pop]", HTMLElement);
    const popLabel = requireElement(pop, "[data-rail-pop-label]", HTMLElement);
    const popGrid = requireElement(pop, "[data-rail-pop-grid]", HTMLElement);
    const shape = () => {
      const mounted2 = grid();
      if (!mounted2) return null;
      return {
        page: Number(requireData(mounted2, "page")),
        pages: Number(requireData(mounted2, "pages")),
        currency: requireData(mounted2, "currency"),
        answer: requireData(mounted2, "answer"),
        qbase: requireData(mounted2, "qbase")
      };
    };
    const pageAt = (clientY, s) => {
      const box = rail.getBoundingClientRect();
      const fraction = Math.min(1, Math.max(0, (clientY - box.top) / box.height));
      return Math.min(s.pages, Math.max(1, Math.round(fraction * (s.pages - 1)) + 1));
    };
    const bar = findElement(document, "header.bar", HTMLElement);
    const placeRail = () => {
      if (bar) rail.style.top = `${bar.getBoundingClientRect().bottom}px`;
    };
    const placeThumb = () => {
      placeRail();
      const s = shape();
      if (!s) return;
      const fraction = s.pages > 1 ? (s.page - 1) / (s.pages - 1) : 0;
      thumb.style.top = `${fraction * 100}%`;
    };
    window.addEventListener("resize", placeRail);
    const peeked = /* @__PURE__ */ new Map();
    const peek = async (page, s) => {
      const key = `${s.answer}:${page}`;
      const held = peeked.get(key);
      if (held) return held;
      const question = new URLSearchParams(s.qbase);
      const { data } = await api.GET("/g/peek", {
        params: {
          query: {
            folder: question.get("folder"),
            album: question.get("album"),
            person: question.get("person"),
            artifact: question.get("artifact"),
            kind: question.get("kind"),
            favorite: question.get("favorite"),
            rating_min: question.get("rating_min") === null ? null : Number(question.get("rating_min")),
            q: question.get("q"),
            sort: question.get("sort"),
            f: question.getAll("f"),
            page,
            count: 9
          }
        }
      });
      if (!data) return null;
      if (data.answer !== s.answer) {
        window.location.reload();
        return null;
      }
      peeked.set(key, data);
      return data;
    };
    const REST_MS = 60;
    let hoverPage = null;
    let resting = 0;
    const show = async (page, s) => {
      const told = await peek(page, s);
      if (!told || hoverPage !== page) return;
      popLabel.textContent = `page ${told.page} of ${told.pages} \xB7 ${told.first_ordinal}\u2013${told.last_ordinal} of ${told.total}`;
      popGrid.replaceChildren(
        ...told.items.map((item) => {
          const img = new Image();
          img.src = `/thumb/${item.slug}`;
          img.alt = item.name;
          return img;
        })
      );
      placePop();
    };
    const MARGIN = 8;
    let pointerY = 0;
    const placePop = () => {
      const top = bar ? bar.getBoundingClientRect().bottom + MARGIN : MARGIN;
      const height = pop.offsetHeight;
      const floor = Math.max(top, window.innerHeight - height - MARGIN);
      pop.style.top = `${Math.min(Math.max(pointerY - height / 2, top), floor)}px`;
    };
    rail.addEventListener("pointermove", (event) => {
      const s = shape();
      if (!s) return;
      const page = pageAt(event.clientY, s);
      pointerY = event.clientY;
      pop.hidden = false;
      placePop();
      if (page === hoverPage) return;
      hoverPage = page;
      clearTimeout(resting);
      resting = window.setTimeout(() => void show(page, s), REST_MS);
    });
    rail.addEventListener("pointerleave", () => {
      clearTimeout(resting);
      pop.hidden = true;
      hoverPage = null;
    });
    rail.addEventListener("click", (event) => {
      const s = shape();
      if (!s) return;
      window.location.assign(`/g?${s.qbase}page=${pageAt(event.clientY, s)}`);
    });
    placeThumb();
    document.body.addEventListener("htmx:afterSwap", placeThumb);
    let viewer = null;
    const lightbox = addressableOverlay({
      root: "[data-lightbox-root]",
      trigger: "a.cell",
      pathPrefix: "/i/",
      dismiss: () => viewer?.unwind() ?? false,
      mounted: (mounted2) => {
        viewer?.release();
        const held = mounted2 && findElement(mounted2, "[data-viewer]", HTMLElement);
        viewer = held ? mountViewer(held, (href) => void lightbox?.open(href, "replace")) : null;
      },
      generation: () => {
        const shown = findElement(document, "[data-lightbox]", HTMLElement);
        const held = shown?.dataset.currency;
        if (held) return held;
        const s = shape();
        return s ? s.currency : "";
      },
      // A 409'd arrow proves the generation moved, not that THIS answer did
      // -- a favorite, a background job's bookkeeping, any commit at all
      // moves data_version. Ask locate for the walked context's (currency,
      // answer): the same answer identity means the mounted walk is still
      // true, so adopt the fresh currency and let the shell retry once. A
      // changed or vanished answer stays a full redraw.
      recover: async () => {
        const shown = findElement(document, "[data-lightbox]", HTMLElement);
        const mounted2 = shown?.dataset.answer || grid()?.dataset.answer || "";
        const slug = shown?.dataset.slug;
        if (!mounted2 || !slug) return false;
        const question = new URLSearchParams(window.location.search);
        const { data } = await api.GET("/g/locate/{slug}", {
          params: {
            path: { slug },
            query: {
              folder: question.get("folder"),
              album: question.get("album"),
              person: question.get("person"),
              artifact: question.get("artifact"),
              kind: question.get("kind"),
              favorite: question.get("favorite"),
              rating_min: question.get("rating_min") === null ? null : Number(question.get("rating_min")),
              q: question.get("q"),
              sort: question.get("sort"),
              size: question.get("size") === null ? null : Number(question.get("size")),
              f: question.getAll("f")
            }
          }
        });
        if (!data?.in_answer || data.answer !== mounted2) return false;
        for (const surface of [shown, grid()]) {
          if (surface) {
            surface.dataset.currency = data.currency;
            surface.dataset.answer = data.answer;
          }
        }
        return true;
      }
    });
    if (lightbox) {
      document.addEventListener("click", (event) => {
        const nav = closestFrom(event.target, "[data-nav]", HTMLAnchorElement);
        if (nav && isPlainClick(event, nav)) {
          event.preventDefault();
          lightbox.open(nav.href, "replace");
        }
      });
    }
  })();

  // src/authored.ts
  var draw = (root, authored) => {
    requireElement(root, "[data-fav]", HTMLElement).setAttribute("aria-pressed", authored.favorite ? "true" : "false");
    const stars = requireElement(root, "[data-stars]", HTMLElement);
    stars.dataset.rating = String(authored.rating ?? 0);
    for (const star of everyElement(stars, "[data-rate]", HTMLElement)) {
      const n = Number(requireData(star, "rate"));
      if (n > 0) {
        star.setAttribute("aria-pressed", authored.rating !== null && authored.rating >= n ? "true" : "false");
      }
    }
    const albums = requireElement(root, "[data-albums]", HTMLElement);
    albums.replaceChildren(
      ...authored.collections.map((held) => {
        const link = document.createElement("a");
        link.href = `/t/${held.slug}`;
        link.textContent = held.name;
        return link;
      })
    );
  };
  var mounted = () => [findElement(document, "[data-lightbox]", HTMLElement), findElement(document, "[data-grid]", HTMLElement)].filter(
    (one) => one !== null
  );
  var asked2 = (qs) => {
    const question = new URLSearchParams(qs ?? "");
    const one = (name) => question.get(name);
    const counted = (name) => {
      const held = question.get(name);
      return held === null ? null : Number(held);
    };
    return {
      folder: one("folder"),
      album: one("album"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted("rating_min"),
      q: one("q"),
      f: question.getAll("f"),
      sort: one("sort"),
      size: counted("size")
    };
  };
  var settle = async (root) => {
    const surfaces = mounted();
    if (!surfaces.length) return;
    const { data, error } = await api.GET("/g/locate/{slug}", {
      params: { path: { slug: requireData(root, "slug") }, query: asked2(root.dataset.qs) }
    });
    if (error || !data) {
      window.location.reload();
      return;
    }
    const held = surfaces[0]?.dataset.answer ?? "";
    if (!data.in_answer || held && data.answer !== held) {
      window.location.reload();
      return;
    }
    for (const surface of surfaces) {
      surface.dataset.currency = data.currency;
      surface.dataset.answer = data.answer;
    }
  };
  var applied = async (root, told) => {
    if (!told.ok) {
      window.alert(told.refusal);
      return;
    }
    draw(root, told.data.authored);
    await settle(root);
  };
  var setFavorite = async (root, value) => {
    const told = await api.POST("/i/{slug}/favorite", {
      params: { path: { slug: requireData(root, "slug") } },
      body: { value }
    });
    await applied(root, answered(told, "the favorite could not be recorded"));
  };
  var setRating = async (root, value) => {
    const told = await api.POST("/i/{slug}/rating", {
      params: { path: { slug: requireData(root, "slug") } },
      body: { value }
    });
    await applied(root, answered(told, "the rating could not be recorded"));
  };
  var setMembership = async (root, collection, value) => {
    const told = await api.POST("/i/{slug}/collections/{collection}", {
      params: { path: { slug: requireData(root, "slug"), collection } },
      body: { value }
    });
    await applied(root, answered(told, "the album membership could not be recorded"));
  };
  var choices = async (root) => {
    const box = requireElement(root, "[data-album-choices]", HTMLElement);
    if (!box.hidden) {
      box.hidden = true;
      return;
    }
    const told = answered(
      await api.GET("/i/{slug}/collection-choices", { params: { path: { slug: requireData(root, "slug") } } }),
      "the albums could not be read"
    );
    if (!told.ok) {
      window.alert(told.refusal);
      return;
    }
    const data = told.data;
    box.replaceChildren(
      ...data.map((one) => {
        const row = document.createElement("label");
        const tick = document.createElement("input");
        tick.type = "checkbox";
        tick.checked = one.filed;
        tick.addEventListener("change", () => {
          void setMembership(root, one.slug, tick.checked);
        });
        row.append(tick, ` ${one.name}`);
        return row;
      })
    );
    if (!data.length) box.textContent = "no albums yet \u2014 make one on /albums";
    box.hidden = false;
  };
  var pressed = (root) => requireElement(root, "[data-fav]", HTMLElement).getAttribute("aria-pressed") === "true";
  document.addEventListener("click", (event) => {
    const root = closestFrom(event.target, "[data-authored]", HTMLElement);
    if (!root) return;
    if (closestFrom(event.target, "[data-fav]", HTMLElement)) {
      void setFavorite(root, !pressed(root));
      return;
    }
    const star = closestFrom(event.target, "[data-rate]", HTMLElement);
    if (star) {
      const n = Number(requireData(star, "rate"));
      void setRating(root, n > 0 ? n : null);
      return;
    }
    if (closestFrom(event.target, "[data-album-picker]", HTMLElement)) void choices(root);
  });
  var strip = () => findElement(document, "[data-lightbox] [data-authored]", HTMLElement) ?? findElement(document, "[data-authored]", HTMLElement);
  register([
    {
      key: "f",
      by: "authored: favorite",
      run: () => {
        const root = strip();
        if (root) void setFavorite(root, !pressed(root));
      }
    },
    {
      key: "a",
      by: "authored: albums",
      run: () => {
        const root = strip();
        if (root) void choices(root);
      }
    },
    ...[1, 2, 3, 4, 5].map((stars) => ({
      key: String(stars),
      by: `authored: ${stars} star${stars === 1 ? "" : "s"}`,
      run: () => {
        const root = strip();
        if (root) void setRating(root, stars);
      }
    })),
    {
      key: "0",
      by: "authored: clear rating",
      run: () => {
        const root = strip();
        if (root) void setRating(root, null);
      }
    }
  ]);

  // src/selection.ts
  var asked3 = () => {
    const question = new URLSearchParams(window.location.search);
    const one = (name) => question.get(name);
    const counted = (name) => {
      const held = question.get(name);
      return held === null ? null : Number(held);
    };
    return {
      folder: one("folder"),
      album: one("album"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted("rating_min"),
      q: one("q"),
      sort: one("sort"),
      size: counted("size"),
      f: question.getAll("f")
    };
  };
  (() => {
    const bar = findElement(document, "[data-curate]", HTMLElement);
    if (!bar) return;
    const count = requireElement(bar, "[data-curate-count]", HTMLElement);
    const albums = requireElement(bar, "[data-bulk-album]", HTMLSelectElement);
    const grid = () => findElement(document, "[data-grid]", HTMLElement);
    let answer = "";
    const selected = /* @__PURE__ */ new Set();
    const draw2 = () => {
      bar.hidden = selected.size === 0;
      count.textContent = `${selected.size} selected`;
      for (const shell of document.querySelectorAll("[data-selection-key]")) {
        const pick = findElement(shell, "[data-pick]", HTMLInputElement);
        if (pick) pick.checked = selected.has(requireData(shell, "selectionKey"));
      }
    };
    const sync = () => {
      const mounted2 = grid();
      if (!mounted2) return;
      const held = requireData(mounted2, "answer");
      if (answer !== held) {
        answer = held;
        selected.clear();
      }
      draw2();
    };
    const shelve = async () => {
      const { data, error } = await api.GET("/albums", { headers: { accept: "application/json" } });
      if (error || !data) return;
      albums.replaceChildren(
        ...data.filter((one) => one.kind !== "smart").map((one) => {
          const choice = document.createElement("option");
          choice.value = one.slug;
          choice.textContent = one.name;
          return choice;
        })
      );
    };
    const settle2 = (told2) => {
      if (!told2.ok) {
        window.alert(told2.refusal);
        return;
      }
      const mounted2 = grid();
      if (told2.data.after.answer !== answer) {
        window.location.reload();
        return;
      }
      if (mounted2) {
        mounted2.dataset.currency = told2.data.after.currency;
        mounted2.dataset.answer = told2.data.after.answer;
      }
      draw2();
    };
    const told = (result) => {
      if (result.response.status === 409) {
        window.location.reload();
        return;
      }
      settle2(answered(result, refusal(result.error, "the selection could not be curated")));
    };
    const items = () => [...selected];
    const favorite = async (value) => {
      told(
        await api.POST("/g/selection/favorite", { params: { query: asked3() }, body: { answer, items: items(), value } })
      );
    };
    const rate = async (value) => {
      told(
        await api.POST("/g/selection/rating", { params: { query: asked3() }, body: { answer, items: items(), value } })
      );
    };
    const file = async (collection, value) => {
      told(
        await api.POST("/g/selection/collections/{collection}", {
          params: { path: { collection }, query: asked3() },
          body: { answer, items: items(), value }
        })
      );
    };
    const place = async (name, kind) => {
      told(
        await api.POST("/g/selection/place", {
          params: { query: asked3() },
          body: { answer, items: items(), name, kind, within_kind: "country" }
        })
      );
    };
    document.addEventListener("change", (event) => {
      const pick = closestFrom(event.target, "[data-pick]", HTMLInputElement);
      if (!pick) return;
      const shell = pick.closest("[data-selection-key]");
      if (!shell) return;
      const key = requireData(shell, "selectionKey");
      if (pick.checked) selected.add(key);
      else selected.delete(key);
      if (selected.size === 1 && !albums.options.length) void shelve();
      draw2();
    });
    bar.addEventListener("click", (event) => {
      const flag = closestFrom(event.target, "[data-bulk-favorite]", HTMLElement);
      if (flag) {
        void favorite(requireData(flag, "bulkFavorite") === "1");
        return;
      }
      const stars = closestFrom(event.target, "[data-bulk-rate]", HTMLElement);
      if (stars) {
        const n = Number(requireData(stars, "bulkRate"));
        void rate(n > 0 ? n : null);
        return;
      }
      const filed = closestFrom(event.target, "[data-bulk-file]", HTMLElement);
      if (filed && albums.value) {
        void file(albums.value, requireData(filed, "bulkFile") === "1");
        return;
      }
      const placed = closestFrom(event.target, "[data-bulk-place]", HTMLElement);
      if (placed) {
        const kind = asPlaceKind(requireElement(bar, "[data-bulk-place-kind]", HTMLSelectElement).value);
        if (requireData(placed, "bulkPlace") !== "1") {
          void place(null, kind);
          return;
        }
        const name = requireElement(bar, "[data-bulk-place-name]", HTMLInputElement).value.trim();
        if (!name) return;
        void place(name, kind);
        return;
      }
      if (closestFrom(event.target, "[data-curate-clear]", HTMLElement)) {
        selected.clear();
        draw2();
      }
    });
    document.body.addEventListener("htmx:afterSwap", sync);
    sync();
  })();
  function asPlaceKind(held) {
    const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"];
    const found = known.find((one) => one === held);
    if (found === void 0) throw new Error(`the place picker offered ${held}, which is not a place kind`);
    return found;
  }
})();
//# sourceMappingURL=gallery.js.map
