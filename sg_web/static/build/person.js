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
  function isPlainClick(event, link) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && link?.getAttribute("target") !== "_blank";
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

  // src/recipe.ts
  function fit(field) {
    if (field.clientWidth === 0) return;
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
  }
  async function copied(button2, text) {
    const was = button2.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button2.textContent = "copied";
      button2.dataset.done = "true";
    } catch {
      button2.textContent = "cannot copy";
    }
    setTimeout(() => {
      button2.textContent = was;
      delete button2.dataset.done;
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
  var DEFAULT_EVERY = 5;
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
    const canPan = () => {
      const size = fitted();
      const box = stageBox.getBoundingClientRect();
      return size.width * look.scale > box.width + 1 || size.height * look.scale > box.height + 1;
    };
    const zoomedIn = () => look.scale > 1 || canPan();
    const paint = () => {
      if (!media) return;
      media.style.transform = `translate(${look.x}px, ${look.y}px) scale(${look.scale})`;
      stageBox.dataset.framing = look.framing;
      root.dataset.absorbed = look.scale > ABSORBED ? "true" : "false";
      root.dataset.zoom = String(Math.round(look.scale * 100));
      root.dataset.magnified = zoomedIn() ? "true" : "false";
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
      for (const button2 of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
        button2.setAttribute("aria-expanded", String(open));
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
      const step = stepTo(rolled > 0 ? "next" : "previous");
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
        frame(zoomedIn() ? "fit" : "actual");
      });
      let dragging = null;
      let from = { x: 0, y: 0, ox: 0, oy: 0 };
      onElement(stageBox, "pointerdown", (event) => {
        if (event.button !== 0 || !canPan()) return;
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
    const stepTo = (wanted) => findElement(root, `[data-nav="${wanted}"]`, HTMLAnchorElement);
    const stepping = (wanted) => () => {
      const step = stepTo(wanted);
      if (step) walk(step.href);
    };
    const middle = (by) => () => zoomAbout(by, window.innerWidth / 2, window.innerHeight / 2);
    bound.push(
      register([
        { key: "z", by: "viewer: fit/actual", run: () => frame(zoomedIn() ? "fit" : "actual") },
        { key: "l", by: "viewer: focus", run: focus },
        { key: "i", by: "viewer: inspector", run: () => showInspector(root.dataset.inspector !== "open") },
        { key: "+", by: "viewer: zoom in", run: middle(1.3) },
        { key: "=", by: "viewer: zoom in", run: middle(1.3) },
        { key: "-", by: "viewer: zoom out", run: middle(1 / 1.3) },
        { key: "ArrowRight", by: "viewer: next", run: stepping("next") },
        { key: "ArrowLeft", by: "viewer: previous", run: stepping("previous") },
        { key: "s", by: "viewer: slideshow", run: () => playing(!isPlaying()) }
      ])
    );
    onDocument("pointermove", wake);
    const shown = findElement(root, "[data-slideshow]", HTMLElement);
    let ticking = 0;
    const isPlaying = () => workspace().showPlaying === true;
    const every = () => {
      const held = workspace().showEvery;
      return typeof held === "number" && held > 0 ? held : DEFAULT_EVERY;
    };
    const wrapping = (on) => {
      root.dataset.wrap = on ? "on" : "off";
      for (const arrow of everyElement(root, "[data-nav-wrap]", HTMLAnchorElement)) {
        if (on) arrow.dataset.nav = requireData(arrow, "navWrap");
        else delete arrow.dataset.nav;
      }
    };
    const advance = () => {
      const step = findElement(root, '[data-nav="next"]:not([data-nav-wrap])', HTMLAnchorElement);
      if (step) {
        walk(step.href);
        return;
      }
      const around = findElement(root, '[data-nav-wrap="next"]', HTMLAnchorElement);
      if (workspace().loop === true && around) {
        walk(around.href);
        return;
      }
      playing(false);
    };
    const playing = (on) => {
      window.clearTimeout(ticking);
      remember({ showPlaying: on });
      if (shown) {
        shown.dataset.playing = on ? "yes" : "no";
        const control = findElement(shown, "[data-show-play]", HTMLElement);
        if (control) {
          control.setAttribute("aria-pressed", on ? "true" : "false");
          control.setAttribute("aria-label", on ? "pause the slideshow" : "start the slideshow");
          control.textContent = on ? "\u23F8" : "\u25B6";
        }
      }
      if (on && shown) ticking = window.setTimeout(advance, every() * 1e3);
    };
    if (shown) {
      const settings = findElement(shown, "[data-show-settings]", HTMLElement);
      const opener = findElement(shown, "[data-show-settings-toggle]", HTMLElement);
      const interval = findElement(shown, "[data-show-every]", HTMLSelectElement);
      const wraps = findElement(shown, "[data-show-wrap]", HTMLInputElement);
      const loops = findElement(shown, "[data-show-loop]", HTMLInputElement);
      const kept2 = workspace();
      if (interval) interval.value = String(every());
      if (wraps) wraps.checked = kept2.wrap === true;
      if (loops) loops.checked = kept2.loop === true;
      wrapping(kept2.wrap === true);
      if (opener && settings) {
        onElement(opener, "click", () => {
          settings.hidden = !settings.hidden;
          opener.setAttribute("aria-expanded", settings.hidden ? "false" : "true");
        });
      }
      const play = findElement(shown, "[data-show-play]", HTMLElement);
      if (play) onElement(play, "click", () => playing(!isPlaying()));
      if (interval) {
        onElement(interval, "change", () => {
          remember({ showEvery: Number(interval.value) });
          if (isPlaying()) playing(true);
        });
      }
      if (wraps) {
        onElement(wraps, "change", () => {
          remember({ wrap: wraps.checked });
          wrapping(wraps.checked);
        });
      }
      if (loops) onElement(loops, "change", () => remember({ loop: loops.checked }));
      playing(isPlaying());
    }
    const strip = findElement(root, "[data-filmstrip-track]", HTMLElement);
    if (strip) {
      const here = findElement(strip, "[data-filmstrip-item][aria-current='true']", HTMLElement);
      here?.scrollIntoView({ block: "nearest", inline: "center" });
      onElement(strip, "click", (event) => {
        const near = closestFrom(event.target, "[data-filmstrip-item]", HTMLAnchorElement);
        if (!near || !isPlainClick(event, near)) return;
        event.preventDefault();
        walk(near.href);
      });
    }
    for (const button2 of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
      onElement(button2, "click", () => showInspector(root.dataset.inspector !== "open"));
    }
    for (const button2 of everyElement(root, "[data-panel-open]", HTMLElement)) {
      onElement(button2, "click", () => {
        showInspector(true);
        panel(requireData(button2, "panelOpen"));
      });
    }
    for (const button2 of everyElement(root, "[data-focus]", HTMLElement)) {
      onElement(button2, "click", focus);
    }
    if (inspector) {
      const kept2 = workspace();
      const generated = inspector.querySelector("[data-panel='creation']") !== null && root.dataset.made === "generated";
      showInspector(kept2.inspector ? kept2.inspector === "open" : false, false);
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
        if (isPlaying()) {
          playing(false);
          return true;
        }
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
        window.clearTimeout(ticking);
        for (const off of bound) off();
        bound.length = 0;
      }
    };
  }

  // src/media.ts
  var pad = (n) => String(n).padStart(2, "0");
  var asPlaceKind = (held) => {
    const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"];
    const found = known.find((one) => one === held);
    if (found === void 0) throw new Error(`the place picker offered ${held}, which is not a place kind`);
    return found;
  };
  (() => {
    for (const node of everyElement(document, "time[data-epoch]", HTMLTimeElement)) {
      const d = new Date(Number(requireData(node, "epoch")) * 1e3);
      const z = node.dataset.domain === "instant" ? "Z" : " wall";
      node.textContent = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}${z}`;
    }
    const video = findElement(document, "video", HTMLVideoElement);
    for (const at of everyElement(document, "[data-said-seek]", HTMLElement)) {
      at.addEventListener("click", () => {
        if (!video) return;
        video.currentTime = Number(requireData(at, "saidSeek")) / 1e3;
        void video.play();
      });
    }
    const judged = findElement(document, "[data-viewer]", HTMLElement);
    if (judged) {
      const slug = requireData(judged, "slug");
      for (const box of everyElement(document, "[data-said-judge]", HTMLElement)) {
        const line = box.closest("[data-said-kind]");
        if (!(line instanceof HTMLElement)) continue;
        for (const thumb of everyElement(box, "[data-said-verdict-set]", HTMLElement)) {
          thumb.addEventListener("click", async () => {
            const wanted = requireData(thumb, "saidVerdictSet");
            const held = line.dataset.saidVerdict;
            const { data, error } = await api.POST("/i/{slug}/said/verdict", {
              params: { path: { slug } },
              body: {
                kind: requireData(line, "saidKind"),
                model_id: requireData(line, "saidModel"),
                model_version: requireData(line, "saidVersion"),
                verdict: held === wanted ? null : wanted
              }
            });
            if (!data) {
              await say(refusal(error, "that verdict was not recorded"));
              return;
            }
            if (data.verdict) line.dataset.saidVerdict = data.verdict;
            else delete line.dataset.saidVerdict;
            for (const one of everyElement(box, "[data-said-verdict-set]", HTMLElement)) {
              one.setAttribute("aria-pressed", String(one.dataset.saidVerdictSet === data.verdict));
            }
          });
        }
      }
    }
    const placeForm = findElement(document, "[data-place-form]", HTMLFormElement);
    if (placeForm) {
      const slug = requireData(placeForm, "slug");
      const value = (name) => requireElement(placeForm, `[name="${name}"]`, HTMLInputElement).value.trim();
      const chosen = (name) => requireElement(placeForm, `[name="${name}"]`, HTMLSelectElement).value;
      const record = async (body) => {
        const { data, error } = await api.POST("/i/{slug}/place", { params: { path: { slug } }, body });
        if (!data) {
          await say(refusal(error, "the place could not be recorded"));
          return;
        }
        window.location.reload();
      };
      placeForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const name = value("name");
        if (!name) return;
        const within = value("within");
        void record({
          name,
          kind: asPlaceKind(chosen("kind")),
          within: within || null,
          within_kind: asPlaceKind(chosen("within_kind"))
        });
      });
      findElement(placeForm, "[data-place-clear]", HTMLElement)?.addEventListener("click", () => {
        void record({ name: null, kind: "locality", within: null, within_kind: "country" });
      });
    }
    const mounted = findElement(document, "[data-viewer]", HTMLElement);
    const viewer = mounted ? mountViewer(mounted, (href) => {
      window.location.assign(href);
    }) : null;
    const back = findElement(document, "[data-return]", HTMLAnchorElement);
    if (!back) return;
    const leave = () => {
      window.location.assign(back.href);
    };
    for (const close of everyElement(document, "[data-close]", HTMLElement)) {
      close.addEventListener("click", leave);
    }
    register([
      {
        key: "Escape",
        by: "media page: leave",
        run: () => {
          if (viewer?.unwind()) return;
          leave();
        }
      }
    ]);
  })();

  // src/overlay.ts
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

  // src/people.ts
  var pad2 = (n) => String(n).padStart(2, "0");
  var spellDays = (root) => {
    for (const node of everyElement(root, "time[data-epoch]:not([data-spelled])", HTMLTimeElement)) {
      const d = new Date(Number(requireData(node, "epoch")) * 1e3);
      node.textContent = `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
      node.dataset.spelled = "";
    }
  };
  (() => {
    spellDays(document);
    new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });
    document.addEventListener("submit", async (event) => {
      const form = closestFrom(event.target, "[data-rename]", HTMLFormElement);
      if (!form) return;
      event.preventDefault();
      const slug = requireData(form, "rename");
      const name = requireElement(form, '[name="name"]', HTMLInputElement).value;
      const { data, error } = await api.POST("/p/{slug}/name", { params: { path: { slug } }, body: { name } });
      if (!data) {
        await say(refusal(error, "that name was refused"));
        return;
      }
      window.location.replace(`/p/${data.slug}`);
    });
    addressableOverlay({
      root: "[data-drawer-root]",
      trigger: "[data-person]",
      pathPrefix: "/p/"
    });
  })();

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
    const said = document.createElement("span");
    said.className = "compare-view-said";
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
    bar.append(said, modes, close);
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
      const label = document.createElement("figcaption");
      const named = document.createElement("b");
      named.className = "compare-letter";
      named.textContent = letter(at2);
      const link = document.createElement("a");
      link.href = `/i/${one.slug}`;
      link.textContent = one.name;
      label.append(named, link);
      column.append(frame, label);
      strip.append(column);
    }
    sheet.append(bar, strip);
    document.body.append(sheet);
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
    drawTray(tray);
  }

  // src/compare-mount.ts
  mountCompare(document.body);
})();
//# sourceMappingURL=person.js.map
