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
    const held2 = node.dataset[key];
    if (held2 === void 0) {
      throw new Error(`expected a data-${key} on ${node.tagName.toLowerCase()}`);
    }
    return held2;
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
  var framed = (question2, submit, dismiss, said3) => ({
    question: question2,
    submit: said3.submit !== void 0 ? said3.submit : submit,
    dismiss: said3.dismiss !== void 0 ? said3.dismiss : dismiss,
    ...said3.detail !== void 0 ? { detail: said3.detail } : {},
    ...said3.grave !== void 0 ? { grave: said3.grave } : {}
  });
  var button = (words, value, kind) => {
    const control = document.createElement("button");
    control.value = value;
    control.className = kind;
    control.textContent = words;
    return control;
  };
  async function ask(asked4, build2) {
    const box = document.createElement("dialog");
    box.className = "ask-box";
    box.innerHTML = `<form method="dialog" class="ask-form">
      <h2 class="ask-question"></h2>
      <p class="ask-detail" hidden></p>
      <div class="ask-body"></div>
      <div class="ask-feet"></div>
    </form>`;
    requireElement(box, ".ask-question", HTMLElement).textContent = asked4.question;
    if (asked4.detail !== void 0) {
      const line = requireElement(box, ".ask-detail", HTMLElement);
      line.textContent = asked4.detail;
      line.hidden = false;
    }
    const read2 = build2(requireElement(box, ".ask-body", HTMLElement), box);
    const feet = requireElement(box, ".ask-feet", HTMLElement);
    if (asked4.submit !== null) {
      feet.append(button(asked4.submit, TAKEN, asked4.grave === true ? "ask-take is-grave" : "ask-take"));
    }
    if (asked4.dismiss !== null) feet.append(button(asked4.dismiss, DISMISSED, "ask-drop"));
    box.addEventListener("click", (event) => {
      const at = box.getBoundingClientRect();
      const inside = event.clientX >= at.left && event.clientX <= at.right && event.clientY >= at.top && event.clientY <= at.bottom;
      if (!inside && event.detail > 0) box.close(DISMISSED);
    });
    const answer = new Promise((settle2) => {
      box.addEventListener(
        "close",
        () => {
          const taken = box.returnValue !== DISMISSED ? read2() : null;
          box.remove();
          settle2(taken);
        },
        { once: true }
      );
    });
    document.body.append(box);
    box.showModal();
    return answer;
  }
  async function panel(title, fill2, dismiss = "close") {
    await ask({ question: title, submit: null, dismiss }, (body) => {
      fill2(body);
      return () => void 0;
    });
  }
  async function say(message, framing = {}) {
    await ask(framed(message, "ok", null, framing), () => () => void 0);
  }
  async function askYesNo(question2, framing = {}) {
    return await ask(framed(question2, "yes", "no", framing), () => () => true) === true;
  }
  async function askText(question2, typed = {}) {
    const said3 = await ask(framed(question2, "save", "cancel", typed), (body) => {
      const field = document.createElement("input");
      field.type = "text";
      field.className = "ask-field";
      field.value = typed.value ?? "";
      field.placeholder = typed.placeholder ?? "";
      field.autofocus = true;
      field.setAttribute("aria-label", typed.label ?? question2);
      body.append(field);
      return () => field.value.trim();
    });
    return said3 ? said3 : null;
  }
  async function askChoice(question2, choices2, framing = {}) {
    if (choices2.length === 0) return null;
    return ask(
      // No affirmative in the feet: the choices are the affirmative.
      framed(question2, null, "cancel", { ...framing, submit: null }),
      (body, box) => {
        const list = document.createElement("div");
        list.className = "ask-choices";
        for (const [index, one] of choices2.entries()) {
          const control = button(one.label, one.value, "ask-choice");
          control.autofocus = index === 0;
          if (one.note !== void 0) {
            const note = document.createElement("span");
            note.className = "ask-choice-note";
            note.textContent = one.note;
            control.append(note);
          }
          list.append(control);
        }
        body.append(list);
        return () => box.returnValue;
      }
    );
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
    const held2 = new FormData(form).get(name);
    return typeof held2 === "string" && held2.trim() ? held2 : null;
  };
  var asListedKind = (held2) => {
    if (held2 !== "album" && held2 !== "flag") {
      throw new Error(`data-convert offered ${held2}, which is not a listed kind`);
    }
    return held2;
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

  // src/diff.ts
  var MOVED = 8;
  var SAMPLE = 512;
  var load = (src) => new Promise((ok, no) => {
    const img = new Image();
    img.decoding = "async";
    img.addEventListener("load", () => ok(img));
    img.addEventListener("error", () => no(new Error(`could not load ${src}`)));
    img.src = src;
  });
  function drawn(img, w, h) {
    const board2 = document.createElement("canvas");
    board2.width = w;
    board2.height = h;
    const hand = board2.getContext("2d", { willReadFrequently: true });
    if (!hand) throw new Error("no 2d context");
    const scale = Math.max(w / img.naturalWidth, h / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    hand.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
    return hand.getImageData(0, 0, w, h);
  }
  async function difference(a, b) {
    const [one, two] = await Promise.all([load(a), load(b)]);
    const ratio = one.naturalWidth / one.naturalHeight;
    const w = Math.max(1, Math.round(ratio >= 1 ? SAMPLE : SAMPLE * ratio));
    const h = Math.max(1, Math.round(ratio >= 1 ? SAMPLE / ratio : SAMPLE));
    const left = drawn(one, w, h);
    const right = drawn(two, w, h);
    const heat = new ImageData(w, h);
    let moved = 0;
    let worst = 0;
    let total = 0;
    const one8 = left.data;
    const two8 = right.data;
    for (let i = 0; i < one8.length; i += 4) {
      const dr = Math.abs((one8[i] ?? 0) - (two8[i] ?? 0));
      const dg = Math.abs((one8[i + 1] ?? 0) - (two8[i + 1] ?? 0));
      const db = Math.abs((one8[i + 2] ?? 0) - (two8[i + 2] ?? 0));
      const delta = Math.max(dr, dg, db);
      total += delta;
      if (delta > worst) worst = delta;
      if (delta < MOVED) continue;
      moved += 1;
      const t = Math.min(1, (delta - MOVED) / (255 - MOVED));
      heat.data[i] = Math.round(183 + (240 - 183) * t);
      heat.data[i + 1] = Math.round(156 + (145 - 156) * t);
      heat.data[i + 2] = Math.round(255 + (60 - 255) * t);
      heat.data[i + 3] = Math.round(90 + 165 * t);
    }
    const pixels2 = one8.length / 4;
    return { moved: moved / pixels2, worst, mean: total / pixels2, heat };
  }
  function paint(board2, heat) {
    board2.width = heat.width;
    board2.height = heat.height;
    const hand = board2.getContext("2d");
    if (!hand) return;
    hand.putImageData(heat, 0, 0);
  }
  function said2(found) {
    if (found.moved < 5e-4) return "identical to the eye \u2014 nothing moved";
    const part = found.moved < 0.01 ? `${(found.moved * 100).toFixed(2)}%` : `${Math.round(found.moved * 100)}%`;
    const how = found.worst > 160 ? "heavily" : found.worst > 60 ? "noticeably" : "slightly";
    return `${part} of the frame changed, ${how} (worst ${found.worst} of 255)`;
  }

  // src/dupes.ts
  (() => {
    const groups = document.querySelector("[data-dupe-groups]");
    if (!(groups instanceof HTMLElement)) return;
    groups.addEventListener("click", (event) => {
      const pick = closestFrom(event.target, "[data-dupe-pick]", HTMLButtonElement);
      if (!pick) return;
      const group = pick.closest("[data-dupe-group]");
      if (!(group instanceof HTMLElement)) return;
      const shown = group.querySelector("[data-dupe-shown]");
      const open = group.querySelector("[data-dupe-open]");
      const title = group.querySelector("[data-dupe-title]");
      if (shown instanceof HTMLImageElement) {
        shown.src = requireData(pick, "thumb");
        shown.alt = requireData(pick, "name");
      }
      if (open instanceof HTMLAnchorElement) open.href = requireData(pick, "href");
      if (title instanceof HTMLElement) title.textContent = requireData(pick, "name");
      for (const other of group.querySelectorAll("[data-dupe-pick]")) {
        other.setAttribute("aria-pressed", String(other === pick));
      }
      void compare(group, pick);
    });
    groups.addEventListener("click", (event) => {
      const mode = closestFrom(event.target, "[data-dupe-mode]", HTMLButtonElement);
      if (!mode) return;
      const group = mode.closest("[data-dupe-group]");
      if (!(group instanceof HTMLElement)) return;
      const canvas = group.querySelector("[data-dupe-canvas]");
      if (!(canvas instanceof HTMLElement)) return;
      const wanted = requireData(mode, "dupeMode");
      canvas.dataset.mode = wanted;
      for (const other of group.querySelectorAll("[data-dupe-mode]")) {
        other.setAttribute("aria-pressed", String(other === mode));
      }
    });
    async function compare(group, pick) {
      const canvas = group.querySelector("[data-dupe-canvas]");
      const heat = group.querySelector("[data-dupe-heat]");
      const measure = group.querySelector("[data-dupe-measure]");
      if (!(canvas instanceof HTMLElement) || !(heat instanceof HTMLCanvasElement)) return;
      if (!(measure instanceof HTMLElement)) return;
      const readout = group.querySelector("[data-dupe-readout]");
      const figure = group.querySelector("[data-dupe-figure]");
      if (!(readout instanceof HTMLElement) || !(figure instanceof HTMLElement)) return;
      const best = requireData(canvas, "best");
      const shown = requireData(pick, "thumb");
      if (shown === best) {
        canvas.dataset.mode = "photo";
        canvas.dataset.same = "";
        readout.hidden = false;
        figure.textContent = "\u2605";
        measure.textContent = "the copy every other one is measured against";
        for (const m of group.querySelectorAll("[data-dupe-mode]")) {
          m.setAttribute("aria-pressed", String(requireData(m, "dupeMode") === "photo"));
        }
        return;
      }
      delete canvas.dataset.same;
      readout.hidden = false;
      figure.textContent = "\u2026";
      measure.textContent = "measuring";
      try {
        const found = await difference(best, shown);
        paint(heat, found.heat);
        figure.textContent = found.moved < 5e-4 ? "0%" : `${(found.moved * 100).toFixed(found.moved < 0.01 ? 2 : 0)}%`;
        measure.textContent = said2(found);
      } catch (why) {
        figure.textContent = "\u2014";
        measure.textContent = why instanceof Error ? why.message : "could not compare these";
      }
    }
    groups.addEventListener("click", async (event) => {
      const button2 = closestFrom(event.target, "[data-not-a-duplicate]", HTMLButtonElement);
      if (!button2) return;
      const one = requireData(button2, "notADuplicate");
      const other = requireData(button2, "against");
      button2.disabled = true;
      const held2 = await api.POST("/dupes/{slug}/not-a-duplicate", {
        params: { path: { slug: one } },
        body: { other }
      });
      if (held2.error) {
        button2.disabled = false;
        await say(refusal(held2.error, "that was not recorded"));
        return;
      }
      const member = button2.closest("[data-dupe-member]");
      if (member instanceof HTMLElement) member.remove();
    });
  })();

  // src/evolution.ts
  var SEQUENCE_FACTS = ["model", "loras", "sampler", "seed"];
  var COMPARE_FACTS = [
    "model",
    "loras",
    "sampler",
    "steps",
    "cfg",
    "seed",
    "scheduler",
    "width",
    "height"
  ];
  var NOUNS = { capture_session: "photographs", file_session: "files" };
  var ENTITIES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  };
  var esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c] ?? c);
  var pct = (v) => v === null ? "\u2014" : `${Math.round(v * 100)}%`;
  function spell(value) {
    if (Array.isArray(value)) return esc(value.join(", "));
    return esc(value ?? "\u2014");
  }
  (() => {
    const here = findElement(document, "[data-evolution]", HTMLElement);
    if (!here) return;
    const root = here;
    const main = requireElement(root, "[data-main]", HTMLElement);
    const selectedPane = requireElement(root, "[data-selected]", HTMLElement);
    const inspector = requireElement(root, "[data-inspector]", HTMLElement);
    const planId = Number(requireData(root, "plan"));
    const space = new URLSearchParams(window.location.search).get("space");
    api.GET("/stories/plans/{plan_id}/evolution", {
      params: { path: { plan_id: planId }, query: space === null ? {} : { space } }
    }).then(({ data, error }) => {
      if (data === void 0) {
        main.textContent = refusal(error, "the plan's measurements could not be read");
        return;
      }
      explore(data);
    }, console.error);
    function explore(view) {
      const members = new Map(view.members.map((m) => [m.ref, m]));
      const transitionTo = new Map(view.transitions.map((t) => [t.after, t]));
      const noun = NOUNS[view.snapshot.subject] ?? "images";
      let selected = null;
      let pair = null;
      const member = (ref) => ref === null ? void 0 : members.get(ref);
      const thumb = (m, cls = "member") => {
        if (m === void 0) return "";
        if (m.media.thumbnail === null) {
          return `<span class="${cls}" data-ref="${esc(m.ref)}" title="${esc(m.media.name)} (file gone)"></span>`;
        }
        return `<img class="${cls}" data-ref="${esc(m.ref)}" src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}" title="${esc(m.ref)} \xB7 ${esc(m.media.name)}">`;
      };
      const clock = (wall) => new Date(wall * 1e3).toISOString().slice(11, 19);
      const when = (m) => {
        const o = m.occurrence;
        if (o === null || o.local_at === null) return "";
        let told = ` \xB7 ${esc(o.precision)} ${clock(o.local_at)} (${esc(o.basis)})`;
        if (o.estimated_at !== null) {
          const finish = o.finished_at === null ? "" : `finish ${clock(o.finished_at)} minus generation time`;
          told += ` <span class="chip chip-inferred" title="${esc(finish)}">\u2248 ${clock(o.estimated_at)} inferred</span>`;
        }
        if (o.conflicts.length) {
          told += ` <span class="chip chip-conflict" title="${esc(o.conflicts.join("; "))}">contested</span>`;
        }
        return told;
      };
      const metric = (label2, v, why) => `<dt>${label2}</dt><dd title="${esc(why)}">${pct(v)}${v === null ? ` <small>${esc(why)}</small>` : ""}</dd>`;
      function sequence() {
        const strip2 = view.phases.map(
          (p) => `<div class="phase" data-phase="${esc(p.id)}"><h3>${esc(p.label)}</h3><div class="members">${p.member_refs.map((r) => thumb(members.get(r))).join("")}</div></div>`
        ).join("");
        const generated = view.snapshot.subject === "generation_session";
        const prompts = [
          "prompt vs previous",
          (t) => t.prompt_cosine,
          (t) => t.prompt_cosine_unavailable ?? null
        ];
        const images = [
          "image vs previous",
          (t) => t.visual_cosine,
          (t) => t.visual_cosine_unavailable ?? null
        ];
        const rows = generated ? [prompts, images] : [images];
        const head = `<tr><th></th>${view.members.map((m) => `<th>${esc(m.ref.replace("member-", ""))}</th>`).join("")}</tr>`;
        const body = rows.map(([label2, get, why]) => {
          const cells = view.members.map((m, i) => {
            const t = transitionTo.get(m.ref);
            if (i === 0 || t === void 0) return "<td>\xB7</td>";
            const v = get(t);
            const cls = (t.phase_boundary ? "boundary " : "") + (v === null ? "unavailable" : "");
            return `<td class="${cls}" title="${esc(why(t))}">${pct(v)}</td>`;
          });
          return `<tr><th>${label2}</th>${cells.join("")}</tr>`;
        }).join("");
        const facts = (generated ? SEQUENCE_FACTS : []).map(
          (key) => `<tr><th>${key}</th>${view.members.map((m) => {
            const boundary = transitionTo.get(m.ref)?.phase_boundary ?? false;
            return `<td class="${boundary ? "boundary" : ""}">${spell(m.generation[key])}</td>`;
          }).join("")}</tr>`
        ).join("");
        main.innerHTML = `<div class="filmstrip">${strip2}</div><div class="tracks"><table>${head}${body}${facts}</table></div>`;
      }
      function drift() {
        const W = 420;
        const H = 320;
        const pad2 = 36;
        const drawn2 = view.transitions.filter(
          (t) => t.prompt_cosine !== null && t.visual_cosine !== null
        );
        const dots = drawn2.map((t) => {
          const x = pad2 + (1 - Math.max(0, t.prompt_cosine)) * (W - 2 * pad2);
          const y = H - pad2 - (1 - Math.max(0, t.visual_cosine)) * (H - 2 * pad2);
          return `<circle data-pair="${esc(t.before)}|${esc(t.after)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${t.phase_boundary ? "#fc6" : "#6cf"}"><title>${esc(t.before)} \u2192 ${esc(t.after)}: prompt ${pct(t.prompt_cosine)}, image ${pct(t.visual_cosine)}</title></circle>`;
        }).join("");
        const missing = view.transitions.length - drawn2.length;
        main.innerHTML = `<div class="drift"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <line x1="${pad2}" y1="${H - pad2}" x2="${W - pad2}" y2="${H - pad2}" stroke="#555"/><line x1="${pad2}" y1="${pad2}" x2="${pad2}" y2="${H - pad2}" stroke="#555"/>
      <text x="${W / 2}" y="${H - 8}" fill="#aaa" font-size="11" text-anchor="middle">prompt change from previous \u2192</text>
      <text x="12" y="${H / 2}" fill="#aaa" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">image change from previous \u2192</text>
      ${dots}</svg>
      <p class="evolution-provenance">each dot is one transition; yellow dots cross a plan phase boundary${missing ? `; ${missing} transition(s) lack a vector and are not drawn` : ""}</p></div>`;
      }
      function families() {
        main.innerHTML = `<div class="families">${view.phases.map(
          (p) => `<div class="family" data-phase="${esc(p.id)}"><h3>${esc(p.label)} <small>\xB7 ${p.member_refs.length}</small></h3>
        <p class="claims">${p.claims.map((c) => esc(c.kind)).join(" \xB7 ") || "no claims"}</p>
        <div class="members">${p.member_refs.map((r) => thumb(members.get(r), p.representative_refs.includes(r) ? "member hero" : "member")).join("")}</div></div>`
        ).join("")}</div>`;
      }
      function lineage() {
        if (!view.lineage.length) {
          main.innerHTML = `<p class="empty">no derivation edges among these ${noun}</p>`;
          return;
        }
        const children = /* @__PURE__ */ new Map();
        for (const e of view.lineage) {
          const held2 = children.get(e.parent);
          if (held2 === void 0) children.set(e.parent, [e.child]);
          else held2.push(e.child);
        }
        const kindOf = new Map(view.lineage.map((e) => [`${e.parent}|${e.child}`, e.kind]));
        const isChild = new Set(view.lineage.map((e) => e.child));
        const roots = [...new Set(view.lineage.map((e) => e.parent))].filter((p) => !isChild.has(p));
        const node = (ref, kind) => {
          const m = members.get(ref);
          const label2 = m === void 0 ? `<span class="kind">outside the session</span> ${esc(ref.slice(0, 8))}` : `${thumb(m)} ${esc(ref)}`;
          const kids = (children.get(ref) ?? []).map((child) => node(child, kindOf.get(`${ref}|${child}`) ?? null)).join("");
          return `<li>${label2}${kind === null ? "" : ` <span class="kind">${esc(kind)}</span>`}${kids ? `<ul>${kids}</ul>` : ""}</li>`;
        };
        main.innerHTML = `<div class="lineage"><ul>${roots.map((r) => node(r, null)).join("")}</ul></div>`;
      }
      function compare() {
        const A = pair === null ? void 0 : members.get(pair[0]);
        const B = pair === null ? void 0 : members.get(pair[1]);
        if (A === void 0 || B === void 0) {
          main.innerHTML = `<p class="empty">select two images (click one, then shift-click another)</p>`;
          return;
        }
        const t = view.transitions.find(
          (x) => x.before === A.ref && x.after === B.ref || x.before === B.ref && x.after === A.ref
        );
        const edge = view.lineage.find(
          (e) => e.parent === A.ref && e.child === B.ref || e.parent === B.ref && e.child === A.ref
        );
        const rows = COMPARE_FACTS.map((k) => {
          const va = A.generation[k];
          const vb = B.generation[k];
          const same = JSON.stringify(va) === JSON.stringify(vb);
          return `<tr><th>${k}</th><td>${same ? "same" : `${spell(va)} \u2192 ${spell(vb)}`}</td></tr>`;
        }).join("");
        main.innerHTML = `<div class="compare"><div>${thumb(A, "big")}<p>${esc(A.ref)} \xB7 ${esc(A.media.name)}</p></div><div>${thumb(B, "big")}<p>${esc(B.ref)} \xB7 ${esc(B.media.name)}</p></div>
      <div class="metrics" style="grid-column: 1 / -1"><dl>
        <dt>prompt cosine (consecutive only)</dt><dd>${t === void 0 ? "not consecutive" : pct(t.prompt_cosine)}</dd>
        <dt>visual cosine (consecutive only)</dt><dd>${t === void 0 ? "not consecutive" : pct(t.visual_cosine)}</dd>
        <dt>lineage</dt><dd>${edge === void 0 ? "no derivation edge" : `${esc(edge.parent)} \u2192 ${esc(edge.child)} (${esc(edge.kind)})`}</dd>
        <dt>prompt diff</dt><dd class="evolution-diff">${diffTokens(A.prompt.effective?.main ?? "", B.prompt.effective?.main ?? "")}</dd>
      </dl><table class="tracks">${rows}</table></div></div>`;
      }
      function panel2() {
        const m = member(selected);
        if (m === void 0) {
          selectedPane.innerHTML = `<p class="empty">select an image</p>`;
          return;
        }
        const eff = m.prompt.effective;
        const org = m.prompt.original;
        const links = [];
        if (m.media.page !== null) links.push(`<a href="${esc(m.media.page)}">open image</a>`);
        if (eff !== null) {
          links.push(`<a href="${esc(view.links.search)}${encodeURIComponent(eff.main)}">images like this prompt</a>`);
        }
        if (eff !== null && eff.prompt_id !== null && view.semantic.provider !== null) {
          links.push(
            `<a href="/prompts/${eff.prompt_id}/neighbours?space=${encodeURIComponent(view.semantic.provider)}">prompts like this</a>`
          );
        }
        if (view.links.gallery_day !== null) {
          links.push(`<a href="${esc(view.links.gallery_day)}">this day in the gallery</a>`);
        }
        selectedPane.innerHTML = `${m.media.thumbnail === null ? "" : `<img src="${esc(m.media.thumbnail)}" alt="${esc(m.media.name)}">`}
      <h2>${esc(m.ref)} \xB7 ${esc(m.media.name)}</h2>
      <p class="evolution-provenance">${esc(m.phase_ref)}${when(m)}</p>
      <h3>effective prompt</h3><pre>${esc(eff === null ? "\u2014 not frozen \u2014" : eff.text)}</pre>
      ${org === null ? `<p class="evolution-provenance">no original prompt was recorded by the generator</p>` : `<h3>as written</h3><pre>${esc(org.text)}</pre><h3>written \u2192 ran</h3><p class="evolution-diff">${diffTokens(org.main, eff?.main ?? "")}</p>`}
      <p class="links">${links.join("")}</p>`;
      }
      function inspect() {
        const m = member(selected);
        if (m === void 0) {
          inspector.innerHTML = "";
          return;
        }
        const t = transitionTo.get(m.ref);
        let html = `<dl class="metrics">${metric(
          "written \u2192 ran",
          m.metrics.original_effective_cosine,
          m.metrics.original_effective_cosine_unavailable ?? null
        )}${metric("prompt \u2194 image", m.metrics.text_image_cosine, m.metrics.text_image_cosine_unavailable ?? null)}</dl>`;
        if (t !== void 0) {
          const rows = [
            ...t.changes.parameters.map(
              (one) => `<tr><th>${esc(one.name)}</th><td>${spell(one.before)} \u2192 ${spell(one.after)}</td></tr>`
            ),
            ...t.changes.loras_added.length ? [`<tr><th>loras_added</th><td>${esc(t.changes.loras_added.join(", "))}</td></tr>`] : [],
            ...t.changes.loras_removed.length ? [`<tr><th>loras_removed</th><td>${esc(t.changes.loras_removed.join(", "))}</td></tr>`] : []
          ].join("");
          html += `<h3>from ${esc(t.before)}${t.phase_boundary ? " \xB7 phase boundary" : ""}</h3><dl class="metrics">${metric(
            "prompt similarity",
            t.prompt_cosine,
            t.prompt_cosine_unavailable ?? null
          )}${metric("visual similarity", t.visual_cosine, t.visual_cosine_unavailable ?? null)}</dl><table class="tracks">${rows || "<tr><td>nothing else changed</td></tr>"}</table>`;
        } else if (!view.plan.sequenced) {
          html += `<p class="evolution-provenance">no transitions: the evidence does not establish an order</p>`;
        }
        const edges = view.lineage.filter((e) => e.parent === m.ref || e.child === m.ref);
        if (edges.length) {
          html += `<h3>lineage</h3><ul class="lineage">${edges.map((e) => `<li>${esc(e.parent)} \u2192 ${esc(e.child)} <span class="kind">${esc(e.kind)}</span></li>`).join("")}</ul>`;
        }
        inspector.innerHTML = html;
      }
      const panels = { sequence, drift, families, lineage, compare };
      const isTab = (name) => name in panels;
      let tab = view.plan.sequenced ? "sequence" : "families";
      function draw4() {
        panels[tab]();
        for (const b of everyElement(root, "[data-tab]", HTMLElement)) {
          b.classList.toggle("on", b.dataset.tab === tab);
        }
        for (const el of everyElement(root, "[data-ref]", HTMLElement)) {
          const ref = el.dataset.ref;
          el.classList.toggle("on", ref === selected);
          el.classList.toggle("pair", pair !== null && ref !== void 0 && pair.includes(ref) && ref !== selected);
        }
        for (const el of everyElement(root, "[data-pair]", HTMLElement)) {
          el.classList.toggle("on", pair !== null && el.dataset.pair === pair.join("|"));
        }
        panel2();
        inspect();
      }
      root.addEventListener("click", (event) => {
        const chosen = closestFrom(event.target, "[data-tab]", HTMLElement);
        if (chosen !== null) {
          const name = requireData(chosen, "tab");
          if (isTab(name)) tab = name;
          draw4();
          return;
        }
        const dot = closestFrom(event.target, "[data-pair]", Element);
        if (dot !== null) {
          const [before, after] = requireAttribute(dot, "data-pair").split("|");
          if (before !== void 0 && after !== void 0) {
            pair = [before, after];
            selected = after;
          }
          draw4();
          return;
        }
        const el = closestFrom(event.target, "[data-ref]", Element);
        if (el === null) return;
        const ref = requireAttribute(el, "data-ref");
        if (event.shiftKey && selected !== null && selected !== ref) {
          pair = [selected, ref];
          tab = "compare";
        } else {
          selected = ref;
        }
        draw4();
      });
      draw4();
    }
  })();
  function requireAttribute(node, name) {
    const held2 = node.getAttribute(name);
    if (held2 === null) throw new Error(`expected a ${name} on ${node.tagName.toLowerCase()}`);
    return held2;
  }
  function diffTokens(a, b) {
    const x = a ? a.split(/\s+/) : [];
    const y = b ? b.split(/\s+/) : [];
    const n = x.length;
    const m = y.length;
    const L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    const at = (i2, j2) => L[i2]?.[j2] ?? 0;
    for (let i2 = n - 1; i2 >= 0; i2--) {
      for (let j2 = m - 1; j2 >= 0; j2--) {
        const row = L[i2];
        if (row === void 0) continue;
        row[j2] = x[i2] === y[j2] ? at(i2 + 1, j2 + 1) + 1 : Math.max(at(i2 + 1, j2), at(i2, j2 + 1));
      }
    }
    const out = [];
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
      if (x[i] === y[j]) {
        out.push(esc(x[i]));
        i++;
        j++;
      } else if (at(i + 1, j) >= at(i, j + 1)) {
        out.push(`<del>${esc(x[i])}</del>`);
        i++;
      } else {
        out.push(`<ins>${esc(y[j])}</ins>`);
        j++;
      }
    }
    while (i < n) out.push(`<del>${esc(x[i++])}</del>`);
    while (j < m) out.push(`<ins>${esc(y[j++])}</ins>`);
    return out.join(" ");
  }

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
    const panel2 = findElement(root, "[data-recipe]", HTMLElement);
    if (!panel2) return;
    for (const section of everyElement(panel2, "[data-recipe-field]", HTMLElement)) {
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
    const all = findElement(panel2, "[data-copy-all]", HTMLElement);
    if (all) all.addEventListener("click", () => void copied(all, wholeRecipe(panel2)));
    const refit = () => {
      for (const field of everyElement(panel2, "[data-scratch]", HTMLTextAreaElement)) fit(field);
    };
    panel2.addEventListener("toggle", refit);
    new MutationObserver(refit).observe(root, { attributeFilter: ["data-inspector"] });
    refit();
  }

  // src/analyze.ts
  function mountAnalyze(root) {
    const panel2 = findElement(root, "[data-analyze]", HTMLElement);
    if (!panel2) return;
    for (const button2 of everyElement(panel2, "[data-copy-prompt]", HTMLElement)) {
      const use = button2.closest("[data-prompt]");
      const text = use && findElement(use, "[data-prompt-text]", HTMLElement);
      if (!text) continue;
      button2.addEventListener("click", () => void copied(button2, text.textContent ?? ""));
    }
  }

  // src/endless.ts
  var WINDOW = 6;
  var REACH = 600;
  function mountEndless(root) {
    const grid = findElement(root, "[data-grid]", HTMLElement);
    if (!grid) return;
    const cells = findElement(grid, "[data-cells]", HTMLElement);
    const pager = findElement(grid, "[data-pager]", HTMLElement);
    if (!cells || !pager) return;
    const pages = Number(requireData(grid, "pages"));
    const first = Number(requireData(grid, "page"));
    const qbase = grid.dataset.qbase ?? "";
    if (!Number.isFinite(pages) || !Number.isFinite(first)) return;
    for (const cell of cells.children) {
      if (cell instanceof HTMLElement) cell.dataset.page = String(first);
    }
    let lowest = first;
    let highest = first;
    let busy = false;
    let again = false;
    const settle2 = (state) => {
      grid.dataset.endless = state;
    };
    const dropped = /* @__PURE__ */ new Map();
    const cellsOf = (page) => [...cells.children].filter(
      (one) => one instanceof HTMLElement && one.dataset.page === String(page)
    );
    const spanOf = (held2) => {
      if (!held2.length) return 0;
      const top = Math.min(...held2.map((one) => one.getBoundingClientRect().top));
      const bottom = Math.max(...held2.map((one) => one.getBoundingClientRect().bottom));
      return bottom - top;
    };
    const padding = () => Number.parseFloat(cells.style.paddingTop || "0") || 0;
    const dropOldest = () => {
      const held2 = cellsOf(lowest);
      if (!held2.length) return;
      const height = spanOf(held2);
      for (const one of held2) one.remove();
      dropped.set(lowest, { height });
      cells.style.paddingTop = `${padding() + height}px`;
      lowest += 1;
    };
    const fetchPage = async (page) => {
      const answered2 = await fetch(`/g/grid?${qbase}page=${page}`, { headers: { accept: "text/html" } });
      if (!answered2.ok) return null;
      const parsed = new DOMParser().parseFromString(await answered2.text(), "text/html");
      const fresh = parsed.querySelector("[data-cells]");
      if (!fresh) return null;
      const made = [];
      for (const one of [...fresh.children]) {
        if (!(one instanceof HTMLElement)) continue;
        one.dataset.page = String(page);
        made.push(one);
      }
      return made;
    };
    const extend = async () => {
      if (highest >= pages) return;
      grid.dataset.loading = "true";
      try {
        const made = await fetchPage(highest + 1);
        if (made) {
          cells.append(...made);
          highest += 1;
          while (highest - lowest + 1 > WINDOW) dropOldest();
        }
      } catch {
      } finally {
        delete grid.dataset.loading;
      }
    };
    const pump = async () => {
      if (busy) {
        again = true;
        return;
      }
      busy = true;
      settle2("working");
      let moved = false;
      try {
        do {
          again = false;
          while (highest < pages) {
            if (!inReach()) break;
            const before = highest;
            await extend();
            if (highest === before) break;
            moved = true;
          }
        } while (again);
      } finally {
        busy = false;
        settle2("idle");
      }
      if (moved && highest < pages && inReach()) {
        requestAnimationFrame(() => void pump());
      }
    };
    const inReach = () => pager.getBoundingClientRect().top <= window.innerHeight + REACH;
    const restore = async () => {
      if (busy || lowest <= 1 || !dropped.has(lowest - 1)) return;
      busy = true;
      settle2("working");
      try {
        const page = lowest - 1;
        const made = await fetchPage(page);
        if (made) {
          cells.prepend(...made);
          const held2 = dropped.get(page);
          dropped.delete(page);
          cells.style.paddingTop = `${Math.max(0, padding() - (held2?.height ?? 0))}px`;
          lowest = page;
          while (highest - lowest + 1 > WINDOW) {
            const last = cellsOf(highest);
            if (!last.length) break;
            for (const one of last) one.remove();
            highest -= 1;
          }
        }
      } catch {
      } finally {
        busy = false;
        settle2("idle");
      }
    };
    const watchDown = new IntersectionObserver(
      (entries) => {
        if (entries.some((one) => one.isIntersecting)) void pump();
      },
      { rootMargin: `${REACH}px 0px` }
    );
    watchDown.observe(pager);
    let shown = first;
    let waiting = 0;
    const follow = () => {
      waiting = 0;
      if (window.scrollY < REACH) void restore();
      if (inReach()) void pump();
      const top = [...cells.children].find(
        (one) => one instanceof HTMLElement && one.getBoundingClientRect().bottom > 0
      );
      const page = Number(top?.dataset.page);
      if (!Number.isFinite(page) || page === shown) return;
      shown = page;
      grid.dataset.page = String(page);
      const held2 = new URLSearchParams(window.location.search);
      if (page > 1) held2.set("page", String(page));
      else held2.delete("page");
      const spelled3 = held2.toString();
      window.history.replaceState(
        window.history.state,
        "",
        spelled3 ? `${window.location.pathname}?${spelled3}` : window.location.pathname
      );
    };
    window.addEventListener(
      "scroll",
      () => {
        if (!waiting) waiting = window.requestAnimationFrame(follow);
      },
      { passive: true }
    );
    settle2("idle");
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

  // src/workspace.ts
  var VERSION = 1;
  var KEY = `sg.workspace.v${VERSION}`;
  function workspace() {
    try {
      const held2 = localStorage.getItem(KEY);
      if (!held2) return {};
      const parsed = JSON.parse(held2);
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
  function board() {
    const held2 = workspace().board;
    return Array.isArray(held2) ? held2 : [];
  }
  function pin(one) {
    const held2 = board().filter((other) => other.id !== one.id);
    remember({ board: [...held2, one] });
  }
  function unpin(id) {
    remember({ board: board().filter((one) => one.id !== id) });
  }

  // src/filters.ts
  var NOT_THE_QUESTION = /* @__PURE__ */ new Set(["page"]);
  var EDITING = "sg.filters.editing";
  function question() {
    const held2 = new URLSearchParams(window.location.search);
    for (const name of NOT_THE_QUESTION) held2.delete(name);
    return held2;
  }
  function go(held2) {
    const spelled3 = held2.toString();
    const url = spelled3 ? `${window.location.pathname}?${spelled3}` : window.location.pathname;
    let editing = false;
    try {
      editing = sessionStorage.getItem(EDITING) === "1";
      sessionStorage.setItem(EDITING, "1");
    } catch {
    }
    if (editing) window.location.replace(url);
    else window.location.assign(url);
  }
  function endSession() {
    try {
      sessionStorage.removeItem(EDITING);
    } catch {
    }
  }
  function held(key, carried) {
    const asked4 = question();
    if (carried === "scope") {
      const value = asked4.get(key);
      return new Set(value ? [value] : []);
    }
    const found = /* @__PURE__ */ new Set();
    for (const spelled3 of asked4.getAll("f")) {
      const parts = spelled3.split(":");
      if (parts.length >= 3 && parts[0] === key) found.add(parts.slice(2).join(":"));
    }
    return found;
  }
  function toggled(key, carried, op, value, on) {
    const asked4 = question();
    if (carried === "scope") {
      if (on) asked4.set(key, value);
      else asked4.delete(key);
      return asked4;
    }
    const spelled3 = `${key}:${op}:${value}`;
    const rest = asked4.getAll("f").filter((one) => one !== spelled3);
    asked4.delete("f");
    for (const one of rest) asked4.append("f", one);
    if (on) asked4.append("f", spelled3);
    return asked4;
  }
  function toggledExact(key, op, param, value) {
    const asked4 = question();
    const mine = `${key}:${op}:${param}=`;
    const rest = asked4.getAll("f").filter((one) => !one.startsWith(mine));
    asked4.delete("f");
    for (const one of rest) asked4.append("f", one);
    if (value !== "") asked4.append("f", `${mine}${value}`);
    return asked4;
  }
  function onlyClause(key, carried, op, value) {
    const asked4 = question();
    if (carried === "scope") {
      if (value === null) asked4.delete(key);
      else asked4.set(key, value);
      return asked4;
    }
    const rest = asked4.getAll("f").filter((one) => !one.startsWith(`${key}:`));
    asked4.delete("f");
    for (const one of rest) asked4.append("f", one);
    if (value !== null) asked4.append("f", `${key}:${op}:${value}`);
    return asked4;
  }
  function counted(n) {
    return n.toLocaleString();
  }
  function operatorFor(told, carried) {
    if (carried === "scope" || !told.multi) return told.ops[0] ?? "eq";
    if (told.multi === "any") return "any";
    return panelState(`all:${told.key}`) ? "eq" : "any";
  }
  function drawList(body, told, carried, again) {
    body.replaceChildren();
    if (!told.options.length) {
      const empty = document.createElement("p");
      empty.className = "filter-note";
      empty.textContent = "nothing here answers this yet";
      body.append(empty);
      return;
    }
    if (told.multi === "both") {
      const choice = document.createElement("div");
      choice.className = "filter-choice";
      choice.dataset.filterChoice = told.key;
      const all = panelState(`all:${told.key}`) === true;
      for (const [mode, said3, why] of [
        ["any", "any of", `media with any one of these ${told.label}s`],
        ["all", "all of", `media carrying every one of these ${told.label}s`]
      ]) {
        const button2 = document.createElement("button");
        button2.type = "button";
        button2.className = "filter-choice-mode";
        button2.dataset.mode = mode;
        button2.title = why;
        button2.textContent = said3;
        button2.setAttribute("aria-pressed", String(all === (mode === "all")));
        button2.addEventListener("click", () => {
          rememberPanel(`all:${told.key}`, mode === "all");
          const wanted = mode === "all" ? "eq" : "any";
          const asked4 = question();
          const rest = asked4.getAll("f").filter((held2) => !held2.startsWith(`${told.key}:`));
          const mine = asked4.getAll("f").filter((held2) => held2.startsWith(`${told.key}:`)).map((held2) => `${told.key}:${wanted}:${held2.split(":").slice(2).join(":")}`);
          asked4.delete("f");
          for (const held2 of [...rest, ...mine]) asked4.append("f", held2);
          if (mine.length) go(asked4);
          else again();
        });
        choice.append(button2);
      }
      body.append(choice);
    }
    if (told.options.length > 12 || told.more > 0) {
      const find = document.createElement("input");
      find.type = "search";
      find.className = "filter-find";
      find.placeholder = `search ${told.label}`;
      find.setAttribute("aria-label", `search ${told.label}`);
      find.addEventListener("input", () => {
        const wanted = find.value.trim().toLowerCase();
        for (const row of everyElement(body, "[data-option]", HTMLElement)) {
          row.hidden = wanted !== "" && !(row.dataset.label ?? "").toLowerCase().includes(wanted);
        }
      });
      body.append(find);
    }
    const list = document.createElement("ul");
    list.className = "filter-list";
    for (const one of told.options) {
      const row = document.createElement("li");
      row.dataset.option = one.value;
      row.dataset.label = one.label;
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "filter-option";
      pick.dataset.chosen = one.chosen ? "true" : "false";
      pick.setAttribute("aria-pressed", one.chosen ? "true" : "false");
      const name = document.createElement("span");
      name.className = "filter-option-label";
      name.textContent = one.label;
      const tally = document.createElement("span");
      tally.className = "filter-option-count";
      tally.textContent = counted(one.count);
      pick.append(name, tally);
      if (one.count === 0 && !one.chosen) pick.disabled = true;
      pick.addEventListener("click", () => {
        go(toggled(told.key, carried, operatorFor(told, carried), one.value, !one.chosen));
      });
      row.append(pick);
      list.append(row);
    }
    body.append(list);
    if (told.more > 0) {
      const rest = document.createElement("p");
      rest.className = "filter-note";
      rest.textContent = `${counted(told.more)} more \u2014 search to narrow`;
      body.append(rest);
    }
  }
  function drawRange(body, key, carried, kind, ops) {
    body.replaceChildren();
    const now = held(key, carried);
    if (kind === "bool") {
      const pair = document.createElement("div");
      pair.className = "filter-choice";
      for (const [value, said3] of [
        ["1", "yes"],
        ["0", "no"]
      ]) {
        const button2 = document.createElement("button");
        button2.type = "button";
        button2.className = "filter-choice-mode";
        button2.dataset.option = value;
        button2.dataset.label = said3;
        button2.textContent = said3;
        const on = now.has(value);
        button2.setAttribute("aria-pressed", String(on));
        button2.addEventListener("click", () => go(onlyClause(key, carried, ops[0] ?? "eq", on ? null : value)));
        pair.append(button2);
      }
      body.append(pair);
      return;
    }
    if (kind === "pair") {
      const form2 = document.createElement("form");
      form2.className = "filter-range";
      const field = document.createElement("input");
      field.type = "text";
      field.placeholder = "key=value";
      field.setAttribute("aria-label", `${key}, written key equals value`);
      field.value = [...now][0] ?? "";
      const apply2 = document.createElement("button");
      apply2.type = "submit";
      apply2.textContent = "apply";
      form2.append(field, apply2);
      form2.addEventListener("submit", (event) => {
        event.preventDefault();
        const wanted = field.value.trim();
        go(onlyClause(key, carried, ops[0] ?? "eq", wanted === "" ? null : wanted));
      });
      body.append(form2);
      return;
    }
    const form = document.createElement("form");
    form.className = "filter-range";
    const fields = [];
    for (const op of ops) {
      if (op !== "gte" && op !== "lte" && op !== "eq") continue;
      const wrap = document.createElement("label");
      wrap.className = "filter-range-field";
      const said3 = document.createElement("span");
      said3.textContent = op === "gte" ? "from" : op === "lte" ? "to" : "exactly";
      const input = document.createElement("input");
      input.type = kind === "date" ? "date" : "number";
      if (kind === "num") input.step = "any";
      input.name = op;
      input.setAttribute("aria-label", `${key} ${said3.textContent}`);
      for (const spelled3 of question().getAll("f")) {
        const parts = spelled3.split(":");
        if (parts[0] === key && parts[1] === op) input.value = parts.slice(2).join(":");
      }
      if (carried === "scope" && op === ops[0]) {
        const value = [...now][0];
        if (value) input.value = value;
      }
      wrap.append(said3, input);
      form.append(wrap);
      fields.push({ op, input });
    }
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.textContent = "apply";
    form.append(apply);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      let asked4 = question();
      for (const { op, input } of fields) {
        const value = input.value.trim();
        if (carried === "scope") {
          asked4 = onlyClause(key, carried, op, value === "" ? null : value);
          continue;
        }
        const rest = asked4.getAll("f").filter((one) => !one.startsWith(`${key}:${op}:`));
        asked4.delete("f");
        for (const one of rest) asked4.append("f", one);
        if (value !== "") asked4.append("f", `${key}:${op}:${value}`);
      }
      go(asked4);
    });
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "filter-range-clear";
    clear.textContent = "clear";
    clear.addEventListener("click", () => go(onlyClause(key, carried, ops[0] ?? "eq", null)));
    form.append(clear);
    body.append(form);
  }
  function drawParamRange(body, param, ops) {
    for (const op of ops) {
      if (op === "eq") continue;
      const row = document.createElement("label");
      row.className = "filter-range";
      row.dataset.paramOp = op;
      const said3 = document.createElement("span");
      said3.textContent = op === "gte" ? "at least" : "at most";
      const box = document.createElement("input");
      box.type = "number";
      box.step = "any";
      box.setAttribute("aria-label", `${param} ${said3.textContent}`);
      const held2 = question().getAll("f").find((one) => one.startsWith(`param.num:${op}:${param}=`));
      if (held2) box.value = held2.slice(held2.lastIndexOf("=") + 1);
      box.addEventListener("change", () => {
        const wanted = box.value.trim();
        go(toggledExact("param.num", op, param, wanted));
      });
      row.append(said3, box);
      body.append(row);
    }
  }
  async function fill(section) {
    const body = findElement(section, "[data-filter-body]", HTMLElement);
    const key = section.dataset.filter;
    if (!body || !key) return;
    const carried = section.dataset.carried ?? "facet";
    const ops = (section.dataset.ops ?? "eq").split(",").filter(Boolean);
    if (section.dataset.listable !== "1") {
      drawRange(body, key, carried, section.dataset.valueKind ?? "int", ops);
      body.dataset.state = "ready";
      return;
    }
    body.dataset.state = "counting";
    body.replaceChildren();
    const waiting = document.createElement("p");
    waiting.className = "filter-note";
    waiting.textContent = "counting\u2026";
    body.append(waiting);
    const asked4 = question();
    asked4.set("key", key);
    try {
      const answered2 = await fetch(`/g/options?${asked4.toString()}`, { headers: { accept: "application/json" } });
      if (!answered2.ok) throw new Error(`${answered2.status}`);
      drawList(body, await answered2.json(), carried, () => void fill(section));
      body.dataset.state = "ready";
    } catch {
      body.replaceChildren();
      const failed = document.createElement("p");
      failed.className = "filter-note warn";
      failed.textContent = "could not count these";
      body.append(failed);
      body.dataset.state = "failed";
    }
  }
  async function drawParamValues(body, param, label2) {
    const said3 = document.createElement("p");
    said3.className = "filter-note";
    said3.textContent = `${label2} \u2014 counting\u2026`;
    body.replaceChildren(said3);
    body.dataset.state = "counting";
    body.dataset.param = param;
    const asked4 = question();
    asked4.set("param", param);
    let told;
    try {
      const answered2 = await fetch(`/g/fields/values?${asked4.toString()}`, { headers: { accept: "application/json" } });
      if (!answered2.ok) throw new Error(`${answered2.status}`);
      told = await answered2.json();
    } catch {
      said3.className = "filter-note warn";
      said3.textContent = `could not count ${label2}`;
      body.dataset.state = "failed";
      return;
    }
    body.replaceChildren();
    const naming = document.createElement("p");
    naming.className = "filter-note";
    naming.textContent = `${label2} \xB7 ${param}`;
    body.append(naming);
    const list = document.createElement("ul");
    list.className = "filter-list";
    for (const one of told.options) {
      const row = document.createElement("li");
      row.dataset.option = one.value;
      row.dataset.label = one.label;
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "filter-option";
      pick.dataset.chosen = one.chosen ? "true" : "false";
      pick.setAttribute("aria-pressed", one.chosen ? "true" : "false");
      const name = document.createElement("span");
      name.className = "filter-option-label";
      name.textContent = one.label;
      const tally = document.createElement("span");
      tally.className = "filter-option-count";
      tally.textContent = counted(one.count);
      pick.append(name, tally);
      pick.addEventListener("click", () => {
        go(toggled("param.is", "facet", "any", `${param}=${one.value}`, !one.chosen));
      });
      row.append(pick);
      list.append(row);
    }
    body.append(list);
    if (told.options.length === 0) {
      const none = document.createElement("p");
      none.className = "filter-note";
      none.textContent = "nothing here holds a value for it";
      body.append(none);
    }
    if (told.more > 0) {
      const rest = document.createElement("p");
      rest.className = "filter-note";
      rest.textContent = `${counted(told.more)} more`;
      body.append(rest);
    }
    const form = document.createElement("form");
    form.className = "filter-range";
    const typed = document.createElement("input");
    typed.type = "text";
    typed.value = `${param}=`;
    typed.setAttribute("aria-label", `${label2}, written key equals value`);
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.textContent = "apply";
    form.append(typed, apply);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const wanted = typed.value.trim();
      go(onlyClause("param.is", "facet", "any", wanted === "" ? null : wanted));
    });
    body.append(form);
    body.dataset.state = "ready";
  }
  var SETTLED_MS = 140;
  function mountFind(drawer, reveal) {
    const box = findElement(drawer, "[data-filter-find]", HTMLElement);
    const field = box && findElement(box, "[data-filter-find-input]", HTMLInputElement);
    const list = box && findElement(box, "[data-filter-found]", HTMLElement);
    if (!box || !field || !list) return;
    let at = -1;
    let ticket = 0;
    let timer = 0;
    const rows = () => everyElement(list, "[data-field]", HTMLElement);
    const highlight = (wanted) => {
      const all = rows();
      if (all.length === 0) {
        at = -1;
        return;
      }
      at = (wanted + all.length) % all.length;
      for (const [index, row] of all.entries()) {
        row.setAttribute("aria-selected", String(index === at));
        if (index === at) row.scrollIntoView({ block: "nearest" });
      }
    };
    const shut = () => {
      list.hidden = true;
      list.replaceChildren();
      field.setAttribute("aria-expanded", "false");
      at = -1;
    };
    const take = (one) => {
      shut();
      field.value = "";
      reveal(one.key);
      if (one.param === null) return;
      const section = findElement(drawer, `[data-filter="${one.key}"]`, HTMLDetailsElement);
      const body = section && findElement(section, "[data-filter-body]", HTMLElement);
      if (!body) return;
      if (one.key === "param.num") {
        body.replaceChildren();
        body.dataset.param = one.param;
        const said3 = document.createElement("p");
        said3.className = "filter-note";
        said3.dataset.paramSpelling = one.param;
        said3.textContent = one.param;
        body.append(said3);
        drawParamRange(body, one.param, one.ops);
        body.dataset.state = "ready";
        return;
      }
      void drawParamValues(body, one.param, one.label);
    };
    const draw4 = (told) => {
      list.replaceChildren();
      if (told.fields.length === 0) {
        const none = document.createElement("p");
        none.className = "filter-note";
        none.textContent = "nothing here answers to that";
        list.append(none);
      }
      for (const one of told.fields) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "filter-found-row";
        row.dataset.field = one.key;
        if (one.param !== null) row.dataset.param = one.param;
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", "false");
        const name = document.createElement("span");
        name.className = "filter-found-label";
        name.textContent = one.label;
        const where = document.createElement("span");
        where.className = "filter-found-group";
        where.textContent = one.curated ? one.group : `${one.group} \xB7 ${one.covered}`;
        row.append(name, where);
        row.addEventListener("click", () => take(one));
        list.append(row);
      }
      if (told.more > 0) {
        const cut = document.createElement("p");
        cut.className = "filter-note";
        cut.textContent = `${told.more} more \u2014 keep typing`;
        list.append(cut);
      }
      list.hidden = false;
      field.setAttribute("aria-expanded", "true");
      highlight(0);
    };
    const look = async () => {
      const wanted = field.value.trim();
      const mine = ++ticket;
      const asked4 = question();
      asked4.set("search", wanted);
      try {
        const answered2 = await fetch(`/g/fields?${asked4.toString()}`, { headers: { accept: "application/json" } });
        if (!answered2.ok) throw new Error(`${answered2.status}`);
        const told = await answered2.json();
        if (mine === ticket) draw4(told);
      } catch {
        if (mine === ticket) shut();
      }
    };
    field.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => void look(), SETTLED_MS);
    });
    field.addEventListener("focus", () => void look());
    field.addEventListener("keydown", (event) => {
      if (list.hidden) return;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        highlight(at + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        highlight(at - 1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        rows()[at]?.click();
      } else if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        shut();
      }
    });
    document.addEventListener("click", (event) => {
      if (!list.hidden && event.target instanceof Node && !box.contains(event.target)) shut();
    });
  }
  function mountFilters(root) {
    const drawer = findElement(root, "[data-filters-panel]", HTMLElement);
    const open = findElement(root, "[data-filters-open]", HTMLElement);
    if (!drawer || !open) return;
    const show = (on, arranged = true) => {
      drawer.hidden = !on;
      root.dataset.filters = on ? "open" : "closed";
      open.setAttribute("aria-expanded", on ? "true" : "false");
      if (arranged) remember({ filters: on ? "open" : "closed" });
      if (!on) endSession();
    };
    open.addEventListener("click", () => show(drawer.hidden !== false));
    const close = findElement(drawer, "[data-filters-close]", HTMLElement);
    if (close) close.addEventListener("click", () => show(false));
    for (const section of everyElement(drawer, "[data-filter]", HTMLDetailsElement)) {
      const key = section.dataset.filter ?? "";
      const said3 = panelState(`filter:${key}`);
      if (said3) section.open = true;
      section.addEventListener("toggle", () => {
        rememberPanel(`filter:${key}`, section.open);
        if (section.open && !section.dataset.filled) {
          section.dataset.filled = "1";
          void fill(section);
        }
      });
      if (section.open) {
        section.dataset.filled = "1";
        void fill(section);
      }
    }
    const reveal = (key) => {
      show(true);
      const section = findElement(drawer, `[data-filter="${key}"]`, HTMLDetailsElement);
      if (!section) return;
      section.open = true;
      if (!section.dataset.filled) {
        section.dataset.filled = "1";
        void fill(section);
      }
      section.scrollIntoView({ block: "nearest" });
    };
    for (const chip of everyElement(root, "[data-chip-edit]", HTMLElement)) {
      chip.addEventListener("click", () => reveal(chip.dataset.chipEdit ?? ""));
    }
    mountFind(drawer, reveal);
    for (const clear of everyElement(root, "[data-filters-clear], [data-chips-clear]", HTMLElement)) {
      clear.addEventListener("click", endSession);
    }
    const search = findElement(root, '[data-ask] input[type="search"]', HTMLInputElement);
    if (search) {
      register([
        {
          key: "/",
          by: "gallery: search",
          run: () => {
            search.focus();
            search.select();
          }
        }
      ]);
    }
    show(workspace().filters === "open", false);
  }

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

  // src/zoom.ts
  var MAX_SCALE = 40;
  var TRAY_MAX_SCALE = 16;

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
    const paint2 = () => {
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
    const clamp2 = (scale) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
    const frame = (framing) => {
      if (!still) return;
      const box = stageBox.getBoundingClientRect();
      const size = fitted();
      const scale = framing === "fit" ? 1 : framing === "fill" ? fillScale(size, box) : still.source ? actualScale(still.source, size) : 1;
      look = { framing, scale: clamp2(scale), x: 0, y: 0 };
      paint2();
      promote();
    };
    const zoomAbout = (factor, clientX, clientY) => {
      if (!still) return;
      const box = stageBox.getBoundingClientRect();
      const next = clamp2(look.scale * factor);
      if (next === look.scale) return;
      const px = clientX - (box.left + box.width / 2);
      const py = clientY - (box.top + box.height / 2);
      const ratio = next / look.scale;
      const held2 = tethered(px - (px - look.x) * ratio, py - (py - look.y) * ratio, next);
      look = { framing: "free", scale: next, ...held2 };
      paint2();
      promote();
    };
    const resettle = () => {
      if (!still) return;
      if (look.framing === "fit" || look.framing === "fill" || look.framing === "actual") {
        frame(look.framing);
        return;
      }
      look = { ...look, ...tethered(look.x, look.y, look.scale) };
      paint2();
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
    const panel2 = (named) => {
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
        const pixels2 = (delta) => event.deltaMode === 0 ? delta : delta * 16;
        if (walking) {
          stepped(pixels2(event.deltaY || event.deltaX));
          return;
        }
        zoomAbout(Math.exp(-pixels2(event.deltaY) / 400), event.clientX, event.clientY);
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
        const held2 = tethered(from.ox + (event.clientX - from.x), from.oy + (event.clientY - from.y), look.scale);
        look = { framing: "free", scale: look.scale, ...held2 };
        paint2();
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
      const held2 = workspace().showEvery;
      return typeof held2 === "number" && held2 > 0 ? held2 : DEFAULT_EVERY;
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
    for (const button2 of everyElement(root, "[data-inspector-toggle]", HTMLElement)) {
      onElement(button2, "click", () => showInspector(root.dataset.inspector !== "open"));
    }
    for (const button2 of everyElement(root, "[data-panel-open]", HTMLElement)) {
      onElement(button2, "click", () => {
        showInspector(true);
        panel2(requireData(button2, "panelOpen"));
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
        const said3 = panelState(named);
        section.open = said3 ?? (generated ? named === "creation" : named === "about");
        onElement(section, "toggle", () => rememberPanel(named, section.open));
      }
    }
    mountRecipe(root);
    root.dataset.inspector = root.dataset.inspector ?? "closed";
    root.dataset.chrome = "visible";
    stageBox.dataset.quality = "preview";
    paint2();
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

  // src/gallery.ts
  var asked = (spelled3, take) => {
    const question2 = new URLSearchParams(spelled3);
    const one = (name) => question2.get(name);
    const counted2 = (name) => {
      const held2 = question2.get(name);
      return held2 === null ? null : Number(held2);
    };
    return {
      take,
      folder: one("folder"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted2("rating_min"),
      q: one("q"),
      sort: one("sort"),
      f: question2.getAll("f")
    };
  };
  (() => {
    mountFilters(document.body);
    mountAnalyze(document.body);
    mountEndless(document.body);
    const ask2 = findElement(document, "[data-ask]", HTMLFormElement);
    if (ask2) {
      const fields = () => [
        ...everyElement(ask2, "input", HTMLInputElement),
        ...everyElement(ask2, "select", HTMLSelectElement)
      ];
      ask2.addEventListener("submit", () => {
        const phrase = requireElement(ask2, '[name="q"]', HTMLInputElement);
        const sort = requireElement(ask2, '[name="sort"]', HTMLSelectElement);
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
      const held2 = mounted2 ? requireData(mounted2, "qbase").replace(/&$/, "") : window.location.search;
      const question2 = new URLSearchParams(held2);
      question2.delete("page");
      question2.delete("size");
      return question2.toString();
    };
    const cutoff = (spelled3) => {
      if (!new URLSearchParams(spelled3).get("q")) return null;
      const mounted2 = grid();
      const total = mounted2 ? Number(requireData(mounted2, "total")) : Number.NaN;
      return Number.isFinite(total) && total > 0 ? total : 1;
    };
    const rememberer = findElement(document, "[data-remember-view]", HTMLElement);
    rememberer?.addEventListener("click", async () => {
      const name = await askText("remember this question as", {
        detail: "it goes in the strip below, and opens the question again -- it makes no collection",
        placeholder: "portraits from June",
        label: "name"
      });
      if (name === null) return;
      const { error } = await api.POST("/views", { body: { name, qs: window.location.search } });
      if (error) {
        await say(refusal(error, "that question was not remembered"));
        return;
      }
      window.location.reload();
    });
    for (const link of everyElement(document, "[data-remembered-open]", HTMLElement)) {
      link.addEventListener("click", () => {
        const id = requireData(link, "rememberedOpen");
        void fetch(`/views/${encodeURIComponent(id)}/opened`, { method: "POST", keepalive: true });
      });
    }
    for (const drop of everyElement(document, "[data-forget-view]", HTMLElement)) {
      drop.addEventListener("click", async () => {
        const id = requireData(drop, "forgetView");
        const held2 = await fetch(`/views/${encodeURIComponent(id)}/forget`, { method: "POST" });
        if (!held2.ok) {
          await say("that question was not forgotten");
          return;
        }
        drop.closest("[data-remembered-view]")?.remove();
      });
    }
    const saver = findElement(document, "[data-save-smart]", HTMLElement);
    saver?.addEventListener("click", async () => {
      const spelled3 = spelling();
      const name = await askText("name this smart collection", {
        detail: "the question you are looking at becomes its rule, and its members follow the library",
        placeholder: "portraits from June",
        label: "collection name"
      });
      if (name === null) return;
      const take = cutoff(spelled3);
      const { data, error } = await api.POST("/albums/smart", { body: { name, ...asked(spelled3, take) } });
      if (!data) {
        await say(refusal(error, "the view could not be saved"));
        return;
      }
      window.location.assign(`/t/${data.slug}`);
    });
    const replacer = findElement(document, "[data-replace-smart]", HTMLElement);
    replacer?.addEventListener("click", async () => {
      const shelf = await api.GET("/albums", { headers: { accept: "application/json" } });
      const smarts = (shelf.data ?? []).filter((held2) => held2.kind === "smart");
      if (smarts.length === 0) {
        await say("no smart collection exists yet", {
          detail: "save this view as a new one instead, and it will be here to replace next time"
        });
        return;
      }
      const named = await askChoice(
        "replace the rule of which smart collection?",
        smarts.map((held2) => ({ value: held2.slug, label: held2.name, note: `/t/${held2.slug}` })),
        { detail: "its members become whatever this question answers, from now on" }
      );
      if (named === null) return;
      const current2 = await api.GET("/t/{slug}", {
        params: { path: { slug: named } },
        headers: { accept: "application/json" }
      });
      if (!current2.data) {
        await say(`no collection at /t/${named}`);
        return;
      }
      const spelled3 = spelling();
      const take = cutoff(spelled3);
      const { data, error } = await api.PUT("/t/{slug}/rule", {
        params: { path: { slug: named } },
        body: { expected_rev: current2.data.definition_rev, ...asked(spelled3, take) }
      });
      if (!data) {
        await say(refusal(error, "the rule could not be replaced"));
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
      const held2 = peeked.get(key);
      if (held2) return held2;
      const question2 = new URLSearchParams(s.qbase);
      const { data } = await api.GET("/g/peek", {
        params: {
          query: {
            folder: question2.get("folder"),
            album: question2.get("album"),
            person: question2.get("person"),
            artifact: question2.get("artifact"),
            kind: question2.get("kind"),
            favorite: question2.get("favorite"),
            rating_min: question2.get("rating_min") === null ? null : Number(question2.get("rating_min")),
            q: question2.get("q"),
            sort: question2.get("sort"),
            f: question2.getAll("f"),
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
          if (!item.thumb) {
            const said3 = document.createElement("span");
            said3.className = "cell-kind";
            said3.dataset.cellKind = item.kind;
            said3.textContent = item.kind === "audio" ? "audio" : "doc";
            said3.title = item.name;
            return said3;
          }
          const img = new Image();
          img.src = item.thumb;
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
        const held2 = mounted2 && findElement(mounted2, "[data-viewer]", HTMLElement);
        viewer = held2 ? mountViewer(held2, (href) => void lightbox?.open(href, "replace")) : null;
      },
      generation: () => {
        const shown = findElement(document, "[data-lightbox]", HTMLElement);
        const held2 = shown?.dataset.currency;
        if (held2) return held2;
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
        const question2 = new URLSearchParams(window.location.search);
        const { data } = await api.GET("/g/locate/{slug}", {
          params: {
            path: { slug },
            query: {
              folder: question2.get("folder"),
              album: question2.get("album"),
              person: question2.get("person"),
              artifact: question2.get("artifact"),
              kind: question2.get("kind"),
              favorite: question2.get("favorite"),
              rating_min: question2.get("rating_min") === null ? null : Number(question2.get("rating_min")),
              q: question2.get("q"),
              sort: question2.get("sort"),
              size: question2.get("size") === null ? null : Number(question2.get("size")),
              f: question2.getAll("f")
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

  // src/keywords.ts
  var draw = (list, keywords) => {
    list.replaceChildren(
      ...keywords.map((one) => {
        const row = document.createElement("li");
        row.className = "keyword-row";
        row.dataset.keyword = one.tag;
        row.dataset.pictures = String(one.pictures);
        const link = document.createElement("a");
        link.className = "keyword-name";
        link.href = `/g?${one.qs}`;
        link.textContent = one.label;
        const count = document.createElement("span");
        count.className = "keyword-count";
        count.textContent = `${one.pictures} picture${one.pictures === 1 ? "" : "s"}`;
        const form = document.createElement("form");
        form.className = "keyword-rename";
        form.dataset.keywordRename = "";
        const box = document.createElement("input");
        box.type = "text";
        box.dataset.keywordRenameInput = "";
        box.maxLength = 100;
        box.autocomplete = "off";
        box.value = one.label;
        box.setAttribute("aria-label", `rename ${one.label}`);
        const go2 = document.createElement("button");
        go2.type = "submit";
        go2.textContent = "rename";
        form.append(box, go2);
        const forget = document.createElement("button");
        forget.type = "button";
        forget.className = "keyword-forget";
        forget.dataset.forget = one.tag;
        forget.dataset.forgetPictures = String(one.pictures);
        forget.title = `take ${one.label} off all ${one.pictures}`;
        forget.textContent = "forget";
        row.append(link, count, form, forget);
        return row;
      })
    );
  };
  var applied = async (list, told) => {
    if (!told.ok) {
      await say(told.refusal);
      return;
    }
    draw(list, told.data);
  };
  (() => {
    const list = document.querySelector("[data-keywords]");
    if (!(list instanceof HTMLElement)) return;
    list.addEventListener("submit", async (event) => {
      const form = closestFrom(event.target, "[data-keyword-rename]", HTMLElement);
      if (!form) return;
      event.preventDefault();
      const row = closestFrom(form, "[data-keyword]", HTMLElement);
      if (!row) return;
      const to = requireElement(form, "[data-keyword-rename-input]", HTMLInputElement).value.trim();
      if (!to || to === row.querySelector(".keyword-name")?.textContent) return;
      await applied(
        list,
        answered(
          await api.POST("/keywords/rename", { body: { name: requireData(row, "keyword"), to } }),
          "the keyword could not be renamed"
        )
      );
    });
    list.addEventListener("click", async (event) => {
      const button2 = closestFrom(event.target, "[data-forget]", HTMLElement);
      if (!button2) return;
      const name = requireData(button2, "forget");
      const pictures = Number(requireData(button2, "forgetPictures"));
      if (!window.confirm(`Take "${name}" off ${pictures} picture${pictures === 1 ? "" : "s"}? This cannot be undone.`)) {
        return;
      }
      await applied(
        list,
        answered(await api.POST("/keywords/forget", { body: { name, pictures } }), "the keyword could not be forgotten")
      );
    });
  })();

  // src/media.ts
  var asPlaceKind = (held2) => {
    const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"];
    const found = known.find((one) => one === held2);
    if (found === void 0) throw new Error(`the place picker offered ${held2}, which is not a place kind`);
    return found;
  };
  (() => {
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
            const held2 = line.dataset.saidVerdict;
            const { data, error } = await api.POST("/i/{slug}/said/verdict", {
              params: { path: { slug } },
              body: {
                kind: requireData(line, "saidKind"),
                model_id: requireData(line, "saidModel"),
                model_version: requireData(line, "saidVersion"),
                verdict: held2 === wanted ? null : wanted
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
    const people = findElement(document, "[data-people]", HTMLElement);
    if (judged && people) {
      const slug = requireData(judged, "slug");
      for (const deny of everyElement(people, "[data-person-deny]", HTMLElement)) {
        deny.addEventListener("click", async () => {
          const who = requireData(deny, "personDeny");
          const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
            params: { path: { slug, person: who } },
            body: { value: true }
          });
          if (!data) {
            await say(refusal(error, "that was not recorded"));
            return;
          }
          people.replaceChildren();
          for (const [at, one] of data.people.entries()) {
            if (at) people.append(document.createTextNode(" \xB7 "));
            const held2 = document.createElement("span");
            held2.className = "person-said";
            held2.dataset.personSaid = one.slug;
            const link = document.createElement("a");
            link.href = one.href;
            link.dataset.person = one.slug;
            link.textContent = one.name ?? one.slug;
            held2.append(link);
            people.append(held2);
          }
          if (data.people.length === 0) {
            const none = document.createElement("span");
            none.className = "muted";
            none.dataset.peopleNone = "";
            none.textContent = "nobody named here now";
            people.append(none);
          }
        });
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
    const mounted2 = findElement(document, "[data-viewer]", HTMLElement);
    const viewer = mounted2 ? mountViewer(mounted2, (href) => {
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

  // src/frames.ts
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  var num = (value) => typeof value === "number" && Number.isFinite(value);
  var str = (value) => typeof value === "string";
  var numOrNull = (value) => value === null || num(value);
  var strOrNull = (value) => value === null || str(value);
  var dataOrNull = (value) => value === null || isRecord(value);
  function reported(held2) {
    return num(held2.job_id) && num(held2.at) && str(held2.type) && numOrNull(held2.item_id) && strOrNull(held2.phase) && str(held2.severity) && strOrNull(held2.message) && dataOrNull(held2.data) && str(held2.text) && strOrNull(held2.condition);
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
    let held2;
    try {
      held2 = JSON.parse(payload);
    } catch {
      return null;
    }
    if (isEventFrame(held2)) return held2;
    if (isPendingFrame(held2)) return held2;
    if (isBacklogFrame(held2)) return held2;
    return null;
  }

  // src/operations.ts
  (() => {
    const here = findElement(document, "[data-console]", HTMLElement);
    if (!here) return;
    const root = here;
    const ROW_H2 = 24;
    const OVERSCAN = 12;
    const TAPE_COLD = 500;
    const TAPE_PAGE = 2e3;
    const RENDER_DEBOUNCE_MS = 400;
    const held2 = [];
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
    const pad2 = (n, w = 2) => String(n).padStart(w, "0");
    function clock(epoch) {
      const d = new Date(epoch * 1e3);
      return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}.${pad2(d.getMilliseconds(), 3)}`;
    }
    function seconds(v) {
      if (v == null) return "\u2014";
      if (v < 60) return `${v.toFixed(1)}s`;
      if (v < 3600) return `${Math.floor(v / 60)}m ${pad2(Math.floor(v % 60))}s`;
      return `${Math.floor(v / 3600)}h ${pad2(Math.floor(v % 3600 / 60))}m`;
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
      const newest = held2.at(-1);
      if (newest !== void 0 && event.id < newest.id) {
        let i = held2.length;
        while (i > 0) {
          const before = held2[i - 1];
          if (before === void 0 || before.id <= event.id) break;
          i--;
        }
        held2.splice(i, 0, event);
      } else {
        held2.push(event);
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
      for (const e of held2) {
        if (previous !== null && e.id !== previous + 1) found++;
        previous = e.id;
      }
      return found;
    }
    const unfiltered = () => !filter.type && !filter.severity && !filter.job;
    function rebuildView() {
      view = held2.filter(passes);
      const skipped = gaps();
      heldEl.hidden = skipped === 0;
      if (skipped) heldEl.textContent = `${skipped} gap(s) in the held ids \u2014 click a dashed row to fetch`;
      countEl.textContent = `${view.length} of ${held2.length} shown${paused ? ` \xB7 paused, ${heldWhilePaused} new held` : ""}`;
      root.dataset.held = String(held2.length);
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
    function paint2() {
      if (paused) return;
      const total = view.length;
      spacer.style.height = `${total * ROW_H2}px`;
      const top = scroller.scrollTop;
      const first = Math.max(0, Math.floor(top / ROW_H2) - OVERSCAN);
      const last = Math.min(total, Math.ceil((top + scroller.clientHeight) / ROW_H2) + OVERSCAN);
      rows.style.transform = `translateY(${first * ROW_H2}px)`;
      rows.textContent = "";
      const headId = held2.at(-1)?.id;
      let previous = first > 0 ? view[first - 1] : void 0;
      for (const e of view.slice(first, last)) {
        if (previous !== void 0 && unfiltered() && e.id !== previous.id + 1) {
          const after = previous;
          const gap = el(
            "li",
            { class: "tape-gap", role: "button", tabindex: "0" },
            `\u2500\u2500 ${e.id - after.id - 1} event(s) not held between #${after.id} and #${e.id} \u2014 fetch \u2500\u2500`
          );
          gap.addEventListener("click", () => void fill2(after.id, e.id));
          rows.appendChild(gap);
        }
        rows.appendChild(rowFor(e, e.id === headId));
        previous = e;
      }
    }
    function repaint(scrollToEnd) {
      rebuildView();
      if (paused) return;
      paint2();
      if (scrollToEnd && follow.checked) scroller.scrollTop = scroller.scrollHeight;
    }
    function select(e) {
      selectedEvent = e.id;
      rawBody.classList.remove("empty");
      rawBody.textContent = JSON.stringify(e, null, 2);
      for (const li of everyElement(rows, "[data-event]", HTMLLIElement)) {
        li.setAttribute("aria-selected", li.dataset.event === String(e.id) ? "true" : "false");
      }
    }
    scroller.addEventListener("scroll", () => {
      if (!paused) paint2();
    });
    window.addEventListener("resize", () => {
      if (!paused) paint2();
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
    async function fill2(after, before) {
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
      const keep2 = scroller.scrollHeight - scroller.scrollTop;
      for (const e of data.events) ingest(e);
      repaint(false);
      scroller.scrollTop = scroller.scrollHeight - keep2;
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
      const say2 = (selector, text) => {
        const node = findElement(root, selector, HTMLElement);
        if (node) node.textContent = text;
      };
      const heartbeat = o.worker.heartbeat_age != null ? `${o.worker.heartbeat_age.toFixed(1)}s ago` : "none";
      const stalled = !o.worker.enabled && o.queue.queued > 0;
      const condition = o.worker.working ? "working" : stalled ? "stalled" : o.worker.enabled ? "idle" : "off";
      const workerCell = findElement(root, "[data-health-worker]", HTMLElement);
      if (workerCell) workerCell.dataset.workerCondition = condition;
      say2(
        "[data-worker-state]",
        stalled ? `disabled \u2014 ${o.queue.queued} queued, nothing will run` : `${o.worker.enabled ? "enabled" : "disabled"} \xB7 ${o.worker.working ? "working" : "idle"} \xB7 thread ${o.worker.thread_alive ? "alive" : "not running"}`
      );
      say2(
        "[data-worker-raw]",
        `${o.worker.thread || "no thread"} \xB7 ${o.worker.owners.length ? o.worker.owners.join(", ") : "no owner"} \xB7 heartbeat ${heartbeat}`
      );
      say2("[data-queue-state]", `${o.queue.queued} queued \xB7 ${o.queue.running} running`);
      const oldest = o.queue.oldest_queued_age != null ? `${Math.round(o.queue.oldest_queued_age)}s` : "\u2014";
      const settled = Object.entries(o.queue.settled_24h).map(([state, n]) => `${n} ${state}`).join(", ") || "nothing";
      say2("[data-queue-raw]", `oldest queued ${oldest} \xB7 settled 24h ${settled}`);
      say2("[data-ledger-state]", `${o.ledger.events.toLocaleString()} events`);
      say2("[data-ledger-raw]", `head #${o.ledger.last_id} \xB7 job_event \xB7 never sampled`);
      say2("[data-coverage-files]", String(o.coverage.files));
      for (const node of everyElement(document, "[data-missing]", HTMLElement)) {
        const n = o.coverage.missing[requireData(node, "missing")];
        if (n != null) node.textContent = `${n} missing`;
      }
    }
    function paintMatrix(jobs, collections) {
      matrixRows.textContent = "";
      const grouped2 = /* @__PURE__ */ new Set();
      for (const group of collections) for (const id of group.steps) grouped2.add(id);
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
        if (grouped2.has(j.id)) continue;
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
      const load2 = closestFrom(ev.target, "[data-items-load], [data-items-more]", HTMLAnchorElement);
      if (load2) {
        ev.preventDefault();
        void loadItems(load2);
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
        node.textContent = `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${clock(epoch)}`;
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
      const before = held2.at(-1)?.id ?? lastId;
      if (!ingest(frame)) return;
      if (paused) heldWhilePaused++;
      if (frame.id > before + 1 && before > 0) void fill2(before, frame.id);
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

  // src/spelling.ts
  var pad = (n) => String(n).padStart(2, "0");
  function spellDays(root) {
    for (const node of everyElement(root, "time[data-epoch]:not([data-spelled])", HTMLTimeElement)) {
      const d = new Date(Number(requireData(node, "epoch")) * 1e3);
      const day = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
      const domain = node.dataset.domain;
      if (domain === "instant" || domain === "wall") {
        const clock = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
        node.textContent = domain === "instant" ? `${day} ${clock}Z` : `${day} ${clock} wall`;
      } else {
        node.textContent = day;
      }
      node.dataset.spelled = "";
    }
  }

  // src/people.ts
  var denied = (picture, who) => {
    const held2 = document.createElement("div");
    held2.className = "cell-denied";
    held2.dataset.personDenied = picture;
    const what = document.createElement("span");
    what.textContent = "not them";
    held2.append(what);
    const undo = document.createElement("button");
    undo.type = "button";
    undo.className = "link";
    undo.textContent = "undo";
    undo.addEventListener("click", async () => {
      undo.disabled = true;
      const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
        params: { path: { slug: picture, person: who } },
        body: { value: false }
      });
      if (!data) {
        undo.disabled = false;
        await say(refusal(error, "that was not withdrawn"));
        return;
      }
      held2.dataset.personDenied = "";
      held2.dataset.personWithdrawn = picture;
      what.textContent = "withdrawn \u2014 they are named here again only when clustering next says so";
      undo.replaceWith(wayBack(picture));
    });
    held2.append(undo);
    return held2;
  };
  var wayBack = (picture) => {
    const link = document.createElement("a");
    link.href = `/i/${picture}`;
    link.textContent = "open the picture";
    return link;
  };
  (() => {
    spellDays(document);
    new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });
    document.addEventListener("submit", async (event) => {
      const form = closestFrom(event.target, "[data-person-rename]", HTMLFormElement);
      if (!form) return;
      event.preventDefault();
      const slug = requireData(form, "personRename");
      const name = requireElement(form, '[name="name"]', HTMLInputElement).value;
      const { data, error } = await api.POST("/p/{slug}/name", { params: { path: { slug } }, body: { name } });
      if (!data) {
        await say(refusal(error, "that name was refused"));
        return;
      }
      const card = form.closest("[data-unknown]");
      if (card instanceof HTMLElement) {
        card.dataset.named = data.name;
        const named = document.createElement("span");
        named.className = "person-name";
        named.textContent = data.name;
        form.replaceWith(named);
        const heading = document.querySelector("[data-unknown-faces] .analyze-of");
        const left = document.querySelectorAll("[data-unknown]:not([data-named])").length;
        if (heading) heading.textContent = left === 0 ? "all named" : `${left} unnamed`;
        return;
      }
      window.location.replace(`/p/${data.slug}`);
    });
    const grid = document.querySelector("[data-person-pictures]");
    if (grid instanceof HTMLElement) {
      const who = requireData(grid, "personPictures");
      grid.addEventListener("click", async (event) => {
        const button2 = closestFrom(event.target, "[data-person-not-here]", HTMLButtonElement);
        if (!button2) return;
        const shell = closestFrom(button2, "[data-person-picture]", HTMLElement);
        if (!shell) return;
        const picture = requireData(shell, "personPicture");
        button2.disabled = true;
        const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
          params: { path: { slug: picture, person: who } },
          body: { value: true }
        });
        if (!data) {
          button2.disabled = false;
          await say(refusal(error, "that was not recorded"));
          return;
        }
        shell.replaceWith(denied(picture, who));
      });
    }
    const pictures = document.querySelector("[data-person-pictures]");
    if (pictures instanceof HTMLElement) {
      const whose = requireData(pictures, "personPictures");
      pictures.addEventListener("click", async (event) => {
        const button2 = closestFrom(event.target, "[data-person-face]", HTMLButtonElement);
        if (!button2) return;
        const picture = requireData(button2, "personFace");
        const held2 = await api.POST("/p/{slug}/face", {
          params: { path: { slug: whose } },
          body: { file: picture }
        });
        if (held2.error) {
          await say(refusal(held2.error, "that face was not chosen"));
          return;
        }
        for (const face of everyElement(document, ".person-face-big", HTMLImageElement)) {
          face.src = `/avatar/${whose}?chosen=${Date.now()}`;
        }
        button2.dataset.chosen = "";
      });
    }
    const folder = document.querySelector("[data-same-as]");
    if (folder instanceof HTMLElement) {
      const keeping = requireData(folder, "sameAs");
      folder.addEventListener("click", async () => {
        const shelf = await api.GET("/people", { headers: { accept: "application/json" } });
        const others = (shelf.data ?? []).filter((one) => one.slug !== keeping);
        if (others.length === 0) {
          await say("there is nobody else to fold in");
          return;
        }
        const chosen = await askChoice(
          "who is the same person as this one?",
          others.map((one) => ({
            value: one.slug,
            label: one.name ?? one.slug,
            note: `${one.pictures} ${one.pictures === 1 ? "picture" : "pictures"} \xB7 /p/${one.slug}`
          })),
          { detail: "their pictures, names and corrections come here, and their address redirects here afterwards" }
        );
        if (chosen === null) return;
        const { data, error } = await api.POST("/p/{slug}/same-as", {
          params: { path: { slug: keeping } },
          body: { other: chosen }
        });
        if (!data) {
          await say(refusal(error, "those two were not merged"));
          return;
        }
        window.location.replace(`/p/${data.slug}`);
      });
    }
    addressableOverlay({
      root: "[data-drawer-root]",
      trigger: "[data-person]",
      pathPrefix: "/p/"
    });
  })();

  // src/story.ts
  var asProfile = (held2) => {
    if (held2 !== "memory" && held2 !== "technical" && held2 !== "compact") {
      throw new Error(`the page offered the profile ${held2}, which no render speaks`);
    }
    return held2;
  };
  (() => {
    const main = findElement(document, "[data-story-render]", HTMLElement);
    if (!main) return;
    const status = findElement(document, "[data-story-status]", HTMLElement);
    const plan_id = Number(requireData(main, "storyPlan"));
    const locale = requireData(main, "storyLocale");
    for (const button2 of everyElement(document, "[data-story-profile-ask]", HTMLElement)) {
      button2.addEventListener("click", async () => {
        const profile = asProfile(requireData(button2, "storyProfileAsk"));
        if (status) status.textContent = `rendering ${profile}\u2026`;
        const { data, error } = await api.POST("/stories/renders", { body: { plan_id, profile, locale } });
        if (!data) {
          if (status) status.textContent = refusal(error, "that render was refused");
          return;
        }
        window.location.href = `/stories/renders/${data.id}`;
      });
    }
  })();

  // src/jobframes.ts
  function isRecord2(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  var num2 = (value) => typeof value === "number" && Number.isFinite(value);
  var str2 = (value) => typeof value === "string";
  var bool = (value) => typeof value === "boolean";
  var numOrNull2 = (value) => value === null || num2(value);
  var strOrNull2 = (value) => value === null || str2(value);
  function isJob(value) {
    return isRecord2(value) && num2(value.id) && str2(value.kind) && str2(value.state) && bool(value.cancel_requested) && numOrNull2(value.total) && num2(value.done_count) && num2(value.created_at) && numOrNull2(value.finished_at) && strOrNull2(value.derive);
  }
  function isSnapshot(value) {
    return isRecord2(value) && value.type === "snapshot" && Array.isArray(value.jobs) && value.jobs.every(isJob);
  }
  function isDelta(value) {
    return isRecord2(value) && value.type === "delta" && num2(value.job) && str2(value.kind) && str2(value.state) && num2(value.done) && numOrNull2(value.total) && bool(value.cancel_requested) && strOrNull2(value.derive) && strOrNull2(value.doing);
  }
  function decodeJobFrame(payload) {
    if (typeof payload !== "string") return null;
    let held2;
    try {
      held2 = JSON.parse(payload);
    } catch {
      return null;
    }
    if (isSnapshot(held2)) return held2;
    if (isDelta(held2)) return held2;
    return null;
  }

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
        const held2 = data;
        const job = held2?.id;
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
      const held2 = surface()?.dataset.axis;
      if (!held2) return [];
      try {
        const told = JSON.parse(held2);
        return Array.isArray(told) ? told : [];
      } catch {
        return [];
      }
    };
    const timeAtFraction = (fraction, start, end) => {
      const held2 = bands();
      const x = Math.min(W, Math.max(0, fraction * W));
      if (!held2.length) return start + x / W * (end - start);
      for (const one of held2) {
        if (x < one.x1) {
          const drawn2 = one.x1 - one.x0;
          if (drawn2 <= 0) return one.t0;
          return one.t0 + (x - one.x0) / drawn2 * (one.t1 - one.t0);
        }
      }
      return end;
    };
    const read2 = () => {
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
      const qs = new URLSearchParams(read2()?.scope ?? "");
      qs.set("start", String(start));
      qs.set("end", String(end));
      if (snap) qs.set("snap", "true");
      return `/timeline?${qs}`;
    };
    const scopeOf = () => {
      const qs = new URLSearchParams(read2()?.scope ?? "");
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
      const held2 = findElement(swap, "[data-strip]", HTMLElement);
      if (held2) held2.dataset.settling = "";
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
      const held2 = e.state;
      const url = typeof held2 === "object" && held2 !== null && "url" in held2 && typeof held2.url === "string" ? held2.url : location.pathname + location.search;
      void move(url, false);
    });
    swap.addEventListener("click", (e) => {
      const a = closestFrom(e.target, "[data-preset], [data-bin-window], [data-month-window]", HTMLAnchorElement);
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      void move(a.getAttribute("href") ?? location.pathname + location.search, true);
    });
    const REACH2 = 0.025;
    const masses = () => {
      const out = [];
      for (const bar of everyElement(swap, ".overview-bar[data-pictures]", SVGRectElement)) {
        const n = Number(bar.dataset.pictures);
        if (n > 0) out.push({ at: Number(bar.dataset.at), end: Number(bar.dataset.end), weight: Math.sqrt(n) });
      }
      return out;
    };
    const pull = (held2, t, field = masses()) => {
      const reach = REACH2 * (held2.extentEnd - held2.extentStart);
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
    const ox = (held2, t) => (t - held2.extentStart) / Math.max(1, held2.extentEnd - held2.extentStart) * W;
    const ot = (held2, x) => held2.extentStart + Math.min(W, Math.max(0, x)) / W * (held2.extentEnd - held2.extentStart);
    const overviewX = (box, clientX) => (clientX - box.left) / (box.width || 1) * W;
    const placeBrush = (overview, held2, start, end) => {
      const x0 = ox(held2, start);
      const x1 = ox(held2, end);
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
    const handAt = (held2, event) => {
      if (event.clientX >= held2.box.left && event.clientX <= held2.box.right) held2.last = event.clientX;
      return held2.last;
    };
    const dragged = (state, event) => {
      const { held: held2, mode, at } = state;
      const x = overviewX(state.box, handAt(state, event));
      const dt = ot(held2, x) - ot(held2, at);
      const narrowest = Math.min(NARROWEST, held2.extentEnd - held2.extentStart);
      let start = held2.start;
      let end = held2.end;
      const field = masses();
      if (mode === "move") {
        const width = end - start;
        end = pull(held2, Math.min(Math.max(held2.extentStart + width, end + dt), held2.extentEnd), field);
        end = Math.min(held2.extentEnd, Math.max(held2.extentStart + width, end));
        start = end - width;
      } else if (mode === "start") {
        start = Math.max(held2.extentStart, Math.min(pull(held2, start + dt, field), end - narrowest));
      } else if (mode === "end") {
        end = Math.min(held2.extentEnd, Math.max(pull(held2, end + dt, field), start + narrowest));
      } else {
        const a = pull(held2, ot(held2, at), field);
        const b = pull(held2, ot(held2, x), field);
        start = Math.max(held2.extentStart, Math.min(a, b));
        end = Math.min(held2.extentEnd, Math.max(a, b, start + narrowest));
      }
      return { start, end };
    };
    swap.addEventListener("pointerdown", (event) => {
      const overview = closestFrom(event.target, "[data-overview]", SVGSVGElement);
      const held2 = read2();
      if (!overview || !held2) return;
      const box = overview.getBoundingClientRect();
      const x = overviewX(box, event.clientX);
      const x0 = ox(held2, held2.start);
      const x1 = ox(held2, held2.end);
      const grip = 8;
      let mode = "new";
      if (Math.abs(x - x0) <= grip) mode = "start";
      else if (Math.abs(x - x1) <= grip) mode = "end";
      else if (x > x0 && x < x1) mode = "move";
      drag = { overview, box, held: held2, mode, at: x, last: event.clientX };
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
        const held2 = read2();
        if (!held2) return;
        const url = urlFor(Math.round(held2.start), Math.round(held2.end));
        history.replaceState({ url }, "", url);
      });
    };
    let pan = null;
    swap.addEventListener("pointerdown", (event) => {
      const axis = closestFrom(event.target, "[data-strip]", HTMLElement);
      const held2 = read2();
      if (!axis || !held2 || event.button !== 0) return;
      pan = {
        axis,
        px: axis.getBoundingClientRect().width || 1,
        x: event.clientX,
        start: held2.start,
        end: held2.end,
        moved: false,
        held: held2
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
      const held2 = read2();
      if (held2) void move(urlFor(Math.round(held2.start), Math.round(held2.end)), true);
    };
    window.addEventListener("pointerup", unpan);
    window.addEventListener("pointercancel", unpan);
    swap.addEventListener(
      "wheel",
      (e) => {
        const stage = closestFrom(e.target, "[data-strip], [data-overview]", Element);
        const held2 = read2();
        if (!stage || !held2 || !(e.ctrlKey || e.metaKey || e.shiftKey)) return;
        e.preventDefault();
        const width = held2.end - held2.start;
        const box = stage.getBoundingClientRect();
        const at = stage.matches("[data-strip]") || stage.closest("[data-strip]") ? timeAtFraction((e.clientX - box.left) / (box.width || 1), held2.start, held2.end) : held2.start + (e.clientX - box.left) / (box.width || 1) * width;
        let start;
        let end;
        if (e.shiftKey) {
          const step = (e.deltaY > 0 ? 1 : -1) * width / 5;
          start = held2.start + step;
          end = held2.end + step;
        } else {
          const factor = e.deltaY > 0 ? 1.25 : 0.8;
          start = at - (at - held2.start) * factor;
          end = at + (held2.end - at) * factor;
        }
        const narrowest = Math.min(NARROWEST, held2.extentEnd - held2.extentStart);
        if (end - start < narrowest) {
          start = at - narrowest / 2;
          end = at + narrowest / 2;
        }
        start = Math.max(held2.extentStart, start);
        end = Math.min(held2.extentEnd, Math.max(end, start + narrowest));
        live(start, end);
      },
      { passive: false }
    );
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    swap.addEventListener("keydown", (e) => {
      if (!closestFrom(e.target, "[data-overview]", Element)) return;
      const held2 = read2();
      if (!held2) return;
      const width = held2.end - held2.start;
      const step = width / 4;
      const go2 = (s, t) => {
        e.preventDefault();
        void move(urlFor(Math.round(s), Math.round(t)), true);
      };
      if (e.key === "ArrowLeft")
        go2(Math.max(held2.extentStart, held2.start - step), Math.max(held2.extentStart + width, held2.end - step));
      if (e.key === "ArrowRight")
        go2(Math.min(held2.extentEnd - width, held2.start + step), Math.min(held2.extentEnd, held2.end + step));
      if (e.key === "+" || e.key === "=") go2(held2.start + width / 4, held2.end - width / 4);
      if (e.key === "-")
        go2(Math.max(held2.extentStart, held2.start - width / 2), Math.min(held2.extentEnd, held2.end + width / 2));
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
        const strip2 = findElement(seg, "[data-segment-strip]", HTMLElement);
        if (!strip2 || strip2.dataset.filled) continue;
        const box = seg.getBoundingClientRect();
        const cols = Math.max(1, Math.floor(box.width / TILE));
        const rows = Math.max(1, Math.floor(box.height / (TILE + 1)));
        strip2.style.setProperty("--cols", String(cols));
        strip2.style.setProperty("--tile", `${TILE}px`);
        const n = Math.min(TILES_MOST, cols * rows);
        strip2.dataset.filled = String(n);
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
          if (data === void 0 || !strip2.isConnected) return;
          strip2.replaceChildren(
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
      const held2 = read2();
      if (!rail || !held2 || event.button !== 0) return;
      scrub = { held: held2, rail, pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
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
      const held2 = state.held;
      const land = (t) => {
        const end = Math.min(held2.extentEnd, Math.max(held2.extentStart + width, t));
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
      const held2 = read2();
      if (held2) void move(urlFor(Math.round(held2.start), Math.round(held2.end)), true);
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
    const ROW = { least: 120, most: 520, fallback: 200 };
    const rowOf = () => workspace().timelineRow ?? ROW.fallback;
    const sizeRows = (px) => {
      const row = Math.min(ROW.most, Math.max(ROW.least, Math.round(px)));
      const s = surface();
      if (s) s.style.setProperty("--row", `${row}px`);
      remember({ timelineRow: row });
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

  // src/authored.ts
  var draw2 = (root, authored) => {
    requireElement(root, "[data-fav]", HTMLElement).setAttribute("aria-pressed", authored.favorite ? "true" : "false");
    const stars = requireElement(root, "[data-stars]", HTMLElement);
    stars.dataset.rating = String(authored.rating ?? 0);
    for (const star of everyElement(stars, "[data-rate]", HTMLElement)) {
      const n = Number(requireData(star, "rate"));
      if (n > 0) {
        star.setAttribute("aria-pressed", authored.rating !== null && authored.rating >= n ? "true" : "false");
      }
    }
    const tags = requireElement(root, "[data-tags]", HTMLElement);
    tags.replaceChildren(
      ...authored.tags.map((held2) => {
        const chip = document.createElement("span");
        chip.className = "authored-tag";
        chip.dataset.tag = held2.tag;
        const link = document.createElement("a");
        link.href = `/g?f=${encodeURIComponent(`tag:eq:${held2.tag}`)}`;
        link.textContent = held2.label;
        const off = document.createElement("button");
        off.type = "button";
        off.className = "authored-untag";
        off.dataset.untag = held2.tag;
        off.title = `remove ${held2.label}`;
        off.setAttribute("aria-label", `remove ${held2.label}`);
        off.textContent = "\xD7";
        chip.append(link, off);
        return chip;
      })
    );
    const albums = requireElement(root, "[data-albums]", HTMLElement);
    albums.replaceChildren(
      ...authored.collections.map((held2) => {
        const link = document.createElement("a");
        link.href = `/t/${held2.slug}`;
        link.textContent = held2.name;
        return link;
      })
    );
  };
  var mounted = () => [findElement(document, "[data-lightbox]", HTMLElement), findElement(document, "[data-grid]", HTMLElement)].filter(
    (one) => one !== null
  );
  var asked2 = (qs) => {
    const question2 = new URLSearchParams(qs ?? "");
    const one = (name) => question2.get(name);
    const counted2 = (name) => {
      const held2 = question2.get(name);
      return held2 === null ? null : Number(held2);
    };
    return {
      folder: one("folder"),
      album: one("album"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted2("rating_min"),
      q: one("q"),
      f: question2.getAll("f"),
      sort: one("sort"),
      size: counted2("size")
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
    const held2 = surfaces[0]?.dataset.answer ?? "";
    if (!data.in_answer || held2 && data.answer !== held2) {
      window.location.reload();
      return;
    }
    for (const surface of surfaces) {
      surface.dataset.currency = data.currency;
      surface.dataset.answer = data.answer;
    }
  };
  var applied2 = async (root, told) => {
    if (!told.ok) {
      await say(told.refusal);
      return;
    }
    draw2(root, told.data.authored);
    await settle(root);
  };
  var setFavorite = async (root, value) => {
    const told = await api.POST("/i/{slug}/favorite", {
      params: { path: { slug: requireData(root, "slug") } },
      body: { value }
    });
    await applied2(root, answered(told, "the favorite could not be recorded"));
  };
  var setRating = async (root, value) => {
    const told = await api.POST("/i/{slug}/rating", {
      params: { path: { slug: requireData(root, "slug") } },
      body: { value }
    });
    await applied2(root, answered(told, "the rating could not be recorded"));
  };
  var setMembership = async (root, collection, value) => {
    const told = await api.POST("/i/{slug}/collections/{collection}", {
      params: { path: { slug: requireData(root, "slug"), collection } },
      body: { value }
    });
    await applied2(root, answered(told, "the album membership could not be recorded"));
  };
  var setTag = async (root, name, value) => {
    const told = await api.POST("/i/{slug}/tags", {
      params: { path: { slug: requireData(root, "slug") } },
      body: { name, value }
    });
    await applied2(root, answered(told, "the keyword could not be recorded"));
  };
  var suggest = async (root) => {
    const list = requireElement(root, "[data-keyword-list]", HTMLElement);
    if (list.dataset.filled) return;
    list.dataset.filled = "1";
    const told = await api.GET("/g/options", { params: { query: { key: "tag" } } });
    if (told.error || !told.data) return;
    list.replaceChildren(
      ...told.data.options.map((one) => {
        const choice = document.createElement("option");
        choice.value = one.label;
        return choice;
      })
    );
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
      await say(told.refusal);
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
    const off = closestFrom(event.target, "[data-untag]", HTMLElement);
    if (off) {
      void setTag(root, requireData(off, "untag"), false);
      return;
    }
    if (closestFrom(event.target, "[data-album-picker]", HTMLElement)) void choices(root);
  });
  document.addEventListener("submit", (event) => {
    const form = closestFrom(event.target, "[data-tagging]", HTMLElement);
    if (!form) return;
    event.preventDefault();
    const root = closestFrom(form, "[data-authored]", HTMLElement);
    if (!root) return;
    const box = requireElement(form, "[data-tag-input]", HTMLInputElement);
    const name = box.value.trim();
    if (!name) return;
    box.value = "";
    void setTag(root, name, true);
  });
  document.addEventListener("focusin", (event) => {
    const box = closestFrom(event.target, "[data-tag-input]", HTMLInputElement);
    if (!box) return;
    const root = closestFrom(box, "[data-authored]", HTMLElement);
    if (root) void suggest(root);
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
    {
      key: "t",
      by: "authored: keyword",
      run: () => {
        const root = strip();
        if (root) findElement(root, "[data-tag-input]", HTMLInputElement)?.focus();
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
    const question2 = new URLSearchParams(window.location.search);
    const one = (name) => question2.get(name);
    const counted2 = (name) => {
      const held2 = question2.get(name);
      return held2 === null ? null : Number(held2);
    };
    return {
      folder: one("folder"),
      album: one("album"),
      person: one("person"),
      artifact: one("artifact"),
      kind: one("kind"),
      favorite: one("favorite"),
      rating_min: counted2("rating_min"),
      q: one("q"),
      sort: one("sort"),
      size: counted2("size"),
      f: question2.getAll("f")
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
    const draw4 = () => {
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
      const held2 = requireData(mounted2, "answer");
      if (answer !== held2) {
        answer = held2;
        selected.clear();
      }
      draw4();
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
        void say(told2.refusal);
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
      draw4();
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
      draw4();
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
        const kind = asPlaceKind2(requireElement(bar, "[data-bulk-place-kind]", HTMLSelectElement).value);
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
        draw4();
      }
    });
    document.body.addEventListener("htmx:afterSwap", sync);
    sync();
  })();
  function asPlaceKind2(held2) {
    const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"];
    const found = known.find((one) => one === held2);
    if (found === void 0) throw new Error(`the place picker offered ${held2}, which is not a place kind`);
    return found;
  }

  // src/compare.ts
  var MOST = 8;
  function kept() {
    const held2 = workspace().compare;
    return Array.isArray(held2) ? held2.filter((one) => one && typeof one.slug === "string") : [];
  }
  function keep(held2) {
    remember({ compare: held2.slice(-MOST) });
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
  function showComparison(held2) {
    const old = document.querySelector("[data-compare-view]");
    if (old) old.remove();
    if (held2.length < 2) return;
    const sheet = document.createElement("div");
    sheet.className = "compare-view";
    sheet.dataset.compareView = "";
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-label", "comparing");
    const bar = document.createElement("header");
    bar.className = "compare-view-bar";
    const said3 = document.createElement("span");
    said3.className = "compare-view-said";
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
    bar.append(said3, modes, zoom, close);
    const strip2 = document.createElement("div");
    strip2.className = "compare-view-strip";
    for (const [at2, one] of held2.entries()) {
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
      strip2.append(column);
    }
    sheet.append(bar, strip2);
    document.body.append(sheet);
    const glass = { scale: 1, x: 0.5, y: 0.5 };
    const magnify = () => {
      for (const column of everyElement(strip2, "[data-compare-column]", HTMLElement)) {
        const shown = column.querySelector(".compare-frame > *");
        if (!shown) continue;
        shown.style.transformOrigin = `${glass.x * 100}% ${glass.y * 100}%`;
        shown.style.transform = glass.scale === 1 ? "" : `scale(${glass.scale})`;
      }
      strip2.dataset.zoomed = String(glass.scale !== 1);
      zoom.textContent = glass.scale === 1 ? "fit" : `${Math.round(glass.scale * 100)}%`;
      zoom.setAttribute("aria-label", glass.scale === 1 ? "fit" : `zoomed to ${Math.round(glass.scale * 100)}%`);
    };
    const clamp2 = (n) => Math.min(1, Math.max(0, n));
    const zoomTo = (scale, x, y) => {
      glass.scale = Math.min(TRAY_MAX_SCALE, Math.max(1, scale));
      if (glass.scale === 1) {
        glass.x = 0.5;
        glass.y = 0.5;
      } else {
        glass.x = clamp2(x);
        glass.y = clamp2(y);
      }
      magnify();
    };
    const fractionIn = (frame, event) => {
      const box = frame.getBoundingClientRect();
      return { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
    };
    strip2.addEventListener(
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
    strip2.addEventListener("pointerdown", (event) => {
      if (glass.scale === 1) return;
      const frame = closestFrom(event.target, ".compare-frame", HTMLElement);
      if (!frame) return;
      dragging = { x: event.clientX, y: event.clientY, frame };
      frame.setPointerCapture(event.pointerId);
    });
    strip2.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const box = dragging.frame.getBoundingClientRect();
      glass.x = clamp2(glass.x - (event.clientX - dragging.x) / box.width / glass.scale);
      glass.y = clamp2(glass.y - (event.clientY - dragging.y) / box.height / glass.scale);
      dragging = { ...dragging, x: event.clientX, y: event.clientY };
      magnify();
    });
    const letGo = () => {
      dragging = null;
    };
    strip2.addEventListener("pointerup", letGo);
    strip2.addEventListener("pointercancel", letGo);
    strip2.addEventListener("dblclick", () => zoomTo(1, 0.5, 0.5));
    let mode = workspace().compareMode === "flip" ? "flip" : "side";
    let at = 0;
    const columns = () => everyElement(strip2, "[data-compare-column]", HTMLElement);
    const paint2 = () => {
      sheet.dataset.mode = mode;
      const all = columns();
      at = (at % all.length + all.length) % all.length;
      for (const [index, column] of all.entries()) {
        column.hidden = mode === "flip" && index !== at;
        column.dataset.showing = String(mode === "side" || index === at);
      }
      const one = held2[at];
      said3.textContent = mode === "side" ? `${held2.length} side by side` : `${letter(at)} of ${held2.length} \xB7 ${one ? one.name : ""}`;
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
        paint2();
      });
      modes.append(button2);
    }
    const step = (by) => {
      at += by;
      if (mode !== "flip") {
        mode = "flip";
        remember({ compareMode: "flip" });
      }
      paint2();
    };
    paint2();
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
    const held2 = kept();
    const open = workspace().tray !== "closed";
    tray.hidden = held2.length === 0;
    tray.dataset.tray = open ? "open" : "closed";
    const count = findElement(tray, "[data-compare-count]", HTMLElement);
    if (count) count.textContent = String(held2.length);
    const compare = findElement(tray, "[data-compare-open]", HTMLButtonElement);
    if (compare) compare.disabled = held2.length < 2;
    const list = findElement(tray, "[data-compare-items]", HTMLElement);
    if (!list) return;
    list.replaceChildren();
    for (const [at, one] of held2.entries()) {
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
      const held2 = kept();
      keep(held2.some((each) => each.slug === one.slug) ? held2.filter((each) => each.slug !== one.slug) : [...held2, one]);
      remember({ tray: "open" });
      drawTray(tray);
    };
    register([{ key: "c", by: "compare: keep this", run: add }]);
    for (const button2 of root.querySelectorAll("[data-compare-selection]")) {
      button2.addEventListener("click", () => {
        const chosen = picked(root);
        if (chosen.length === 0) return;
        const held2 = kept();
        const fresh = chosen.filter((one) => !held2.some((each) => each.slug === one.slug));
        keep([...held2, ...fresh]);
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
    const said3 = document.createElement("span");
    said3.className = "cell-kind";
    said3.dataset.cellKind = kind ?? "";
    said3.dataset.brokenPicture = "";
    said3.setAttribute("aria-hidden", "true");
    said3.textContent = kind === "audio" ? "audio" : "doc";
    return said3;
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

  // src/spelling-mount.ts
  spellDays(document);
  new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });

  // src/field.ts
  var TINY = 26;
  var PAGE_COVER = 0.62;
  var IN_FLIGHT = 6;
  var TIME_H = 132;
  var TIME_W0 = 3200;
  var TOP_INSET = 62;
  var ROW_H = 260;
  var GAP = 8;
  var VOID_W = 190;
  var CARD_W = 300;
  var CARD_H = 186;
  var clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  var easeOut = (t) => 1 - (1 - t) ** 3;
  var lerp = (a, b, t) => a + (b - a) * t;
  var lerpBox = (a, b, t) => ({
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
    w: lerp(a.w, b.w, t),
    h: lerp(a.h, b.h, t)
  });
  function seeded(slug) {
    let h = 0;
    for (let i = 0; i < slug.length; i++) h = h * 31 + slug.charCodeAt(i) | 0;
    return `hsl(${Math.abs(h) % 360} 14% 22%)`;
  }
  function token(root, name) {
    return getComputedStyle(root).getPropertyValue(name).trim() || "#888";
  }
  function averaged(img) {
    try {
      const board2 = document.createElement("canvas");
      board2.width = 1;
      board2.height = 1;
      const hand = board2.getContext("2d", { willReadFrequently: true });
      if (!hand) return null;
      hand.drawImage(img, 0, 0, 1, 1);
      const [r, g, b] = hand.getImageData(0, 0, 1, 1).data;
      return `rgb(${r ?? 0} ${g ?? 0} ${b ?? 0})`;
    } catch {
      return null;
    }
  }
  function pixels(event) {
    if (event.deltaMode === 1) return event.deltaY * 16;
    if (event.deltaMode === 2) return event.deltaY * window.innerHeight;
    return event.deltaY;
  }
  function spelled2(at, span) {
    const d = new Date(at * 1e3);
    if (span > 86400 * 365 * 4) return String(d.getFullYear());
    if (span > 86400 * 120) return d.toLocaleDateString(void 0, { month: "short", year: "numeric" });
    if (span > 86400 * 3) return d.toLocaleDateString(void 0, { day: "numeric", month: "short" });
    return d.toLocaleTimeString(void 0, { hour: "2-digit", minute: "2-digit" });
  }
  function mountField(root) {
    const found = findElement(root, "[data-field]", HTMLElement);
    const surface = findElement(root, "[data-field-canvas]", HTMLCanvasElement);
    const held2 = findElement(root, "[data-cells]", HTMLElement);
    if (!found || !surface) return;
    const context = surface.getContext("2d");
    if (!context) return;
    const stage = found;
    const board2 = surface;
    const cells = held2;
    const hand = context;
    const chip = findElement(stage, "[data-field-chip]", HTMLElement);
    const sheet = findElement(stage, "[data-field-sheet]", HTMLElement);
    const sheetName = findElement(stage, "[data-field-name]", HTMLElement);
    const sheetWhen = findElement(stage, "[data-field-when]", HTMLElement);
    const sheetOpen = findElement(stage, "[data-field-open]", HTMLAnchorElement);
    const clock = findElement(stage, "[data-field-clock]", HTMLElement);
    const count = findElement(stage, "[data-field-count]", HTMLElement);
    let nodes = [];
    let mode = "rank";
    let cards = [];
    let cam = { x: 0, y: 0, k: 0.2 };
    let bounds = { x: 0, y: 0, w: 1e3, h: 1e3 };
    let hovering = null;
    let page = null;
    let width = 0;
    let height = 0;
    let runs = [];
    let morphAt = 0;
    const MORPH = 620;
    let flight = null;
    let drawing = false;
    let loading = 0;
    let whole = false;
    let total = 0;
    let span = [0, 0];
    let samples = [];
    let stride = 1;
    let covering = null;
    let cut = false;
    let asking = false;
    let asked4 = 0;
    let settled = false;
    function made(key, slug, name, kind, thumb, ar, moment, dated, copies) {
      return {
        key,
        slug,
        name,
        kind,
        thumb,
        ar,
        moment,
        dated,
        copies,
        rank: { x: 0, y: 0, w: 0, h: 0 },
        time: { x: 0, y: 0, w: 0, h: 0 },
        box: { x: 0, y: 0, w: 0, h: 0 },
        from: { x: 0, y: 0, w: 0, h: 0 },
        tint: seeded(slug),
        img: null,
        full: null,
        state: "cold"
      };
    }
    function question2() {
      const asked5 = new URLSearchParams(window.location.search);
      for (const drop of ["page", "size", "view"]) asked5.delete(drop);
      return asked5;
    }
    async function fetchAnswer() {
      const asked5 = question2();
      try {
        const outline = await fetch(`/g/field/shape?${asked5}`, { headers: { accept: "application/json" } });
        if (!outline.ok) return;
        const shape = await outline.json();
        const stamps = shape.samples;
        if (!Array.isArray(stamps) || stamps.length < 1) return;
        span = [stamps[0] ?? 0, stamps[stamps.length - 1] ?? 0];
        total = shape.total ?? 0;
        samples = stamps;
        stride = Math.max(1, shape.stride ?? 1);
        await fetchWindow(span[0], span[1]);
      } catch {
      }
    }
    async function fetchWindow(after, before) {
      if (asking) return;
      asking = true;
      const mine = ++asked4;
      try {
        const wanted = new URLSearchParams(question2());
        wanted.set("after", String(after));
        wanted.set("before", String(before));
        const answer = await fetch(`/g/field/window?${wanted}`, { headers: { accept: "application/json" } });
        if (!answer.ok || mine !== asked4) return;
        const told = await answer.json();
        if (!told || typeof told !== "object") return;
        const held3 = told;
        if (!Array.isArray(held3.items) || !held3.items.length) return;
        nodes = held3.items.map((raw) => {
          const one = raw;
          const slug = one.slug ?? "";
          return made(
            slug,
            slug,
            one.name ?? slug,
            "image",
            one.thumb ?? null,
            one.ar && Number.isFinite(one.ar) ? one.ar : 1,
            typeof one.moment === "number" && Number.isFinite(one.moment) ? one.moment : null,
            one.dated !== false,
            one.copies ?? 1
          );
        });
        whole = true;
        covering = [after, before];
        cut = (held3.more ?? 0) > 0;
        if (count) {
          count.textContent = cut ? `${nodes.length.toLocaleString()} of the ${(held3.held ?? 0).toLocaleString()} here \u2014 zoom in for the rest` : `${nodes.length.toLocaleString()} of ${total.toLocaleString()}`;
          count.hidden = false;
        }
        const first = !settled;
        settled = true;
        layout();
        for (const n of nodes) {
          n.box = mode === "time" ? { ...n.time } : { ...n.rank };
          n.from = { ...n.box };
        }
        if (first) {
          resize();
          fit2(false);
        } else draw4();
      } catch {
      } finally {
        asking = false;
      }
    }
    function ingest() {
      if (whole || !cells) return;
      const seen = new Map(nodes.map((n) => [n.key, n]));
      const held3 = [];
      for (const shell2 of cells.querySelectorAll("[data-selection-key]")) {
        if (!(shell2 instanceof HTMLElement)) continue;
        const key = shell2.dataset.selectionKey;
        if (!key) continue;
        const kept2 = seen.get(key);
        if (kept2) {
          held3.push(kept2);
          continue;
        }
        const link = shell2.querySelector("[data-slug]");
        if (!(link instanceof HTMLElement)) continue;
        const slug = link.dataset.slug ?? "";
        const picture = shell2.querySelector("img");
        const raw = shell2.dataset.moment;
        const moment = raw ? Number(raw) : null;
        held3.push(
          made(
            key,
            slug,
            shell2.querySelector(".cell-name")?.textContent?.trim() ?? slug,
            link.dataset.kind ?? "image",
            picture instanceof HTMLImageElement ? picture.src : null,
            // `--ar` is what the server already computed for the justified
            // grid: the picture's own proportion, or 1 for a file nothing
            // has measured. Reading it keeps one answer to that question.
            Number(getComputedStyle(shell2).getPropertyValue("--ar")) || 1,
            moment !== null && Number.isFinite(moment) ? moment : null,
            shell2.dataset.dated !== "file",
            Number(shell2.querySelector(".cell-copies")?.textContent ?? "1") || 1
          )
        );
      }
      const grew = held3.length !== nodes.length;
      nodes = held3;
      layout();
      if (grew) {
        for (const n of nodes) {
          n.box = mode === "time" ? { ...n.time } : { ...n.rank };
          n.from = { ...n.box };
        }
      }
    }
    function layoutRanked() {
      const area = nodes.reduce((sum, n) => sum + ROW_H * ROW_H * n.ar, 0);
      const shape = Math.max(0.5, width / Math.max(1, height - TOP_INSET));
      const wide = clamp(Math.sqrt(area * shape), ROW_H * 3, ROW_H * 40);
      let x = 0;
      let y = 0;
      let row = [];
      const flush = (last) => {
        if (!row.length) return;
        const used = row.reduce((sum, n) => sum + ROW_H * n.ar, 0);
        const gaps = GAP * (row.length - 1);
        const scale = last ? 1 : (wide - gaps) / used;
        const h = ROW_H * scale;
        let at2 = 0;
        for (const n of row) {
          const w = h * n.ar;
          n.rank = { x: at2, y, w, h };
          at2 += w + GAP;
        }
        y += h + GAP;
        row = [];
        x = 0;
      };
      for (const n of nodes) {
        const w = ROW_H * n.ar;
        if (x > 0 && x + w > wide) flush(false);
        row.push(n);
        x += w + GAP;
      }
      flush(true);
    }
    function layoutTime() {
      const placed = nodes.filter((n) => n.moment !== null);
      if (!placed.length) {
        for (const n of nodes) n.time = { ...n.rank };
        return;
      }
      const first = Math.min(...placed.map((n) => n.moment ?? 0));
      const sorted = [...nodes].sort((a, b) => (a.moment ?? first) - (b.moment ?? first));
      runs = samples.length > 1 ? axisOf(samples, stride) : axisOf(
        sorted.map((n) => n.moment ?? first),
        1
      );
      const lanes = [];
      let r = 0;
      sorted.forEach((n) => {
        const t = n.moment ?? first;
        while (r < runs.length - 1 && t > (runs[r]?.t1 ?? 0)) r += 1;
        const run = runs[r];
        const w = TIME_H * n.ar;
        const x = run ? placed_at(run, t) : 0;
        let lane = lanes.findIndex((edge) => x >= edge);
        if (lane === -1) {
          lane = lanes.length;
          lanes.push(0);
        }
        lanes[lane] = x + w + GAP;
        n.time = { x, y: -lane * (TIME_H + GAP), w, h: TIME_H };
      });
    }
    function placed_at(run, t) {
      const width2 = run.x1 - run.x0;
      const marks = run.at;
      if (marks.length < 2) return run.x0 + width2 / 2;
      let lo = 0;
      let hi = marks.length - 1;
      while (lo < hi) {
        const mid = lo + hi >> 1;
        if ((marks[mid] ?? 0) < t) lo = mid + 1;
        else hi = mid;
      }
      const before = marks[Math.max(0, lo - 1)] ?? t;
      const after = marks[lo] ?? t;
      const inner = after > before ? clamp((t - before) / (after - before), 0, 1) : 0;
      return run.x0 + (Math.max(0, lo - 1) + inner) / (marks.length - 1) * width2;
    }
    function axisOf(times, weight) {
      if (times.length < 2) return [];
      const steps = [];
      for (let i = 1; i < times.length; i++) steps.push((times[i] ?? 0) - (times[i - 1] ?? 0));
      const ordered = [...steps].sort((a, b) => a - b);
      const median = ordered[Math.floor(ordered.length / 2)] ?? 0;
      const wide = Math.max(3600, median * 60);
      const cuts = [];
      let start = times[0] ?? 0;
      let at2 = [start];
      for (let i = 1; i < times.length; i++) {
        const t = times[i] ?? 0;
        if ((steps[i - 1] ?? 0) > wide) {
          cuts.push({ t0: start, t1: times[i - 1] ?? start, count: at2.length, at: at2 });
          start = t;
          at2 = [t];
        } else at2.push(t);
      }
      cuts.push({ t0: start, t1: times[times.length - 1] ?? start, count: at2.length, at: at2 });
      const total2 = cuts.reduce((sum, c) => sum + c.count, 0);
      const content = Math.max(TIME_W0, total2 * weight * TIME_H * 0.42);
      const built = [];
      let x = 0;
      cuts.forEach((c, i) => {
        const w = c.count / total2 * content;
        built.push({ t0: c.t0, t1: c.t1, x0: x, x1: x + w, count: c.count * weight, gapAfter: 0, at: c.at });
        x += w;
        const next = cuts[i + 1];
        if (next) {
          const held3 = built[built.length - 1];
          if (held3) held3.gapAfter = next.t0 - c.t1;
          x += VOID_W;
        }
      });
      return built;
    }
    function timeAt(x) {
      const opening = runs[0];
      if (!opening) return 0;
      if (x <= opening.x0) return opening.t0;
      for (const run of runs) {
        if (x <= run.x1) {
          const width2 = run.x1 - run.x0;
          return width2 <= 0 ? run.t0 : run.t0 + (x - run.x0) / width2 * (run.t1 - run.t0);
        }
        if (run.gapAfter && x <= run.x1 + VOID_W) {
          return run.t1 + (x - run.x1) / VOID_W * run.gapAfter;
        }
      }
      return runs[runs.length - 1]?.t1 ?? 0;
    }
    let refocusing = 0;
    function refocus() {
      if (mode !== "time" || !samples.length || !covering) return;
      window.clearTimeout(refocusing);
      refocusing = window.setTimeout(() => {
        if (!covering) return;
        const t0 = timeAt(cam.x - width / 2 / cam.k);
        const t1 = timeAt(cam.x + width / 2 / cam.k);
        const [c0, c1] = covering;
        const outside = t0 < c0 || t1 > c1;
        const closer = cut && t1 - t0 < (c1 - c0) * 0.6;
        if (outside || closer) void fetchWindow(t0, t1);
      }, 280);
    }
    function layout() {
      layoutRanked();
      layoutTime();
      layoutBoard();
      measure();
    }
    function layoutBoard() {
      for (const card of cards) {
        card.box = { x: card.pin.x, y: card.pin.y, w: CARD_W, h: CARD_H };
      }
    }
    function loadBoard() {
      const held3 = board();
      const seen = new Map(cards.map((c) => [c.pin.id, c]));
      cards = held3.map(
        (one) => seen.get(one.id) ?? {
          pin: one,
          box: { x: one.x, y: one.y, w: CARD_W, h: CARD_H },
          covers: [],
          held: null,
          state: "cold"
        }
      );
      for (const card of cards) {
        const now = held3.find((one) => one.id === card.pin.id);
        if (now) card.pin = now;
      }
      layoutBoard();
      for (const card of cards) void fillCard(card);
    }
    async function fillCard(card) {
      if (card.state !== "cold") return;
      card.state = "loading";
      const show = (src) => {
        const img = new Image();
        img.decoding = "async";
        img.addEventListener("load", () => {
          card.covers.push(img);
          draw4();
        });
        img.src = src;
      };
      if (card.pin.kind === "compare") {
        const [one, two] = card.pin.against ?? ["", ""];
        const held3 = board();
        const left = held3.find((p) => p.id === one);
        const right = held3.find((p) => p.id === two);
        if (!left || !right) {
          card.state = "failed";
          card.held = 0;
          draw4();
          return;
        }
        try {
          const asked5 = new URLSearchParams({ a: left.at, b: right.at });
          const answer = await fetch(`/g/field/against?${asked5}`, { headers: { accept: "application/json" } });
          if (!answer.ok) throw new Error(String(answer.status));
          card.against = await answer.json();
          card.held = card.against.both;
          card.state = "warm";
          const telling = [...card.against.left_only, ...card.against.right_only];
          for (const one2 of (telling.length ? telling : card.against.shared).slice(0, 4)) {
            if (one2.thumb) show(one2.thumb);
          }
          draw4();
        } catch {
          card.state = "failed";
          draw4();
        }
        return;
      }
      if (card.pin.kind === "picture") {
        card.held = 1;
        card.state = "warm";
        show(`/preview/${card.pin.at}`);
        return;
      }
      try {
        const asked5 = new URLSearchParams(card.pin.at);
        asked5.set("after", "0");
        asked5.set("before", "99999999999");
        asked5.set("most", "4");
        const answer = await fetch(`/g/field/window?${asked5}`, { headers: { accept: "application/json" } });
        if (!answer.ok) throw new Error(String(answer.status));
        const told = await answer.json();
        card.held = told.held ?? 0;
        card.state = "warm";
        for (const one of told.items ?? []) if (one.thumb) show(one.thumb);
        draw4();
      } catch {
        card.state = "failed";
        draw4();
      }
    }
    function measure() {
      if (mode === "board") {
        if (!cards.length) {
          bounds = { x: -CARD_W, y: -CARD_H, w: CARD_W * 2, h: CARD_H * 2 };
          return;
        }
        let bx0 = Infinity;
        let by0 = Infinity;
        let bx1 = -Infinity;
        let by1 = -Infinity;
        for (const card of cards) {
          bx0 = Math.min(bx0, card.box.x);
          by0 = Math.min(by0, card.box.y);
          bx1 = Math.max(bx1, card.box.x + card.box.w);
          by1 = Math.max(by1, card.box.y + card.box.h);
        }
        bounds = { x: bx0, y: by0, w: bx1 - bx0, h: by1 - by0 };
        return;
      }
      if (!nodes.length) return;
      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const n of nodes) {
        const b = mode === "time" ? n.time : n.rank;
        x0 = Math.min(x0, b.x);
        y0 = Math.min(y0, b.y);
        x1 = Math.max(x1, b.x + b.w);
        y1 = Math.max(y1, b.y + b.h);
      }
      bounds = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }
    const fitScale = () => Math.min(width / Math.max(1, bounds.w), (height - TOP_INSET) / Math.max(1, bounds.h)) * 0.94;
    function fit2(animate = true) {
      const whole2 = fitScale();
      const tall = (height - TOP_INSET) / Math.max(1, bounds.h) * 0.9;
      const k = mode === "time" ? clamp(tall, whole2, 1.4) : whole2;
      const to = {
        x: mode === "time" ? bounds.x + bounds.w - width / 2 / k + 40 : bounds.x + bounds.w / 2,
        // Centred in the band BELOW the floating controls rather than in
        // the whole box, so fitting never parks the first row under them.
        y: bounds.y + bounds.h / 2 - TOP_INSET / 2 / k,
        k
      };
      if (animate) flyTo(to, 520);
      else {
        cam = to;
        draw4();
      }
    }
    function flyTo(to, ms = 460) {
      flight = { from: { ...cam }, to, at: performance.now(), ms };
      tick();
    }
    function enter(n) {
      const b = n.box;
      const k = Math.min(width / b.w, height / b.h) * 0.86;
      flyTo({ x: b.x + b.w / 2, y: b.y + b.h / 2, k }, 480);
    }
    function anchor() {
      const halfW = width / 2 / cam.k;
      const halfH = height / 2 / cam.k;
      cam.x = clamp(cam.x, bounds.x - halfW / 2, bounds.x + bounds.w + halfW / 2);
      cam.y = clamp(cam.y, bounds.y - halfH / 2, bounds.y + bounds.h + halfH / 2);
    }
    const toWorld = (sx, sy) => ({
      x: (sx - width / 2) / cam.k + cam.x,
      y: (sy - height / 2) / cam.k + cam.y
    });
    function at(sx, sy) {
      const p = toWorld(sx, sy);
      for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n) continue;
        const b = n.box;
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return n;
      }
      return null;
    }
    function want(n) {
      if (n.state !== "cold" || !n.thumb || loading >= IN_FLIGHT) return;
      n.state = "loading";
      loading += 1;
      const img = new Image();
      img.decoding = "async";
      img.addEventListener("load", () => {
        loading -= 1;
        n.img = img;
        n.state = "warm";
        const mean = averaged(img);
        if (mean) n.tint = mean;
        draw4();
      });
      img.addEventListener("error", () => {
        loading -= 1;
        n.state = "failed";
        draw4();
      });
      img.src = n.thumb;
    }
    function wantFull(n) {
      if (n.full || !n.slug) return;
      const img = new Image();
      img.decoding = "async";
      img.addEventListener("load", () => {
        n.full = img;
        draw4();
      });
      img.src = `/preview/${n.slug}`;
    }
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      const rect = board2.getBoundingClientRect();
      width = Math.max(1, Math.round(rect.width));
      height = Math.max(1, Math.round(rect.height));
      board2.width = Math.round(width * dpr);
      board2.height = Math.round(height * dpr);
      hand.setTransform(dpr, 0, 0, dpr, 0, 0);
      hand.imageSmoothingEnabled = true;
      hand.imageSmoothingQuality = "high";
      draw4();
    }
    function draw4() {
      if (drawing) return;
      drawing = true;
      requestAnimationFrame(() => {
        drawing = false;
        paint2();
      });
    }
    function paint2() {
      const ground = token(stage, "--sunken");
      const line = token(stage, "--line");
      const brand = token(stage, "--brand");
      const accent = token(stage, "--accent");
      const faint = token(stage, "--ink-faint");
      const panel2 = token(stage, "--panel");
      const ink = token(stage, "--ink");
      hand.save();
      hand.fillStyle = ground;
      hand.fillRect(0, 0, width, height);
      const k = cam.k;
      const left = cam.x - width / 2 / k;
      const right = cam.x + width / 2 / k;
      const top = cam.y - height / 2 / k;
      const bottom = cam.y + height / 2 / k;
      if (mode === "board") {
        paintBoard(panel2, line, ink, faint, brand, accent);
        paintMinimap(accent, line, panel2, faint);
        hand.restore();
        return;
      }
      if (mode === "time") paintTimeRules(left, right, line, faint, ground);
      const near = [...nodes].sort(
        (a, b) => Math.abs(a.box.x - cam.x) + Math.abs(a.box.y - cam.y) - (Math.abs(b.box.x - cam.x) + Math.abs(b.box.y - cam.y))
      );
      for (const n of near) {
        const b = n.box;
        if (b.x + b.w < left || b.x > right || b.y + b.h < top || b.y > bottom) continue;
        want(n);
      }
      for (const n of nodes) {
        const b = n.box;
        const sx = (b.x - cam.x) * k + width / 2;
        const sy = (b.y - cam.y) * k + height / 2;
        const sw = b.w * k;
        const sh = b.h * k;
        if (sx + sw < -40 || sx > width + 40 || sy + sh < -40 || sy > height + 40) continue;
        if (n.copies > 1 && sw > TINY) {
          hand.fillStyle = n.tint;
          hand.globalAlpha = 0.5;
          for (let i = Math.min(n.copies - 1, 3); i > 0; i--) {
            hand.fillRect(sx + i * 3, sy - i * 3, sw, sh);
          }
          hand.globalAlpha = 1;
        }
        const picture = n === page && n.full ? n.full : n.img;
        if (sw < TINY || !picture) {
          hand.fillStyle = n.tint;
          hand.fillRect(sx, sy, sw, sh);
        } else {
          hand.drawImage(picture, sx, sy, sw, sh);
        }
        if (!n.dated && sw > TINY) {
          hand.fillStyle = accent;
          hand.fillRect(sx, sy + sh - 3, sw, 3);
        }
        if (n === hovering && n !== page) {
          hand.strokeStyle = brand;
          hand.lineWidth = 2;
          hand.strokeRect(sx - 1, sy - 1, sw + 2, sh + 2);
        }
        if (sw > 150 && n !== page) {
          hand.fillStyle = panel2;
          hand.fillRect(sx, sy + sh - 22, sw, 22);
          hand.fillStyle = ink;
          hand.font = "500 12px system-ui, sans-serif";
          hand.textBaseline = "middle";
          const room = sw - 16;
          let text = n.name;
          while (hand.measureText(text).width > room && text.length > 4) text = `${text.slice(0, -5)}\u2026`;
          hand.fillText(text, sx + 8, sy + sh - 11);
        }
      }
      paintMinimap(accent, line, panel2, faint);
      hand.restore();
    }
    function paintTimeRules(left, right, line, faint, ground) {
      if (runs.length < 1) return;
      const first = runs[0]?.t0 ?? 0;
      const span2 = Math.max(1, (runs[runs.length - 1]?.t1 ?? first) - first);
      const sxOf = (wx) => (wx - cam.x) * cam.k + width / 2;
      const chipped = (text, x, y, centred = false) => {
        const w = hand.measureText(text).width + 12;
        const cx = centred ? x - w / 2 : x;
        hand.fillStyle = ground;
        hand.globalAlpha = 0.88;
        hand.beginPath();
        hand.roundRect(cx, y - 3, w, 18, 4);
        hand.fill();
        hand.globalAlpha = 1;
        hand.fillStyle = faint;
        hand.fillText(text, cx + 6, y);
      };
      hand.save();
      hand.font = "500 11px system-ui, sans-serif";
      hand.textBaseline = "top";
      for (const run of runs) {
        if (run.x1 < left || run.x0 > right) continue;
        const x0 = sxOf(run.x0);
        hand.strokeStyle = line;
        hand.lineWidth = 1;
        hand.beginPath();
        hand.moveTo(Math.round(x0) + 0.5, 0);
        hand.lineTo(Math.round(x0) + 0.5, height);
        hand.stroke();
        if (sxOf(run.x1) - x0 > 62) chipped(spelled2(run.t0, span2), x0 + 4, height - 24);
        if (!run.gapAfter) continue;
        const gx = sxOf(run.x1);
        const gw = VOID_W * cam.k;
        if (gx + gw < 0 || gx > width) continue;
        hand.save();
        hand.beginPath();
        hand.rect(gx, 0, gw, height);
        hand.clip();
        hand.strokeStyle = line;
        hand.lineWidth = 1;
        for (let d = -height; d < gw + height; d += 9) {
          hand.beginPath();
          hand.moveTo(gx + d, height);
          hand.lineTo(gx + d + height, 0);
          hand.stroke();
        }
        hand.restore();
        hand.strokeStyle = line;
        hand.setLineDash([4, 4]);
        hand.beginPath();
        hand.moveTo(Math.round(gx) + 0.5, 0);
        hand.lineTo(Math.round(gx) + 0.5, height);
        hand.moveTo(Math.round(gx + gw) + 0.5, 0);
        hand.lineTo(Math.round(gx + gw) + 0.5, height);
        hand.stroke();
        hand.setLineDash([]);
        if (gw > 74) chipped(`${lasted(run.gapAfter)}, nothing`, gx + gw / 2, height / 2, true);
      }
      hand.restore();
    }
    function lasted(seconds) {
      const days = seconds / 86400;
      if (days >= 730) return `${Math.round(days / 365)} years`;
      if (days >= 60) return `${Math.round(days / 30)} months`;
      if (days >= 13) return `${Math.round(days / 7)} weeks`;
      if (days >= 1.5) return `${Math.round(days)} days`;
      const hours = seconds / 3600;
      return hours >= 1.5 ? `${Math.round(hours)} hours` : `${Math.max(1, Math.round(seconds / 60))} minutes`;
    }
    function paintBoard(panel2, line, ink, faint, brand, accent) {
      if (!cards.length) {
        hand.fillStyle = faint;
        hand.font = "500 15px system-ui, sans-serif";
        hand.textAlign = "center";
        hand.textBaseline = "middle";
        hand.fillText("Nothing on the board yet \u2014 open a question and press Pin", width / 2, height / 2);
        hand.textAlign = "left";
        return;
      }
      const k = cam.k;
      for (const card of cards) {
        const b = card.box;
        const sx = (b.x - cam.x) * k + width / 2;
        const sy = (b.y - cam.y) * k + height / 2;
        const sw = b.w * k;
        const sh = b.h * k;
        if (sx + sw < -40 || sx > width + 40 || sy + sh < -40 || sy > height + 40) continue;
        const r = Math.min(14 * k, sh / 2);
        hand.save();
        hand.beginPath();
        hand.roundRect(sx, sy, sw, sh, Math.max(1, r));
        hand.fillStyle = panel2;
        hand.fill();
        hand.strokeStyle = card === holding?.card ? brand : line;
        hand.lineWidth = card === holding?.card ? 2 : 1;
        hand.stroke();
        hand.clip();
        const coverH = sh * 0.7;
        if (card.covers.length) {
          const each = sw / card.covers.length;
          card.covers.forEach((img, i) => {
            hand.drawImage(img, sx + i * each, sy, each, coverH);
          });
        } else {
          hand.fillStyle = token(stage, "--sunken");
          hand.fillRect(sx, sy, sw, coverH);
        }
        if (sw > 120) {
          hand.fillStyle = ink;
          hand.font = `600 ${Math.min(15, Math.max(10, 15 * k))}px system-ui, sans-serif`;
          hand.textBaseline = "top";
          let name = card.pin.name;
          const room = sw - 22;
          while (hand.measureText(name).width > room && name.length > 4) name = `${name.slice(0, -5)}\u2026`;
          hand.fillText(name, sx + 11, sy + coverH + 9 * k);
          hand.fillStyle = faint;
          hand.font = `400 ${Math.min(12.5, Math.max(9, 12.5 * k))}px system-ui, sans-serif`;
          const said3 = card.state === "failed" ? card.pin.kind === "compare" ? "one of the two is gone" : "could not answer" : card.held === null ? "counting\u2026" : card.pin.kind === "compare" && card.against ? card.against.only_left + card.against.only_right === 0 ? `the same ${card.against.both.toLocaleString()} pictures` : `${card.against.both.toLocaleString()} in both \xB7 showing the ${(card.against.only_left + card.against.only_right).toLocaleString()} that differ` : card.pin.kind === "picture" ? "one picture" : `${card.held.toLocaleString()} pictures`;
          hand.fillText(said3, sx + 11, sy + coverH + 28 * k);
          hand.fillStyle = brand;
          hand.font = `600 ${Math.min(10.5, Math.max(8, 10.5 * k))}px system-ui, sans-serif`;
          hand.fillText(
            card.pin.kind.toUpperCase(),
            sx + sw - 11 - hand.measureText(card.pin.kind.toUpperCase()).width,
            sy + coverH + 10 * k
          );
        }
        hand.restore();
        if (card === hoverCard && card !== holding?.card) {
          hand.strokeStyle = accent;
          hand.lineWidth = 2;
          hand.beginPath();
          hand.roundRect(sx - 1, sy - 1, sw + 2, sh + 2, Math.max(1, r));
          hand.stroke();
        }
      }
    }
    function paintMinimap(accent, line, panel2, faint) {
      if (!nodes.length) return;
      const mw = 150;
      const mh = 34;
      const mx = width - mw - 14;
      const my = height - mh - 14;
      hand.save();
      hand.globalAlpha = 0.94;
      hand.fillStyle = panel2;
      hand.beginPath();
      hand.roundRect(mx, my, mw, mh, 6);
      hand.fill();
      hand.strokeStyle = line;
      hand.lineWidth = 1;
      hand.stroke();
      hand.clip();
      const s = Math.min(mw / Math.max(1, bounds.w), mh / Math.max(1, bounds.h)) * 0.84;
      const ox = mx + mw / 2 - (bounds.x + bounds.w / 2) * s;
      const oy = my + mh / 2 - (bounds.y + bounds.h / 2) * s;
      hand.fillStyle = faint;
      for (const n of nodes) {
        const b = n.box;
        hand.fillRect(ox + b.x * s, oy + b.y * s, Math.max(1, b.w * s), Math.max(1, b.h * s));
      }
      hand.strokeStyle = accent;
      hand.lineWidth = 1.5;
      hand.strokeRect(
        ox + (cam.x - width / 2 / cam.k) * s,
        oy + (cam.y - height / 2 / cam.k) * s,
        width / cam.k * s,
        height / cam.k * s
      );
      hand.restore();
    }
    function tick() {
      const now = performance.now();
      let more = false;
      if (morphAt) {
        const t = clamp((now - morphAt) / MORPH, 0, 1);
        const e = easeOut(t);
        for (const n of nodes) n.box = lerpBox(n.from, mode === "time" ? n.time : n.rank, e);
        if (t >= 1) morphAt = 0;
        else more = true;
      }
      if (flight) {
        const t = clamp((now - flight.at) / flight.ms, 0, 1);
        const e = easeOut(t);
        cam = {
          x: lerp(flight.from.x, flight.to.x, e),
          y: lerp(flight.from.y, flight.to.y, e),
          k: flight.from.k * (flight.to.k / flight.from.k) ** e
        };
        if (t >= 1) flight = null;
        else more = true;
      }
      settle2();
      paint2();
      if (more) requestAnimationFrame(tick);
    }
    function settle2() {
      let found2 = null;
      for (const n of nodes) {
        if (n.box.h * cam.k >= height * PAGE_COVER && n.box.w * cam.k >= width * 0.4) {
          found2 = n;
          break;
        }
      }
      if (found2 === page) return;
      page = found2;
      if (sheet) sheet.hidden = page === null;
      stage.dataset.fieldPage = page ? "" : "none";
      if (!page) return;
      wantFull(page);
      if (sheetName) sheetName.textContent = page.name;
      if (sheetOpen) sheetOpen.href = `/i/${page.slug}`;
      if (sheetWhen) {
        const when = page.moment === null ? null : new Date(page.moment * 1e3).toLocaleString(void 0, {
          day: "numeric",
          month: "long",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit"
        });
        sheetWhen.textContent = when === null ? "no time recorded" : page.dated ? when : `${when} \u2014 when the file landed. Nothing has read this one's own date yet.`;
        sheetWhen.dataset.dated = page.dated ? "read" : "file";
      }
    }
    const peekImage = findElement(stage, "[data-field-peek-image]", HTMLImageElement);
    const peekName = findElement(stage, "[data-field-peek-name]", HTMLElement);
    const peekWhen = findElement(stage, "[data-field-peek-when]", HTMLElement);
    function peek(over, x, y) {
      if (!chip) return;
      if (!over || over === page) {
        chip.hidden = true;
        return;
      }
      chip.hidden = false;
      if (peekImage && over.thumb && peekImage.getAttribute("src") !== over.thumb) {
        peekImage.src = over.thumb;
      }
      if (peekName) peekName.textContent = over.name;
      if (peekWhen) {
        peekWhen.textContent = over.moment === null ? "no time recorded" : new Date(over.moment * 1e3).toLocaleString(void 0, {
          day: "numeric",
          month: "short",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit"
        });
        peekWhen.dataset.dated = over.dated ? "read" : "file";
      }
      const box = chip.getBoundingClientRect();
      const left = x + 18 + box.width > width ? x - 18 - box.width : x + 18;
      const top = y + 18 + box.height > height ? y - 18 - box.height : y + 18;
      chip.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
    }
    let dragging = false;
    let moved = 0;
    let lastX = 0;
    let lastY = 0;
    let hoverCard = null;
    let holding = null;
    function cardUnder(sx, sy, except) {
      const p = toWorld(sx, sy);
      for (let i = cards.length - 1; i >= 0; i--) {
        const card = cards[i];
        if (!card || card === except) continue;
        const b = card.box;
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return card;
      }
      return null;
    }
    function cardAt(sx, sy) {
      const p = toWorld(sx, sy);
      for (let i = cards.length - 1; i >= 0; i--) {
        const card = cards[i];
        if (!card) continue;
        const b = card.box;
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return card;
      }
      return null;
    }
    function openCard(card) {
      window.location.href = card.pin.kind === "picture" ? `/i/${card.pin.at}` : `/field${card.pin.at ? `?${card.pin.at}` : ""}`;
    }
    board2.addEventListener("pointerdown", (event) => {
      dragging = true;
      moved = 0;
      lastX = event.clientX;
      lastY = event.clientY;
      board2.setPointerCapture(event.pointerId);
      board2.style.cursor = "grabbing";
      if (mode === "board") {
        const rect = board2.getBoundingClientRect();
        const under = cardAt(event.clientX - rect.left, event.clientY - rect.top);
        if (under) {
          const p = toWorld(event.clientX - rect.left, event.clientY - rect.top);
          holding = { card: under, dx: p.x - under.box.x, dy: p.y - under.box.y };
          cards = [...cards.filter((c) => c !== under), under];
        }
      }
      if (event.pointerType !== "mouse") {
        const rect = board2.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        hovering = at(x, y);
        peek(hovering, x, y);
        draw4();
      }
    });
    board2.addEventListener("pointermove", (event) => {
      const rect = board2.getBoundingClientRect();
      if (holding && dragging) {
        const p = toWorld(event.clientX - rect.left, event.clientY - rect.top);
        moved += Math.abs(event.clientX - lastX) + Math.abs(event.clientY - lastY);
        lastX = event.clientX;
        lastY = event.clientY;
        holding.card.box.x = p.x - holding.dx;
        holding.card.box.y = p.y - holding.dy;
        hoverCard = cardUnder(event.clientX - rect.left, event.clientY - rect.top, holding.card);
        draw4();
        return;
      }
      if (mode === "board" && !dragging) {
        const over2 = cardAt(event.clientX - rect.left, event.clientY - rect.top);
        if (over2 !== hoverCard) {
          hoverCard = over2;
          board2.style.cursor = over2 ? "pointer" : "grab";
          draw4();
        }
        return;
      }
      if (dragging) {
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);
        cam.x -= dx / cam.k;
        cam.y -= dy / cam.k;
        anchor();
        lastX = event.clientX;
        lastY = event.clientY;
        flight = null;
        settle2();
        refocus();
        draw4();
        return;
      }
      const over = at(event.clientX - rect.left, event.clientY - rect.top);
      if (over !== hovering) {
        hovering = over;
        board2.style.cursor = over ? "pointer" : "grab";
        draw4();
      }
      peek(hovering, event.clientX - rect.left, event.clientY - rect.top);
    });
    const release = (event) => {
      if (!dragging) return;
      dragging = false;
      board2.style.cursor = hovering ? "pointer" : "grab";
      if (event.pointerType !== "mouse") {
        hovering = null;
        peek(null, 0, 0);
        draw4();
      }
      if (holding) {
        const moving = holding;
        holding = null;
        if (moved > 6) {
          const where = board2.getBoundingClientRect();
          const onto = cardUnder(event.clientX - where.left, event.clientY - where.top, moving.card);
          if (onto && onto.pin.kind !== "compare" && moving.card.pin.kind !== "compare") {
            pin({
              id: `pin-${Date.now().toString(36)}`,
              kind: "compare",
              name: `${moving.card.pin.name} vs ${onto.pin.name}`,
              at: "",
              against: [moving.card.pin.id, onto.pin.id],
              x: Math.round((moving.card.pin.x + onto.pin.x) / 2),
              y: Math.round(Math.max(moving.card.pin.y, onto.pin.y) + CARD_H + 40)
            });
            moving.card.box.x = moving.card.pin.x;
            moving.card.box.y = moving.card.pin.y;
            hoverCard = null;
            loadBoard();
            measure();
            fit2();
            tally();
            return;
          }
          pin({ ...moving.card.pin, x: Math.round(moving.card.box.x), y: Math.round(moving.card.box.y) });
          moving.card.pin = { ...moving.card.pin, x: moving.card.box.x, y: moving.card.box.y };
          measure();
          draw4();
          return;
        }
        openCard(moving.card);
        return;
      }
      if (moved > 6) return;
      const rect = board2.getBoundingClientRect();
      const hit = at(event.clientX - rect.left, event.clientY - rect.top);
      if (hit) enter(hit);
      else if (page) fit2();
    };
    board2.addEventListener("pointerup", release);
    board2.addEventListener("pointercancel", () => {
      dragging = false;
      moved = 0;
      board2.style.cursor = hovering ? "pointer" : "grab";
    });
    board2.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const rect = board2.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        const before = toWorld(mx, my);
        const by = Math.exp(-pixels(event) * (event.ctrlKey ? 0.01 : 22e-4));
        cam.k = clamp(cam.k * by, fitScale(), 14);
        const after = toWorld(mx, my);
        cam.x += before.x - after.x;
        cam.y += before.y - after.y;
        anchor();
        flight = null;
        settle2();
        refocus();
        draw4();
      },
      { passive: false }
    );
    const shell = stage.parentElement;
    function arrange(wanted, button2) {
      for (const other of stage.querySelectorAll("[data-field-mode]")) {
        other.setAttribute("aria-pressed", String(other === button2));
      }
      if (shell) shell.dataset.arrangement = wanted;
      if (wanted === "grid") return;
      if (wanted !== mode) {
        for (const n of nodes) n.from = { ...n.box };
        mode = wanted;
        if (mode === "board") loadBoard();
        if (count) {
          if (mode === "board") {
            const held3 = board().length;
            count.textContent = held3 ? `${held3} on the board` : "nothing on the board yet";
            count.hidden = false;
          } else if (covering) {
            count.textContent = cut ? `${nodes.length.toLocaleString()} of the pictures here \u2014 zoom in for the rest` : `${nodes.length.toLocaleString()} of ${total.toLocaleString()}`;
          }
        }
        measure();
        morphAt = mode === "board" ? 0 : performance.now();
        stage.dataset.fieldMode = mode;
      }
      resize();
      fit2();
      tick();
    }
    stage.addEventListener("click", (event) => {
      const button2 = closestFrom(event.target, "[data-field-mode]", HTMLButtonElement);
      if (button2) {
        const said3 = button2.dataset.fieldMode;
        arrange(said3 === "time" ? "time" : said3 === "grid" ? "grid" : said3 === "board" ? "board" : "rank", button2);
        return;
      }
      if (closestFrom(event.target, "[data-field-fit]", HTMLButtonElement)) fit2();
      if (closestFrom(event.target, "[data-field-pin]", HTMLButtonElement)) {
        const asked5 = question2().toString();
        const said3 = new URLSearchParams(asked5).get("q");
        const already = board().find((one) => one.at === asked5);
        if (already) {
          unpin(already.id);
          if (mode === "board") {
            loadBoard();
            measure();
            draw4();
            tally();
          } else say2("taken off the board");
          return;
        }
        const held3 = board();
        pin({
          id: `pin-${Date.now().toString(36)}`,
          kind: "query",
          name: said3 || "The whole library",
          at: asked5,
          x: held3.length % 4 * (CARD_W + 40),
          y: Math.floor(held3.length / 4) * (CARD_H + 40)
        });
        if (mode === "board") {
          loadBoard();
          measure();
          draw4();
          tally();
        } else say2("on the board");
      }
    });
    function tally() {
      if (!count) return;
      const held3 = board().length;
      count.textContent = held3 ? `${held3} on the board` : "nothing on the board yet";
      count.hidden = false;
    }
    function say2(what) {
      if (!count) return;
      const was = count.textContent;
      const hidden = count.hidden;
      count.textContent = what;
      count.hidden = false;
      window.setTimeout(() => {
        count.textContent = was;
        count.hidden = hidden;
      }, 1800);
    }
    const zoomed = (by) => flyTo({ ...cam, k: clamp(cam.k * by, fitScale(), 14) }, 240);
    register([
      { key: "z", by: "the field: show all of them", run: () => fit2() },
      { key: "+", by: "the field: closer", run: () => zoomed(1.6) },
      { key: "=", by: "the field: closer", run: () => zoomed(1.6) },
      { key: "-", by: "the field: further back", run: () => zoomed(1 / 1.6) }
    ]);
    if (cells) {
      new MutationObserver(() => {
        ingest();
        draw4();
      }).observe(cells, { childList: true, subtree: false });
    }
    new ResizeObserver(() => resize()).observe(board2);
    let watching = null;
    const density = () => {
      watching?.removeEventListener("change", density);
      watching = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      watching.addEventListener("change", density);
      resize();
    };
    density();
    stage.hidden = false;
    if (shell) shell.dataset.arrangement = mode;
    ingest();
    stage.dataset.fieldMode = mode;
    resize();
    fit2(false);
    void fetchAnswer();
    if (clock) clock.hidden = true;
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
          const go2 = document.createElement("button");
          go2.type = "button";
          go2.className = "link";
          go2.textContent = "reload";
          go2.addEventListener("click", () => reg.waiting?.postMessage({ type: "SKIP_WAITING" }));
          notice.append(go2);
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

  // src/panes.ts
  var htmxOf = () => {
    const held2 = window.htmx;
    return held2 && typeof held2.process === "function" ? held2 : null;
  };
  function mountPanes(root) {
    const found = findElement(root, "[data-panes]", HTMLElement);
    if (!found) return;
    const deck = found;
    const open = [];
    function frame(title, mode, href) {
      const pane = document.createElement("aside");
      pane.className = "pane";
      pane.dataset.paneMode = mode;
      pane.setAttribute("role", mode === "overlay" ? "dialog" : "region");
      pane.setAttribute("aria-label", title);
      if (mode === "overlay") pane.setAttribute("aria-modal", "true");
      pane.innerHTML = `
      <header class="pane-bar" data-pane-bar>
        <b class="pane-title"></b>
        <span class="pane-modes" role="group" aria-label="how to show this">
          <button type="button" data-pane-mode-set="dock" title="beside the canvas">Dock</button>
          <button type="button" data-pane-mode-set="overlay" title="over the canvas">Overlay</button>
          <button type="button" data-pane-mode-set="window" title="as a movable window">Window</button>
        </span>
        <a class="pane-away" data-pane-away>Open as a page</a>
        <button type="button" class="pane-shut" data-pane-shut aria-label="close">&times;</button>
      </header>
      <div class="pane-body" data-pane-body>
        <p class="pane-waiting">fetching&hellip;</p>
      </div>`;
      const named2 = pane.querySelector(".pane-title");
      if (named2) named2.textContent = title;
      const away = pane.querySelector("[data-pane-away]");
      if (away instanceof HTMLAnchorElement) away.href = href;
      return pane;
    }
    async function show(href, title, mode) {
      const pane = frame(title, mode, href);
      deck.append(pane);
      open.push(pane);
      requestAnimationFrame(() => {
        pane.setAttribute("data-pane-in", "");
        settle2();
      });
      const body = pane.querySelector("[data-pane-body]");
      if (!(body instanceof HTMLElement)) return;
      try {
        const answer = await fetch(href, { headers: { accept: "text/html" } });
        if (!answer.ok) throw new Error(`${answer.status}`);
        const told = new DOMParser().parseFromString(await answer.text(), "text/html");
        const stage = told.querySelector("main.stage");
        if (!stage) throw new Error("that surface has no stage to show");
        body.replaceChildren(document.importNode(stage, true));
        htmxOf()?.process(body);
        spellDays(body);
        const wanted = href.includes("#") ? href.slice(href.indexOf("#") + 1) : "";
        if (wanted) {
          const part = body.querySelector(`#${CSS.escape(wanted)}`);
          if (part instanceof HTMLElement) part.scrollIntoView({ block: "start" });
        }
        offerPins(body);
      } catch (why) {
        body.replaceChildren(said3(why, href));
      }
    }
    function said3(why, href) {
      const told = document.createElement("p");
      told.className = "pane-waiting";
      told.textContent = `could not open this here \u2014 ${why instanceof Error ? why.message : "it did not answer"}. `;
      const link = document.createElement("a");
      link.href = href;
      link.textContent = "open it as a page instead";
      told.append(link);
      return told;
    }
    function asPin(href) {
      const path = href.replace(/^https?:\/\/[^/]+/, "").split(/[?#]/)[0] ?? "";
      const slug = decodeURIComponent(path.slice(3));
      if (!slug) return null;
      if (path.startsWith("/p/")) return { kind: "person", at: `person=${encodeURIComponent(slug)}` };
      if (path.startsWith("/t/")) return { kind: "album", at: `album=${encodeURIComponent(slug)}` };
      if (path.startsWith("/f/")) return { kind: "folder", at: `folder=${encodeURIComponent(slug)}` };
      if (path.startsWith("/i/")) return { kind: "picture", at: slug };
      return null;
    }
    function named(link, kind) {
      const own = link.querySelector('[class$="-name"]');
      const said4 = own?.textContent?.trim();
      if (said4) return said4;
      const visible = [...link.childNodes].filter((one) => !(one instanceof Element && one.getAttribute("aria-hidden") === "true")).map((one) => one.textContent ?? "").join(" ");
      const first = visible.split("\n").map((one) => one.trim()).find((one) => one.length > 0);
      if (first && !/^[\d,]+\s+pictures?$/i.test(first)) return first;
      return kind === "person" ? "Someone not named yet" : first ?? "";
    }
    function offerPins(body) {
      const held3 = new Set(board().map((one) => one.at));
      for (const link of body.querySelectorAll("a[href]")) {
        if (!(link instanceof HTMLAnchorElement)) continue;
        if (link.dataset.pinOffered !== void 0) continue;
        const what = asPin(link.getAttribute("href") ?? "");
        if (!what) continue;
        link.dataset.pinOffered = "";
        const button2 = document.createElement("button");
        button2.type = "button";
        button2.className = "pin-offer";
        button2.dataset.pinAt = what.at;
        button2.dataset.pinKind = what.kind;
        button2.dataset.pinName = named(link, what.kind);
        button2.textContent = held3.has(what.at) ? "on the board" : "pin";
        button2.title = "keep this on the board";
        link.insertAdjacentElement("afterend", button2);
      }
    }
    function shut(pane) {
      pane.removeAttribute("data-pane-in");
      const at = open.indexOf(pane);
      if (at >= 0) open.splice(at, 1);
      settle2();
      window.setTimeout(() => pane.remove(), 260);
    }
    function settle2() {
      const docked = open.filter((p) => p.dataset.paneMode === "dock" && p.hasAttribute("data-pane-in"));
      document.body.dataset.panesDocked = docked.length ? "yes" : "no";
      document.body.dataset.panesOpen = open.length ? "yes" : "no";
    }
    deck.addEventListener("click", (event) => {
      const pane = closestFrom(event.target, ".pane", HTMLElement);
      if (!pane) return;
      if (closestFrom(event.target, "[data-pane-shut]", HTMLButtonElement)) {
        shut(pane);
        return;
      }
      const offer = closestFrom(event.target, "[data-pin-at]", HTMLButtonElement);
      if (offer) {
        const at = offer.dataset.pinAt ?? "";
        const already = board().find((one) => one.at === at);
        if (already) {
          unpin(already.id);
          offer.textContent = "pin";
        } else {
          const count = board().length;
          pin({
            id: `pin-${Date.now().toString(36)}`,
            kind: offer.dataset.pinKind ?? "person",
            name: offer.dataset.pinName ?? at,
            at,
            x: count % 4 * 340,
            y: Math.floor(count / 4) * 226
          });
          offer.textContent = "on the board";
        }
        return;
      }
      const wanted = closestFrom(event.target, "[data-pane-mode-set]", HTMLButtonElement);
      if (wanted) {
        pane.dataset.paneMode = wanted.dataset.paneModeSet ?? "overlay";
        pane.setAttribute("role", pane.dataset.paneMode === "overlay" ? "dialog" : "region");
        settle2();
      }
    });
    document.addEventListener("click", (event) => {
      const asked4 = closestFrom(event.target, "[data-pane-open]", HTMLElement);
      if (!asked4) return;
      const href = asked4.dataset.paneOpen;
      if (!href) return;
      if (event instanceof MouseEvent && (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey)) return;
      event.preventDefault();
      const mode = asked4.dataset.paneMode;
      void show(
        href,
        asked4.dataset.paneTitle ?? asked4.textContent?.trim() ?? "",
        mode === "dock" ? "dock" : mode === "window" ? "window" : "overlay"
      );
    });
    let held2 = null;
    let fromX = 0;
    let fromY = 0;
    let atX = 0;
    let atY = 0;
    deck.addEventListener("pointerdown", (event) => {
      const bar = closestFrom(event.target, "[data-pane-bar]", HTMLElement);
      const pane = bar && closestFrom(event.target, ".pane", HTMLElement);
      if (!bar || !pane || pane.dataset.paneMode !== "window") return;
      if (closestFrom(event.target, "button, a", HTMLElement)) return;
      held2 = pane;
      fromX = event.clientX;
      fromY = event.clientY;
      atX = Number(pane.dataset.paneX ?? 0);
      atY = Number(pane.dataset.paneY ?? 0);
      bar.setPointerCapture(event.pointerId);
    });
    deck.addEventListener("pointermove", (event) => {
      if (!held2) return;
      const x = atX + event.clientX - fromX;
      const y = atY + event.clientY - fromY;
      held2.dataset.paneX = String(x);
      held2.dataset.paneY = String(y);
      held2.style.translate = `${x}px ${y}px`;
    });
    const drop = () => {
      held2 = null;
    };
    deck.addEventListener("pointerup", drop);
    deck.addEventListener("pointercancel", drop);
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !open.length) return;
      const top = open[open.length - 1];
      if (!top) return;
      event.preventDefault();
      shut(top);
    });
    settle2();
  }

  // src/shortcuts.ts
  var SPELLED = {
    ArrowLeft: "\u2190",
    ArrowRight: "\u2192",
    ArrowUp: "\u2191",
    ArrowDown: "\u2193",
    " ": "Space",
    Escape: "Esc"
  };
  var spell2 = (key) => SPELLED[key] ?? (key.length === 1 ? key.toUpperCase() : key);
  function grouped() {
    const groups = /* @__PURE__ */ new Map();
    for (const { key, by } of registered()) {
      const cut = by.indexOf(":");
      const where = cut === -1 ? "everywhere" : by.slice(0, cut).trim();
      const does = cut === -1 ? by : by.slice(cut + 1).trim();
      const held2 = groups.get(where) ?? [];
      held2.push({ key: spell2(key), does });
      groups.set(where, held2);
    }
    return groups;
  }
  function draw3(body) {
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
        const said3 = document.createElement("dd");
        said3.textContent = does;
        list.append(term, said3);
      }
      section.append(list);
      body.append(section);
    }
  }
  function showShortcuts() {
    void panel("what the keyboard does", draw3);
  }
  function mountShortcuts(root) {
    register([{ key: "?", by: "what the keyboard does", run: showShortcuts }]);
    for (const button2 of root.querySelectorAll("[data-shortcuts-open]")) {
      button2.addEventListener("click", showShortcuts);
    }
  }

  // src/entries/app.ts
  mountInstall();
  mountServiceWorker();
  mountShortcuts(document);
  mountField(document.body);
  mountPanes(document.body);
})();
//# sourceMappingURL=app.js.map
