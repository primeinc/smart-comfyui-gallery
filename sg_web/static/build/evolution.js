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
      const metric = (label, v, why) => `<dt>${label}</dt><dd title="${esc(why)}">${pct(v)}${v === null ? ` <small>${esc(why)}</small>` : ""}</dd>`;
      function sequence() {
        const strip = view.phases.map(
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
        const body = rows.map(([label, get, why]) => {
          const cells = view.members.map((m, i) => {
            const t = transitionTo.get(m.ref);
            if (i === 0 || t === void 0) return "<td>\xB7</td>";
            const v = get(t);
            const cls = (t.phase_boundary ? "boundary " : "") + (v === null ? "unavailable" : "");
            return `<td class="${cls}" title="${esc(why(t))}">${pct(v)}</td>`;
          });
          return `<tr><th>${label}</th>${cells.join("")}</tr>`;
        }).join("");
        const facts = (generated ? SEQUENCE_FACTS : []).map(
          (key) => `<tr><th>${key}</th>${view.members.map((m) => {
            const boundary = transitionTo.get(m.ref)?.phase_boundary ?? false;
            return `<td class="${boundary ? "boundary" : ""}">${spell(m.generation[key])}</td>`;
          }).join("")}</tr>`
        ).join("");
        main.innerHTML = `<div class="filmstrip">${strip}</div><div class="tracks"><table>${head}${body}${facts}</table></div>`;
      }
      function drift() {
        const W = 420;
        const H = 320;
        const pad = 36;
        const drawn = view.transitions.filter(
          (t) => t.prompt_cosine !== null && t.visual_cosine !== null
        );
        const dots = drawn.map((t) => {
          const x = pad + (1 - Math.max(0, t.prompt_cosine)) * (W - 2 * pad);
          const y = H - pad - (1 - Math.max(0, t.visual_cosine)) * (H - 2 * pad);
          return `<circle data-pair="${esc(t.before)}|${esc(t.after)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${t.phase_boundary ? "#fc6" : "#6cf"}"><title>${esc(t.before)} \u2192 ${esc(t.after)}: prompt ${pct(t.prompt_cosine)}, image ${pct(t.visual_cosine)}</title></circle>`;
        }).join("");
        const missing = view.transitions.length - drawn.length;
        main.innerHTML = `<div class="drift"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#555"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#555"/>
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
          const held = children.get(e.parent);
          if (held === void 0) children.set(e.parent, [e.child]);
          else held.push(e.child);
        }
        const kindOf = new Map(view.lineage.map((e) => [`${e.parent}|${e.child}`, e.kind]));
        const isChild = new Set(view.lineage.map((e) => e.child));
        const roots = [...new Set(view.lineage.map((e) => e.parent))].filter((p) => !isChild.has(p));
        const node = (ref, kind) => {
          const m = members.get(ref);
          const label = m === void 0 ? `<span class="kind">outside the session</span> ${esc(ref.slice(0, 8))}` : `${thumb(m)} ${esc(ref)}`;
          const kids = (children.get(ref) ?? []).map((child) => node(child, kindOf.get(`${ref}|${child}`) ?? null)).join("");
          return `<li>${label}${kind === null ? "" : ` <span class="kind">${esc(kind)}</span>`}${kids ? `<ul>${kids}</ul>` : ""}</li>`;
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
      function panel() {
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
      function draw() {
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
        panel();
        inspect();
      }
      root.addEventListener("click", (event) => {
        const chosen = closestFrom(event.target, "[data-tab]", HTMLElement);
        if (chosen !== null) {
          const name = requireData(chosen, "tab");
          if (isTab(name)) tab = name;
          draw();
          return;
        }
        const dot = closestFrom(event.target, "[data-pair]", Element);
        if (dot !== null) {
          const [before, after] = requireAttribute(dot, "data-pair").split("|");
          if (before !== void 0 && after !== void 0) {
            pair = [before, after];
            selected = after;
          }
          draw();
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
        draw();
      });
      draw();
    }
  })();
  function requireAttribute(node, name) {
    const held = node.getAttribute(name);
    if (held === null) throw new Error(`expected a ${name} on ${node.tagName.toLowerCase()}`);
    return held;
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
})();
//# sourceMappingURL=evolution.js.map
