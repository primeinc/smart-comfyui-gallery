# The contract between the application and the browser.
#
# Python states each JSON shape once, as a pydantic model beside the route it
# belongs to. Litestar renders those into OpenAPI 3.1. openapi-typescript
# turns that document into the browser's types. Nothing on the TypeScript
# side of this seam is written by hand, so the two halves cannot drift into
# describing different things while both type checking.
#
#   sg_web/*.py  ->  openapi.json  ->  frontend/src/generated/api.d.ts

set script-interpreter := ['bash', '-euo', 'pipefail']

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere.
# Exported rather than interpolated so the body reads it as a value.
export PYTHON := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The application's OpenAPI document, written the way litestar's own
# `schema openapi` command writes it (litestar/cli/commands/schema.py
# _generate_openapi_schema): its serializer, its formatting. Built over a
# throwaway home, so the document describes the application and never what
# happens to be in ~/.smartgallery.
#
# The destination is an environment variable rather than a hard-coded
# path, because the gate needs the same document written somewhere it can
# compare WITHOUT overwriting the committed one.
[private]
[script]
write-schema:
    "$PYTHON" - <<'PY'
    import os
    import pathlib
    import tempfile

    import msgspec
    from litestar.serialization import encode_json, get_serializer

    from sg_web.app import build_app

    with tempfile.TemporaryDirectory() as scratch:
        app = build_app(scratch, worker=False)
        serializer = get_serializer(app.type_encoders)
        document = msgspec.json.format(
            encode_json(app.openapi_schema.to_schema(), serializer=serializer), indent=4
        )
    pathlib.Path(os.environ["SCHEMA_OUT"]).write_bytes(document + b"\n")
    PY

# The document, into the tree, for `just api types`.
schema:
    SCHEMA_OUT=frontend/openapi.json just api write-schema

# The browser's view of that document. Generated, never edited.
types: schema
    npm run --silent api-types

# The gate, in two halves that together pin the whole chain, and they run
# TOGETHER.
#
# The document is regenerated from the running application and compared
# with the committed one, so a handler whose model changed without this
# being run fails here. The types are checked with openapi-typescript's
# own `--check`, which compares without rewriting: the browser cannot be
# typed against a contract the server stopped honouring, and a passing
# check leaves the tree exactly as it found it.
#
# The halves were serial for one reason: the first REWROTE the file the
# second reads, so running them together raced on a truncated document.
# Writing to a temporary file removes the reason, halves the wall clock,
# and means the gate no longer edits the tree it is judging.
[parallel]
check: schema-fresh types-fresh

[private]
[script]
schema-fresh:
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
    SCHEMA_OUT="$scratch/openapi.json" just api write-schema
    if ! cmp -s "$scratch/openapi.json" frontend/openapi.json; then
      echo 'the OpenAPI document is stale: run `just api types` and commit the result' >&2
      diff --unified=2 frontend/openapi.json "$scratch/openapi.json" >&2 || true
      exit 1
    fi

[private]
types-fresh:
    npm run --silent api-types-check
