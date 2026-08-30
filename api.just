# The contract between the application and the browser:
#
#   sg_web/*.py  ->  openapi.json  ->  frontend/src/generated/api.d.ts

# Python states each JSON shape once beside its route, litestar renders
# OpenAPI 3.1, openapi-typescript generates the browser's types. Nothing on
# the TypeScript side is hand-written, so the halves cannot drift.

set script-interpreter := ['bash', '-euo', 'pipefail']

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere.
# Exported rather than interpolated so the body reads it as a value.
export PYTHON := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The application's OpenAPI document, written the way litestar's own command
# writes it (litestar/cli/commands/schema.py _generate_openapi_schema). Built
# over a throwaway home, so it describes the application and not one machine.

# The destination is an environment variable: the gate writes the same
# document somewhere it can compare without overwriting the committed one.
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

# Two halves that together pin the whole chain. The document is regenerated
# from the running application and compared with the committed one, so a
# handler whose model changed without this being run fails here.

# The types are checked with openapi-typescript's own `--check`, which
# compares without rewriting. Both write to a temporary file, so the gate
# never edits the tree it is judging.
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
      diff --unified=2 frontend/openapi.json "$scratch/openapi.json" >&2 \n        || echo "and diff could not describe the difference" >&2
      exit 1
    fi

[private]
types-fresh:
    npm run --silent api-types-check
