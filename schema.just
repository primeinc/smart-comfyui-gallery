# Historical schemas: the authentic db/schema.sql of an older version.

# A test that proves a migration works on a vN database has to HAVE one.
# Inverting today's schema builds a fixture no real vN database matches, and
# chaining forward from v1 is circular: a step proved by its own peers.

#   the schema that shipped   `git show <sha>:db/schema.sql`

# So a fixture seeds from the file that shipped. Vendored rather than read
# from git at test time: the suite must run from a checkout with no git, and
# a vendored record is never edited, only added.

set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Every version git has a schema.sql for, and the commit that shipped it.
[doc('List each schema version and the commit whose db/schema.sql stamps it')]
[script]
versions:
    cd "$(git rev-parse --show-toplevel)"
    # rev-list --reverse is chronological, and the LAST commit stamping a
    # version is the one that shipped it -- so awk keeps the newest by
    # overwriting, never by sorting on the sha, which orders lexically.
    for sha in $(git rev-list --reverse HEAD -- db/schema.sql); do
        v=$(git show "$sha:db/schema.sql" | grep -oE 'user_version[[:space:]]*=[[:space:]]*[0-9]+' | grep -oE '[0-9]+')
        echo "$v $sha"
    done | awk '{ last[$1] = $2 } END { for (v in last) print v, last[v] }' | sort -n

# The LAST commit stamping a version is the one that shipped it: earlier
# ones are that version still being written.
[doc('Vendor the authentic db/schema.sql for VERSION into tests/schemas/vNN.sql')]
[script]
vendor VERSION:
    cd "$(git rev-parse --show-toplevel)"
    want="$1"
    sha=""
    for candidate in $(git rev-list --reverse HEAD -- db/schema.sql); do
        v=$(git show "$candidate:db/schema.sql" | grep -oE 'user_version[[:space:]]*=[[:space:]]*[0-9]+' | grep -oE '[0-9]+')
        [ "$v" = "$want" ] && sha="$candidate"
    done
    if [ -z "$sha" ]; then
        echo "no commit ships a db/schema.sql stamped v$want" >&2
        exit 1
    fi
    mkdir -p tests/schemas
    out=$(printf 'tests/schemas/v%02d.sql' "$want")
    git show "$sha:db/schema.sql" > "$out"
    echo "$out  <- $sha"

# Proof that a vendored schema is what it claims and that the shipped
# steps carry it to today. Run it after vendoring a new one.
[doc('Build each vendored schema, migrate it forward, and check drift')]
[script]
prove:
    cd "$(git rev-parse --show-toplevel)"
    {{ python }} -m tests.schemas
