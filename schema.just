# Historical schemas: the authentic db/schema.sql of an older version.
#
# A test that proves a migration works on a vN database has to HAVE a vN
# database. There are three ways to get one and only one of them is
# honest:
#
#   invert today's schema     what this repo did for 35 versions
#   chain forward from v1     circular; a step is proved by its own peers
#   the schema that shipped   `git show <sha>:db/schema.sql`
#
# The first is why `test_the_shipped_steps_take_a_v1_database_to_the_
# current_build` was green over a path no real v1 database can walk: the
# fixture was today's `folder` with the stamp lowered, so it carried
# `missing_since`, which a real v1 folder does not have and no step adds.
# Measured 2026-08-25: an authentic v1 dies at step 4 with `no such
# column: missing_since`; an authentic v30 migrates to v36 with drift [].
#
# So a fixture seeds from the file that shipped. Vendored rather than
# read from git at test time: the suite must run from a checkout with no
# git, and these are immutable records -- a vendored one is never edited,
# only added.

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
