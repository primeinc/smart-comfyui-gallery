"""What the structural rules agree to -- the decisions, one place.

Every entry here is a promise about the tree that the rules in
sglint/rules.py hold it to. Adding a name is a decision; the exactness
rule (SG102) prunes a name the moment nothing writes it, so these lists
are the tree's current truth.
"""

from __future__ import annotations

#: Directories that are not this project's code.
NOT_OURS = frozenset({".git", ".venv", "__pycache__", "node_modules", "vendor"})
#: The test files SG007 does not judge: proving a rule fires means
#: handing it source, which is the one place that is the point.
SOURCE_INSPECTION_EXCUSED: frozenset[str] = frozenset({"test_sglint_has_teeth.py"})

#: Our code that does not ship to a user: builds throwaway state on purpose.
TOOLING = frozenset({"tests", "benchmarks", "experiments", "probes", "sglint"})
#: The application pairing every consumer of the database lives in.
NEW_PAIRING = frozenset({"db", "sg_web", "metaparse", "vision", "story_renderers"})

# --- SG1xx: SQL is built from structure only ---------------------------------------------------

#: Names allowed to appear in a SQL f-string, and what each carries: a
#: promise that the value is SQL text this codebase wrote, never anything
#: a caller supplied.
SQL_STRUCTURE: dict[str, str] = {
    # runs of "?" -- one per bound value
    "marks": "placeholders",
    # db/stories.py _marks: one "?" per frozen member id, nothing else
    "_marks(file_ids)": "placeholders",
    "_marks(members)": "placeholders",
    "','.join('?' * len(batch))": "placeholders",
    "','.join('?' * len(kinds))": "placeholders",
    # job-claim clauses in db/jobs.py: `runnable` is a module literal, and
    # `kind_filter` is "" or an IN (...) of placeholders, params bound
    "runnable": "clause",
    "kind_filter": "clause",
    # "" or a COALESCE-over-setting guard; the key and default are bound
    "gate_filter": "clause",
    # db/resultset.py membership: a conjunction of module-literal
    # predicates (every value in them bound), and ASC/DESC chosen from
    # the validated sort vocabulary -- eligibility is an intersection,
    # constructed, so the pieces are structure by definition
    "' AND '.join(where)": "clause",
    # db/runner.py submit_ingest: the freshness predicate, chosen between
    # two literals here -- the folder it may be bounded to is BOUND.
    "where": "clause",
    # db/runner.py _UNDER: the recursive CTE naming a folder's subtree.
    # A literal in this module; its one parameter is bound.
    "_UNDER": "clause",
    # db/collections.py _claim_revision: "col = ?," assignments chosen
    # from the module's fixed patch vocabulary, every value bound
    "sets": "clause",
    "order": "keyword",
    # db/resultset.py table sorts: the LEFT JOIN a sortable column needs
    # to be reachable, chosen from COLUMN_JOINS -- a module literal, one
    # entry per alias, and the only bound value in any of them (the
    # actor, in `rating`) rides as a `?`.
    "joined": "clause",
    # table / column names checked against sqlite_master or fixed registries
    "table": "identifier",
    "column": "identifier",
    "name": "identifier",
    # db/migrate.py, the step that installs the answer-generation
    # triggers: both come from a literal tuple in the step itself
    # (("INSERT", "ins"), ("UPDATE", "upd"), ("DELETE", "del")), and a
    # trigger's name and its event are structure -- neither can be bound.
    "verb": "keyword",
    "short": "identifier",
    # db/migrate.py `rebuilt`, the table-rebuild dance: the column list of
    # the table being rebuilt, read from `PRAGMA table_info` rather than
    # written out -- which is what stopped one step inventing a `job`
    # table with no `heartbeat_at`. Column names are structure.
    "named": "identifier",
    "', '.join(named)": "identifier",
    # the same list on the reading side, where a column the new table
    # cannot simply carry over is an expression from the step's own
    # `reading` literal (`CAST(inode AS TEXT)`). Both are structure.
    "reads": "identifier",
    # the transient name the old table is renamed to, f-string over
    # `table`, which is an identifier three lines above.
    "aside": "identifier",
    # "" or a WHERE over module-literal predicates: an inverse narrowing a
    # CHECK drops the rows the forward step's wider vocabulary allowed.
    "only": "clause",
    # db/jobs.py: the one module-literal column list the active and recent
    # reads share, so both rows carry the same shape
    "_LISTED": "identifier",
    # db/jobs.py TERMINAL_SQL: the settled-state IN (...) list, built at
    # import from the TERMINAL tuple beside JobState -- fixed words from
    # a Literal, never anything a caller supplied. One spelling, so the
    # statements cannot drift from the guard in `settle`.
    "TERMINAL_SQL": "clause",
    "jobs.TERMINAL_SQL": "clause",
    # db/resultset.py moment sort: the human timeline's one axis
    # (db/context.py HUMAN_MOMENT, a module literal) and the running
    # interpretation policy, an int constant from code, never request data
    "HUMAN_MOMENT": "clause",
    "int(POLICY_VERSION)": "identifier",
}

# --- SG4xx: the web adapters own no semantics ------------------------------------------------

#: Methods that run a statement against a connection.
STATEMENT_METHODS = frozenset({"execute", "executemany", "executescript", "cursor"})

#: Presentation modules that may not run a statement, each with the
#: `from db import ...` vocabulary it may speak: address resolution,
#: configuration and the one answer seam -- never a query module.
ADAPTER_DB_VOCABULARY: dict[str, frozenset[str]] = {
    "sg_web/collection_authoring.py": frozenset({"collection_rules", "collections", "connect", "naming", "settings"}),
    "sg_web/person_view.py": frozenset({"authored", "connect", "facets", "naming", "pages", "resultset", "settings"}),
    "sg_web/place_view.py": frozenset({"connect", "facets", "pages"}),
    # `library` is the marker-verified reachability probe the folders
    # index reports online state through -- presence, not membership.
    "sg_web/folder_view.py": frozenset({"connect", "facets", "library", "naming", "pages", "resultset", "settings"}),
    "sg_web/collection_view.py": frozenset(
        {"collection_rules", "collections", "connect", "facets", "naming", "pages", "resultset", "settings"}
    ),
    "sg_web/artifact_view.py": frozenset({"connect", "naming", "pages", "resultset", "settings"}),
    # facets and vocabulary are VOCABULARY, not query paths: they parse
    # and spell the closed registry and describe what its keys are
    # called, and neither touches a connection. `discovery` does read
    # one -- but only to count what one dimension's values would leave,
    # and it takes that count through resultset.scope_of rather than
    # writing membership SQL of its own, which is the whole point of it
    # existing beside the ResultSet instead of inside this adapter.
    # `analysis` is the other presentation of one answer: it aggregates
    # over the SAME membership, taken through resultset.scope_of, and
    # writes no WHERE clause of its own about which media are included.
    # `pages` for the TABLE presentation only: one read of the columns a
    # grid cell deliberately does not carry, over the ids the ResultSet
    # already returned. It never asks which files those are.
    # `catalog` is `discovery` for the OTHER axis and admitted on the
    # same terms: discovery counts what one dimension's values would
    # leave, catalog ranks which dimensions are worth offering at all,
    # and both take their membership through resultset.scope_of rather
    # than writing a WHERE clause about which media are included.
    "sg_web/gallery.py": frozenset(
        {
            "analysis",
            "catalog",
            "connect",
            "discovery",
            "facets",
            "naming",
            "pages",
            "places",
            "resultset",
            "settings",
            # Remembered questions, listed on the page where questions
            # are asked. Read only: the ADDRESS of a question, never a
            # rule, so nothing here defines what one means.
            "views",
            "vocabulary",
        }
    ),
    "sg_web/media_authored.py": frozenset(
        {"authored", "collections", "connect", "context", "naming", "pages", "places"}
    ),
    "sg_web/media_view.py": frozenset(
        {"authored", "connect", "derived", "facets", "naming", "pages", "places", "resultset", "settings"}
    ),
    "sg_web/curating.py": frozenset(
        {"authored", "collections", "connect", "context", "naming", "places", "resultset", "settings"}
    ),
    "sg_web/story_view.py": frozenset(
        {"connect", "derived", "evolution", "facets", "naming", "pages", "planning", "rendering", "settings", "stories"}
    ),
    # `when` is VOCABULARY on the same terms as `facets` above: it holds
    # how wide a claim at each precision is (`when.SPAN`) and how long a
    # day, a month and a year are, and it touches no connection -- it
    # imports dataclasses, datetime and re, and nothing else.
    #
    # Admitted because the alternative was worse. This adapter kept its
    # own `_SPAN = {"day": .., "hour": .., "minute": ..}` and indexed it
    # with a `time_precision` read out of the database, so the day a file
    # could be dated to its month, `/timeline` answered 500 with
    # `KeyError: 'month'`. A table keyed by a vocabulary another module
    # owns has to BE that vocabulary; a second copy only fails on the one
    # input the copy never heard of.
    "sg_web/timeline_view.py": frozenset(
        {"connect", "context", "facets", "pages", "planning", "rendering", "resultset", "settings", "when"}
    ),
    # `jobs` is vocabulary here, not a query path: the console spells
    # the job kinds and states the table constrains, and touches no
    # connection through it.
    "sg_web/operations.py": frozenset(
        {
            # The authored layer, READ, for the export that makes it
            # portable. Admitted on the narrowest terms: this module
            # calls `authored.exported` and nothing else there, because
            # an application whose thesis is custody of your own data
            # must not be the only place that data can exist -- and the
            # console is where a person looks for "take this with me".
            # Authoring itself stays in the media and person adapters.
            "authored",
            "connect",
            "derived",
            "inspecting",
            "jobs",
            "ledger",
            "library",
            "pages",
            "prompts",
            "runner",
            "scan",
            # What runs without being asked. The console is where a
            # schedule is read and set; the RUNNER is the only thing
            # that acts on one (`runner.run_schedules`), so this module
            # never starts a collection itself -- it writes a row saying
            # when one should start.
            "scheduling",
            "settings",
            # The verdicts, added up. Admitted on the same terms as
            # `inspecting`: a read-only aggregate that derives numbers
            # from rows and writes nothing -- and one that deliberately
            # never joins the derived layer it is about, so a rebuild
            # cannot look like people changing their minds.
            "verdicts",
        }
    ),
    # the composition root: it wires every seam and runs none of them
    "sg_web/app.py": frozenset(
        {
            "authored",
            "collections",
            "connect",
            "derived",
            "detect",
            "jobs",
            "ledger",
            "library",
            "migrate",
            "naming",
            "oriented",
            "pages",
            "prompts",
            "retrieval",
            "runner",
            "sample",
            "scan",
            "settings",
            # Remembered questions. A saved view is the ADDRESS of a
            # question and never a rule, so this module stores and lists
            # spellings and defines no query semantics -- which is the
            # same line the smart-collection route holds by sending the
            # canonical spelling instead of a rule shape.
            "views",
        }
    ),
}

#: Calls an adapter must still make -- the delegation it exists for.
ADAPTER_MUST_CALL: dict[str, frozenset[str]] = {
    "sg_web/collection_authoring.py": frozenset(
        {
            "create_listed",
            "create_smart",
            "update_definition",
            "replace_rule",
            "convert_to_smart",
            "convert_to_listed",
            # The after-state a write hands back: the address and the
            # revision, built by the module that owns the collection's
            # representations.
            "write_answer",
        }
    ),
    "sg_web/media_authored.py": frozenset({"set_favorite", "set_rating", "set_membership", "set_place", "media_state"}),
}

#: Qualified calls (`module.attr`) a module must make.
MUST_CALL_QUALIFIED: dict[str, frozenset[str]] = {
    "sg_web/gallery.py": frozenset({"resultset.page", "resultset.peek", "resultset.locate"}),
    "sg_web/asking.py": frozenset({"resultset.parse"}),
    "sg_web/media_view.py": frozenset({"resultset.neighborhood"}),
}
#: Qualified calls a module must never make.
MUST_NOT_CALL_QUALIFIED: dict[str, frozenset[str]] = {
    "sg_web/media_view.py": frozenset({"pages.neighbour"}),
}
#: `from <module> import <name>` a file must carry.
MUST_IMPORT: dict[str, tuple[str, str]] = {
    "sg_web/gallery.py": ("sg_web.asking", "gallery_query"),
}
#: Words a source file must not contain at all.
MUST_NOT_CONTAIN: dict[str, tuple[str, ...]] = {
    # `sg_web/media_view.py: "neighbour"` has no entry here: the viewer takes
    # its walk from `resultset.neighborhood`, and MUST_NOT_CALL_QUALIFIED
    # states that over the AST, where a call is a call.
    "db/pages.py": ("ARTIFACT_FILES", "WORKFLOW_FILES", "def artifact_files", "def workflow_files"),
    "sg_web/app.py": ("add_to_collection", "remove_from_collection"),
    # uvicorn workers > 1 splits the in-memory feed across processes
    "sg_web/__main__.py": ("workers=",),
    # the Explorer page neither reaches nor reasons; the view module
    # neither writes, loads a model, nor owns a URL
    "frontend/src/evolution.ts": ("fetch(", "XMLHttpRequest", "localStorage", "cosine ="),
    "db/evolution.py": (
        "INSERT",
        "UPDATE",
        "DELETE",
        "encoder(",
        "encode_query",
        "manager_for",
        "generation_prompt",
        "/thumb/",
        "/search",
    ),
    # the story seam: no model, no network, no rewriting a frozen row;
    # the adapter never reaches around it to a source table
    "db/stories.py": ("UPDATE story_snapshot", "import requests", "import httpx", "import openai", "anthropic"),
    "sg_web/story_view.py": ("FROM ", "JOIN ", "execute(", "derived_"),
    # the narrator owns no model and lays out nothing; the package token
    # is the one render policy
    "db/rendering.py": ("import openai", "anthropic", "import requests", "import httpx", "torch", "jinja"),
    "story_renderers/claims.py": ("execute(", "sqlite3", "jinja", "import db"),
    "story_renderers/formatting.py": ("POLICY_VERSION",),
    "sg_web/templates/story.html": ("|safe", "{% set", "cosine", "similarity", "claim.kind", "execute"),
    # a grouper consumes the per-claim occurrence interface, never a
    # source table: reading `file` or `capture` directly would be a second
    # interpretation of time beside db/context.py's one
    "db/events.py": ("FROM file", "FROM capture", "FROM generation", "JOIN entity", "FROM entity"),
    # no sibling reaches the judge: the same claims give the same verdict
    # whatever else is in the folder, so the folder is not an input
    "db/when.py": ("folder_id",),
    # the planner writes no story with a model and calls nothing over the
    # network; it reads frozen evidence and structures it
    "db/planning.py": ("import openai", "anthropic", "import requests", "import httpx", "torch"),
}
#: Words a file must not contain BEFORE a marker: the narrator (everything
#: above its persistence section) never touches a connection.
#: {module: (marker, banned words, functions cut out before the sweep)}.
MUST_NOT_CONTAIN_BEFORE: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "db/rendering.py": (
        "# --- persistence",
        ("execute(", "FROM ", "JOIN ", "sqlite3", "(conn", "conn,", "conn)"),
        (),
    ),
    # The planner structures frozen evidence: it holds no connection, runs
    # no statement and loads no model. `engine_for` resolves which provider
    # is CONFIGURED, which is the one thing before the marker that needs a
    # connection, so it is cut out rather than the words being widened.
    "db/planning.py": (
        "# --- persistence and orchestration",
        ("execute(", "FROM ", "JOIN ", "sqlite3", "(conn", "conn,", "conn)"),
        ("engine_for",),
    ),
}
#: Words a file must not contain AFTER its module docstring.
MUST_NOT_CONTAIN_AFTER_DOCSTRING: dict[str, tuple[str, ...]] = {
    "story_renderers/claims.py": ("POLICY_VERSION",),
}
#: A function (class.method) whose signature must not carry a parameter.
NO_PARAMETER_NAMED: dict[str, tuple[tuple[str, str], ...]] = {
    "db/rendering.py": (("TemplateStoryRenderer.render", "conn"),),
    # the judge weighs one file's own claims; a collapsed sibling set is
    # not evidence about it
    "db/when.py": (("judge_generation", "collapsed"),),
}
#: Regexes no file of a package may match.
PACKAGE_FORBIDDEN_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    # REPLACE fires no DELETE trigger: it strands an FTS entry and inflates
    # param_key. Use ON CONFLICT(file_id, source, key) DO UPDATE.
    "db": ((r"INSERT\s+OR\s+REPLACE\s+INTO\s+file_param", "INSERT OR REPLACE INTO file_param strands an FTS entry"),),
}
#: db/pages.py ships the page queries as module constants: at least this
#: many uppercase SELECT constants, or a page restated its query elsewhere.
PAGE_QUERIES_MINIMUM: int = 12
#: Words a source file must contain.
MUST_CONTAIN: dict[str, tuple[str, ...]] = {
    # One process, one channel: the feed is in-memory, so a second
    # worker would be a second channel and half the subscribers would
    # never hear a job move (sg_web/__main__.py starts no workers).
    "sg_web/app.py": ("collections.set_membership", "MemoryChannelsBackend()"),
    # ONE viewer, and it answers for every kind the schema admits. The two
    # containers include it; neither may grow a second stage, which is what
    # naming the kinds HERE and nowhere else holds them to.
    "sg_web/templates/_media_viewer.html": ("video", "animated_image", "image", "audio", "item.stage"),
    # The container adapters are containers: each mounts the one viewer.
    "sg_web/templates/media.html": ("_media_viewer.html",),
    "sg_web/templates/_media_lightbox.html": ("_media_viewer.html",),
    # The variant addresses are spelled once each, where the stage is built.
    "sg_web/media_view.py": ("/media/", "/preview/"),
    # the grouping input IS the occurrence interface, named out loud
    "db/events.py": ("context.occurrences(",),
}
#: Single-item desired-state adapters that must delegate to their _many
#: form (so validation cannot fork), and the _many must use executemany.
ONE_DELEGATES_TO_MANY: tuple[tuple[str, str], ...] = (
    ("set_favorite", "set_favorite_many"),
    ("set_rating", "set_rating_many"),
    ("set_membership", "set_membership_many"),
    ("set_place", "set_place_many"),
)
ONE_TO_MANY_MODULES: tuple[str, ...] = ("db/authored.py", "db/collections.py")
#: Modules whose every executed statement must be a literal: no road from
#: stored text to execution.
LITERAL_STATEMENTS_ONLY: tuple[str, ...] = ("db/collection_rules.py",)

# --- SG5xx: templates and scripts carry no query logic --------------------------------------

#: Closed vocabularies the HTTP contract restates as `Literal`, and the CHECK
#: constraint each one must equal: {module: {name: (table, column)}}.
#:
#: The wire says the members out loud so the browser is given a union instead
#: of `string`, which means they are a human copy of the schema until
#: something compares the two. This is text against text -- an assignment in a
#: module and a constraint in the DDL -- so it is a lint, not a test: no
#: database is built and no application is served to learn that two lists
#: disagree.
WIRE_VOCABULARIES: dict[str, dict[str, tuple[str, str]]] = {
    "db/jobs.py": {
        "JobKind": ("job", "kind"),
        "JobState": ("job", "state"),
        "ItemState": ("job_item", "state"),
    },
    "db/ledger.py": {"EventType": ("job_event", "type"), "Severity": ("job_event", "severity")},
    "db/resultset.py": {"MediaKind": ("file", "kind")},
    "sg_web/media_view.py": {"PlaceKind": ("place", "kind")},
}

SURFACE_FORBIDDEN_WORDS: tuple[str, ...] = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "/search")

# --- SG80x: the repo sweep's own sanity floors ------------------------------------------------

#: Below this many tracked files the git sweep declares itself blind
#: (SG800) rather than passing on a repository it cannot be seeing --
#: this repo tracks several hundred, so a double-digit answer means the
#: command ran somewhere wrong.
TRACKED_MINIMUM: int = 100
#: The same self-doubt for the line-ending parse: fewer entries than
#: this, or fewer LF files than LF_MINIMUM, means `i/<eolinfo>` was not
#: parsed, not that the tree changed character.
ENDINGS_MINIMUM: int = 100
LF_MINIMUM: int = 50
#: How long one git subprocess may run before the sweep is called hung.
#: Fifteen minutes is far past any healthy `ls-files` or checkout-index
#: on this tree; the value exists to fail rather than hang, not to pace.
GIT_TIMEOUT_SECONDS: int = 900
#: How much of a failed checkout's stderr a finding carries -- enough
#: to name the cause, bounded so one finding cannot become a dump.
STDERR_SHOWN: int = 200

#: The shell every full page is a child of, and how a page says so.
SHELL_TEMPLATE = "base.html"
EXTENDS_SHELL = '{% extends "base.html" %}'

#: A capability nothing in the interface reaches, and why that is the
#: right answer. Anything absent from here must be reachable and anything
#: here must still exist, and SG010 checks both halves.
UNSURFACED = {
    # Infrastructure. Nothing should link to these.
    "/health": "a liveness probe for whatever runs the process",
    "/favicon.ico": "the browser asks for it; no page links to it",
    "/ws/jobs/frames": "the shell's activity feed connects by attribute, not by a written address",
    "/ws/events/frames": "the console's ledger feed, the same way",
    # Machine reads on purpose: a person reaches the same facts through a page.
    "/ways": "the query vocabulary, for a client building a question by hand",
    "/clusterings": (
        "every face-clustering run as JSON. A person reads the same facts as sentences: the "
        "People page's empty state says where each run stands, and "
        "/operations/clusterings/{left}/against/{right} shows two of them side by side"
    ),
    "/views": (
        "the saved questions as JSON. A person meets them where they were saved -- the filter "
        "bar offers them by name, and each one opens as the question it stands for"
    ),
    "/timeline/at": "a moment lookup for a client driving the timeline itself",
    "/timeline/density": "the same, at bin resolution",
    "/timeline/pictures": "the same, over a span",
    "/stories/snapshots": "one frozen snapshot by id, for a client walking a plan",
    # Reached by an address the SERVER hands the client, so the string
    # check below cannot see it.
    "/search": "linked from the evolution surface as view.links.search (frontend/src/evolution.ts)",
    # The sweeps in their machine spelling, kept as the API a script uses: a
    # person starts the same work from the operations console, which posts to
    # /operations/jobs/{kind} and is answered by LAUNCHERS (sg_web/operations.py).
    "/jobs/verify": "console sweep `verify`",
    "/jobs/faces": "console sweep `faces`",
    "/jobs/annotate": "console sweep `annotate`",
    "/jobs/catch-up": "console sweep `catch_up`",
    "/jobs/thumbs": "console sweep `thumbs`",
    "/jobs/phash": "console sweep `phash`",
    "/jobs/embed": "console sweep `embed`",
    "/jobs/embed_prompts": "console sweep `embed_prompts`",
    "/jobs/dupes": "console sweep `dupes`",
    "/jobs/context": "console sweep `context`",
    "/jobs/events": "console sweep `events`",
    "/jobs/cluster": "console sweep `cluster`",
}

#: Capabilities that are not addresses. Same contract as above -- recorded
#: with the evidence, so nobody has to rediscover them -- and each says
#: what would have to happen first, because none of these can be surfaced
#: by drawing something.
#:
#: That is the useful part. "No affordance" reads like a decision somebody
#: could reverse this afternoon; every one of these is blocked further
#: back, and saying where stops the next person starting at the wrong end.
UNSURFACED_BEYOND_ROUTES = {
    "face age/sex/pose": (
        "vision/faces.py computes them and derived_face_instance stores them. "
        "Blocked at the contract: sg_web/media_view.py Person carries slug, name, href and a count, "
        "so there is nothing for a surface to draw until the wire says it"
    ),
    "lineage": (
        "db/lineage.py is called only from tests, so file_derivation is never written. "
        "The inspector's parents and children rows already exist and read it, and are therefore "
        "structurally always empty -- built correctly against a table production never fills"
    ),
    "watched folders": (
        "db/jobs.py watch_folder and unwatch_folder have no route, so there is no address for a control to post to"
    ),
    "face_across_runs": (
        "db/pages.py has no caller for it; the clusterings compare page renders `disagreements` instead, "
        "which is run-level. Per-face divergence needs the route to ask for it"
    ),
    "comments": (
        "db/authored.py comment and edit_comment have no caller and nothing reads the table. "
        "Not a missing edit button: there is no write path and no read path, so the whole capability "
        "is behind a route that does not exist"
    ),
    "job kinds sample_frames, remix, zip": (
        "db/jobs.py JobKind declares fourteen kinds; db/runner.py HANDLERS implements eleven. "
        "These three are words with nothing behind them and never have been: `git log -S` over "
        "db/runner.py finds no commit that ever added a handler for one, where the same search "
        "finds b0e3083 for walk. Nothing should draw a control for them -- there is no code to run. "
        "They are held in place by db/schema.sql:664, the CHECK on job.kind, which SG709 pins "
        "JobKind equal to, so dropping them from the Literal alone trips that rule instead. "
        "Removing them is jobs.py, schema.sql, a new migration step and console.py -- a database "
        "contract change. Two traps for whoever does it: the four occurrences in db/migrate.py are "
        "frozen history and must not be edited, and schema.sql:768 and :780 carry `remix` in a "
        "different vocabulary, the derivation kinds, which this has nothing to do with"
    ),
}

#: A job kind no console button starts, and what starts it instead.
#:
#: Not every capability belongs on a button. A kind here is reached by
#: doing the thing it belongs to, and the entry names that thing so the
#: claim can be checked rather than trusted -- SG012 checks both halves,
#: that the kind still exists and that what is said to start it still
#: does.
STARTED_ELSEWHERE: dict[str, str] = {
    "walk": (
        "queued by `catch_up` as its first step (db/runner.py). Walking the roots alone finds "
        "files and reads none of them, which settles `done` having apparently done nothing; "
        "'bring the library up to date' is the affordance, and it walks first."
    ),
    "story_plan": (
        "queued by opening a sitting that has no story yet -- /stories/sessions/{id}, which IS "
        "the session card's link on the timeline (sg_web/templates/_timeline_session.html). "
        "Telling a story is something done to one sitting, never a sweep over the library."
    ),
}

#: Pages that own their whole document instead of extending the shell, and
#: why. SG502 accepts an entry here in place of EXTENDS_SHELL.
OWN_DOCUMENT: dict[str, str] = {
    "field.html": (
        "the canvas surface. A rail above a camera and a pager below it is a canvas in a box, "
        "which is the thing this page exists to stop being -- so it carries its destinations as "
        "a launcher over the picture plane instead, and the crawl in "
        "tests/test_the_shell_mounts_every_surface.py accepts either shell"
    ),
}

#: The functions that open a connection meant to outlive the call, and why.
#: SG103 takes returning or yielding as a transfer; it does not take storing
#: one, because that would let any leak be hidden inside an object. Every
#: entry is the real thing, and each says what it is for.
CONNECTION_KEPT: frozenset[str] = frozenset(
    {
        # One read-only monitor per database file, shared by every
        # projection currency read. It is the connection that SEES other
        # writers' commits, so outliving a request is its whole purpose.
        "db/resultset.py:currency",
        # The schema master a backup is taken from, one per DDL text.
        # Every test starts from a copy of it, so closing it would make
        # each one rebuild the schema.
        "tests/staging.py:fresh_schema",
        # The Stage's data_version monitor: an idle reader whose whole
        # purpose is outliving each test to see other connections'
        # commits, so a restore nothing wrote through is skipped.
        "tests/staging.py:_data_version",
        # The restore's SOURCE: the frozen template, read-only. Same
        # bytes for every restore in the module, so re-opening it per
        # test buys nothing and costs a connection's PRAGMAs on the
        # hot path. Re-opened after a rebuild writes a new template.
        "tests/staging.py:_from_template",
        # The restore's DESTINATION. Held for the same reason, and it
        # is between transactions whenever a test can see it, so it
        # takes no lock the application's own connections would meet.
        "tests/staging.py:_into_db",
        # All three end in Stage.close_held, at module teardown.
    }
)

#: Route handlers whose JSON answer SG413 does not yet hold to a wire
#: model. This is the migration's remaining surface, written down: a
#: handler leaves the list by naming its answer, never by being excused,
#: and SG413 reports an entry here that no longer offends, so the list can
#: only shrink.
RESPONSE_CONTRACT_RESERVED: frozenset[str] = frozenset(
    {
        "sg_web/app.py:add_root",
        "sg_web/app.py:album_add",
        "sg_web/app.py:album_remove",
        "sg_web/app.py:all_settings",
        "sg_web/app.py:cancel_job",
        "sg_web/app.py:change_setting",
        "sg_web/app.py:choose_primary",
        "sg_web/app.py:clusterings",
        "sg_web/app.py:dupes",
        "sg_web/app.py:front",
        "sg_web/app.py:prompt_neighbours",
        "sg_web/app.py:roots",
        "sg_web/app.py:scan_root",
        "sg_web/app.py:search",
        "sg_web/app.py:submit_annotate",
        "sg_web/app.py:submit_cluster",
        "sg_web/app.py:submit_context",
        "sg_web/app.py:submit_dupes",
        "sg_web/app.py:submit_embed",
        "sg_web/app.py:submit_embed_prompts",
        "sg_web/app.py:submit_events",
        "sg_web/app.py:submit_faces",
        "sg_web/app.py:submit_ingest",
        "sg_web/app.py:submit_phash",
        "sg_web/app.py:submit_thumbs",
        "sg_web/app.py:submit_verify",
        "sg_web/app.py:ways",
        "sg_web/artifact_view.py:lora_page",
        "sg_web/artifact_view.py:loras_index",
        "sg_web/artifact_view.py:model_page",
        "sg_web/artifact_view.py:models_index",
        "sg_web/artifact_view.py:workflow_page",
        "sg_web/artifact_view.py:workflows_index",
        "sg_web/folder_view.py:folder_page",
        "sg_web/folder_view.py:folders_index",
        "sg_web/person_view.py:person_page",
        "sg_web/place_view.py:places_index",
        "sg_web/story_view.py:plan_document",
        "sg_web/story_view.py:plan_snapshot",
        "sg_web/story_view.py:render_document",
        "sg_web/story_view.py:snapshot_document",
        "sg_web/story_view.py:stories_index",
    }
)

#: Route handlers whose JSON body SG412 does not hold to a Wire contract,
#: and why. Empty, and meant to stay that way: every JSON body this
#: application takes is a named contract. SG412 reports an entry here that
#: no longer offends, so a line cannot outlive its reason.
REQUEST_CONTRACT_RESERVED: frozenset[str] = frozenset()

#: Each half of the sweep counts on its own. One shared minimum could not see
#: every script leaving `sg_web/static` for `frontend/src`, because the
#: templates alone cleared it.
TEMPLATE_MINIMUM: int = 30
SCRIPT_MINIMUM: int = 11

# --- SG6xx: every derived table has a producer something calls -------------------------------

#: Derived tables whose writer is declared reserved: the job that writes
#: them does not exist yet. Each line is a decision on the record.
DERIVED_RESERVED: dict[str, str] = {}
DERIVED_PRODUCER_PACKAGES: tuple[str, ...] = ("db", "vision", "sg_web")

# --- SG7xx: the schema contract ---------------------------------------------------------------

#: Columns ending in _id that are deliberately not references to another row.
NOT_A_REFERENCE: frozenset[tuple[str, str]] = frozenset(
    {
        ("entity", "id"),
        ("root", "id"),
        ("user", "id"),
        ("job", "id"),
        ("derived_media_sample", "id"),
        ("derived_face_scan", "model_id"),  # a backend's name, the same text derived_face_instance carries
        ("derived_face_scan", "model_version"),
        ("comment", "id"),
        ("feedback", "id"),
        ("derivation_intent", "id"),
        ("file_derivation", "id"),
        ("derived_face_cluster", "id"),
        ("derived_face_instance", "id"),
        ("derived_annotation", "id"),
        ("region", "id"),
        # A verdict names the producer it judged and must NOT reference
        # it: a re-run replaces the judged row and the verdict has to
        # survive that. A foreign key with any ON DELETE is
        # wrong in both directions here -- CASCADE deletes the human's
        # words with the machine's, SET NULL erases which model was
        # judged, which is the only thing that makes the verdict
        # aggregable afterwards. Copied text, on purpose.
        ("feedback", "model_id"),
        ("feedback", "model_version"),
        # backend identity strings ("insightface", "qwen-vl"), not rows
        ("derived_embedding", "model_id"),
        ("derived_face_cluster", "model_id"),
        ("derived_face_instance", "model_id"),
        ("derived_annotation", "model_id"),
        ("derived_file_person", "model_id"),
        ("derived_face_run", "model_id"),
        ("job_item", "item_id"),
        # The filesystem's own identifier for a directory or a file --
        # NTFS FileID, an ReFS FileId128 -- kept as its exact decimal
        # spelling. It names a row in no table of ours and there is
        # nothing to reference: it is minted by the volume, is lost on a
        # copy or a restore, and is absent entirely on filesystems that
        # report none.
        ("folder", "fs_id"),
        ("file", "fs_id"),
        ("job_event", "id"),
        # the unit a job's handler was given, an integer the job interprets
        ("job_event", "item_id"),
    }
)
#: TEXT columns whose name looks like a closed vocabulary but whose values
#: are genuinely open. Each is a decision on the record, not an oversight
#: -- SG711 reads a name ending in one of its suffixes as an enum unless it
#: is here.
FREE_TEXT: frozenset[str] = frozenset(
    {
        # the location of a root, which is the one place a path IS the fact
        "path",
        # a person's own words, and the name of a thing as its metadata
        # spelled it
        "name",
        "note",
        "summary",
        "description",
        "body",
        "text",
    }
)

#: The column-name endings SG711 reads as naming a fixed set.
VOCABULARY_ENDINGS: tuple[str, ...] = (
    "kind",
    "state",
    "role",
    "verdict",
    "space",
    "carrier",
    "source",
    "severity",
    "sex",
    "policy",
)

#: Composite-key tables that must be WITHOUT ROWID (sqlite.org/withoutrowid.html).
WITHOUT_ROWID: tuple[str, ...] = ("file_artifact", "derived_file_person", "collection_file", "rating", "favorite")
#: The long tail keeps its rowid: multi-KB values are what the optimization is not for.
KEEPS_ROWID: tuple[str, ...] = ("file_param",)
#: References the product's correctness rests on, and where each must point.
LOAD_BEARING_REFERENCES: dict[tuple[str, str], str] = {
    ("file", "folder_id"): "folder",
    ("file", "id"): "entity",
    ("folder", "parent_id"): "folder",
    ("folder", "root_id"): "root",
    ("derived_file_person", "person_id"): "person",
    ("derived_file_person", "file_id"): "file",
    ("file_artifact", "artifact_id"): "artifact",
    ("collection_file", "collection_id"): "collection",
    ("capture", "file_id"): "file",
    ("file_param", "file_id"): "file",
    ("file_relation", "related_id"): "file",
    ("slug_history", "entity_id"): "entity",
    ("derivation_intent", "parent_id"): "file",
    ("file_derivation", "child_id"): "file",
    ("generation", "workflow_id"): "artifact",
    ("generation_prompt", "prompt_id"): "prompt",
    ("generation_prompt", "file_id"): "generation",
    ("collection", "parent_id"): "collection",
    ("derived_face_instance", "sample_id"): "derived_media_sample",
    ("derived_face_cluster", "person_id"): "person",
    ("job", "target_id"): "entity",
}


# --- SG713 / SG415: a vocabulary and the code that must cover it --------------------------------

#: A dispatch table and the closed vocabulary every one of its keys must
#: cover: {module: {table: (vocabulary module, vocabulary name)}}. Both
#: halves are module-level literals, so this is text against text -- an
#: event the ledger can write and the console cannot say is caught without
#: building a database or rendering anything.
VOCABULARY_HANDLERS: dict[str, dict[str, tuple[str, str]]] = {
    "sg_web/console.py": {"RENDERINGS": ("db/ledger.py", "EventType")},
}

#: Job handler registries whose every handler must reach the reporting
#: seam: {module: registry name}. A long handler that says nothing between
#: item.started and item.done is a frozen progress bar.
HANDLER_REGISTRIES: dict[str, str] = {"db/runner.py": "HANDLERS"}

#: A kind whose handler only dispatches, and the functions it dispatches
#: to -- those are what must report, not the router.
HANDLER_DISPATCH: dict[str, tuple[str, ...]] = {
    "hash": ("_verify_item", "_perceptual_item", "_thumbs_item", "_dupe_groups_item"),
}

#: The call that IS reporting.
REPORTING_CALL = "report"
