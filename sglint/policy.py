"""What the structural rules agree to -- the decisions, one place.

Every entry here is a promise about the tree that the rules in
sglint/rules.py hold it to. Adding a name is a decision; the exactness
rule (SG102) prunes a name the moment nothing writes it, so these lists
are the tree's current truth, never a record of what used to be.
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
    # db/collections.py _claim_revision: "col = ?," assignments chosen
    # from the module's fixed patch vocabulary, every value bound
    "sets": "clause",
    "order": "keyword",
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
    # db/migrate.py, the step that widens job.kind: the column list of the
    # table being rebuilt, read from `PRAGMA table_info` rather than
    # written out -- which is what stopped that step inventing a `job`
    # table with no `heartbeat_at`. Column names are structure.
    "named": "identifier",
    # db/jobs.py: the one module-literal column list the active and recent
    # reads share, so both rows carry the same shape
    "_LISTED": "identifier",
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
    "sg_web/gallery.py": frozenset(
        {
            "analysis",
            "connect",
            "discovery",
            "facets",
            "naming",
            "pages",
            "places",
            "resultset",
            "settings",
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
    "sg_web/timeline_view.py": frozenset(
        {"connect", "context", "facets", "pages", "planning", "rendering", "resultset", "settings"}
    ),
    # `jobs` is vocabulary here, not a query path: the console spells
    # the job kinds and states the table constrains, and touches no
    # connection through it.
    "sg_web/operations.py": frozenset(
        {
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
            "settings",
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
            # The after-state a write hands back. It used to be `view` --
            # the whole management document, assembled and thrown away --
            # and it is now the address and the revision, still built by
            # the module that owns the collection's representations.
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
    # No entry for `sg_web/media_view.py: "neighbour"`. What it meant to
    # hold is that the viewer takes its walk from `resultset.neighborhood`
    # and never from `pages.neighbour` -- a second opinion about what
    # "next" means -- and MUST_NOT_CALL_QUALIFIED above states exactly
    # that, over the AST, where a call is a call. The substring form
    # added nothing and could not work: `pages.neighbour` is a live
    # function (db/pages.py), the word appears 51 times across this tree
    # including the route `/prompts/{id}/neighbours`, and SG406 announced
    # it as "deleted on purpose" -- which was never true. It fired on a
    # comment.
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
    "sg_web/media_view.py": {"PlaceKind": ("place", "kind")},
}

SURFACE_FORBIDDEN_WORDS: tuple[str, ...] = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "/search")

#: The shell every full page is a child of, and how a page says so.
SHELL_TEMPLATE = "base.html"
EXTENDS_SHELL = '{% extends "base.html" %}'

#: The functions that open a connection meant to outlive the call, and why.
#: SG103 takes returning or yielding as a transfer; it does not take storing
#: one, because that would let any leak be hidden inside an object. These two
#: are the real thing, and each says what it is for.
CONNECTION_KEPT: frozenset[str] = frozenset(
    {
        # One read-only monitor per database file, shared by every
        # projection currency read. It is the connection that SEES other
        # writers' commits, so outliving a request is its whole purpose.
        "db/resultset.py:currency",
        # The schema master a backup is taken from, one per DDL text.
        # Building it costs 11ms and ~250 tests start from a 0.5ms copy of
        # it; closing it would make every one of them pay again.
        "tests/staging.py:fresh_schema",
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
        "sg_web/person_view.py:people_index",
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
