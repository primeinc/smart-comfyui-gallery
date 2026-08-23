"""What the structural rules agree to -- the decisions, one place.

Every entry here is a promise about the tree that the rules in
sglint/rules.py hold it to. Adding a name is a decision; the exactness
rule (SG102) prunes a name the moment nothing writes it, so these lists
are the tree's current truth, never a record of what used to be.
"""

from __future__ import annotations

#: Directories that are not this project's code.
NOT_OURS = frozenset({".git", ".venv", "__pycache__", "node_modules", "vendor"})
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
    # facets is vocabulary, not a query path: it parses and spells the
    # closed registry and never touches a connection
    "sg_web/gallery.py": frozenset({"connect", "facets", "naming", "places", "resultset", "settings"}),
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
        {"connect", "context", "facets", "pages", "planning", "resultset", "settings"}
    ),
    "sg_web/operations.py": frozenset(
        {"connect", "derived", "inspecting", "ledger", "library", "pages", "prompts", "runner", "scan", "settings"}
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
            "view",
        }
    ),
    "sg_web/media_authored.py": frozenset({"set_favorite", "set_rating", "set_membership", "set_place", "media_state"}),
}

#: Qualified calls (`module.attr`) a module must make.
MUST_CALL_QUALIFIED: dict[str, frozenset[str]] = {
    "sg_web/gallery.py": frozenset({"resultset.page", "resultset.peek", "resultset.locate"}),
    "sg_web/asking.py": frozenset({"resultset.parse"}),
    "sg_web/media_view.py": frozenset({"resultset.locate"}),
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
    "sg_web/media_view.py": ("neighbour",),
    "db/pages.py": ("ARTIFACT_FILES", "WORKFLOW_FILES", "def artifact_files", "def workflow_files"),
    "sg_web/app.py": ("add_to_collection", "remove_from_collection"),
    # the Explorer page neither reaches nor reasons; the view module
    # neither writes, loads a model, nor owns a URL
    "sg_web/static/evolution.js": ("fetch(", "XMLHttpRequest", "localStorage", "cosine ="),
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
}
#: Words a file must not contain BEFORE a marker: the narrator (everything
#: above its persistence section) never touches a connection.
MUST_NOT_CONTAIN_BEFORE: dict[str, tuple[str, tuple[str, ...]]] = {
    "db/rendering.py": ("# --- persistence", ("execute(", "FROM ", "JOIN ", "sqlite3", "(conn", "conn,", "conn)")),
}
#: Words a file must not contain AFTER its module docstring.
MUST_NOT_CONTAIN_AFTER_DOCSTRING: dict[str, tuple[str, ...]] = {
    "story_renderers/claims.py": ("POLICY_VERSION",),
}
#: A function (class.method) whose signature must not carry a parameter.
NO_PARAMETER_NAMED: dict[str, tuple[tuple[str, str], ...]] = {
    "db/rendering.py": (("TemplateStoryRenderer.render", "conn"),),
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
    "sg_web/app.py": ("collections.set_membership",),
    "sg_web/templates/media.html": ("video", "animated_image", "image", "audio", "/media/"),
    "sg_web/templates/_media_lightbox.html": ("video", "animated_image", "image", "audio", "/media/"),
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

SURFACE_FORBIDDEN_WORDS: tuple[str, ...] = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "/search")
SURFACE_MINIMUM: int = 9

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
        ("job_event", "id"),
        # the unit a job's handler was given, an integer the job interprets
        ("job_event", "item_id"),
    }
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
