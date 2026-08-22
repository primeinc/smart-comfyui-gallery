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
}

# --- SG2xx: the database is opened in one place ----------------------------------------------

#: Files that hold raw sqlite3.connect calls by decision.
RAW_CONNECT_DECIDED = frozenset({"connect.py", "migrate.py", "build.py"})

# --- SG3xx: the heavy layer stays lazy --------------------------------------------------------

#: Slow to import, or belonging to a dependency group the core install
#: does not carry. Reaching one at import time is the defect.
HEAVY_IMPORTS: dict[str, str] = {
    "torch": "the one that cost 2.67s",
    "transformers": "pulls torch",
    "open_clip": "pulls torch",
    "faiss": "optional, and the vendored build probes CUDA on import",
    "insightface": "optional, pulls onnxruntime",
    "onnxruntime": "optional",
    "mobile_sam": "optional, pulls torch",
    "huggingface_hub": "optional -- and its absence stopped the container",
    "llama_cpp": "optional",
    "sentence_transformers": "pulls torch",
}

# --- SG4xx: the web adapters own no semantics ------------------------------------------------

#: Methods that run a statement against a connection.
STATEMENT_METHODS = frozenset({"execute", "executemany", "executescript", "cursor"})

#: Presentation modules that may not run a statement, each with the
#: `from db import ...` vocabulary it may speak: address resolution,
#: configuration and the one answer seam -- never a query module.
ADAPTER_DB_VOCABULARY: dict[str, frozenset[str]] = {
    "sg_web/collection_authoring.py": frozenset({"collection_rules", "collections", "connect", "naming", "settings"}),
    "sg_web/person_view.py": frozenset({"authored", "connect", "naming", "pages", "resultset", "settings"}),
    # `library` is the marker-verified reachability probe the folders
    # index reports online state through -- presence, not membership.
    "sg_web/folder_view.py": frozenset({"connect", "library", "naming", "pages", "resultset", "settings"}),
    "sg_web/collection_view.py": frozenset(
        {"collection_rules", "collections", "connect", "naming", "pages", "resultset", "settings"}
    ),
    "sg_web/artifact_view.py": frozenset({"connect", "naming", "pages", "resultset", "settings"}),
    # facets is vocabulary, not a query path: it parses and spells the
    # closed registry and never touches a connection
    "sg_web/gallery.py": frozenset({"connect", "facets", "naming", "resultset", "settings"}),
    "sg_web/media_authored.py": frozenset({"authored", "collections", "connect", "naming", "pages"}),
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
    "sg_web/media_authored.py": frozenset({"set_favorite", "set_rating", "set_membership", "media_state"}),
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
}
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
DERIVED_RESERVED: dict[str, str] = {
    # Captions, OCR, tags -- the 'annotate' job kind exists in the schema
    # CHECK; the job that writes these rows does not exist yet.
    "derived_annotation": "writer ['annotate'] called by nothing outside db/derived.py",
}
DERIVED_PRODUCER_PACKAGES: tuple[str, ...] = ("db", "vision", "sg_web")
