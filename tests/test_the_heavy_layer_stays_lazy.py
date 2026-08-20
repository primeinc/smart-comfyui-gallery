"""Starting the gallery must not load the AI stack.

The AI layer is optional. Someone who has not enabled it, or has not
installed it, still gets a gallery -- and it has to start at the speed of
a gallery, not at the speed of torch.

Measured on this repo: `smartgallery_ai.faces` imported torch at module
scope for a single CUDA helper, and `import smartgallery` cost 2.67s
because of it. Moving that one import inside its function took the same
import to 0.94s and every affected test from ~3.1s to ~1.1s. Later,
provision.py importing huggingface_hub at module scope cost another 0.6s
and stopped the shipped container from starting at all.

pylint's PLC0415 says the opposite: it wants every import at the top of
its file. Following it here would undo both fixes and put four seconds
back into every start. What matters is not where an import is written but
what it costs to reach -- so this is the check, by name, for the packages
that are expensive or optional.

Static, so nothing is imported to find out. An import inside a function is
what the AI layer is supposed to look like and is left alone; only what
runs when the module is read is in scope here.
"""

from __future__ import annotations

import ast

from source_tree import parsed, sources

# What the gallery is, without the AI layer switched on.
_SHIPPED = ("smartgallery.py", "sg_auth.py", "sqlbind.py", "urlfetch.py", "smartgallery_ai", "omniquery", "metaparse")

# Slow to import, or belonging to a dependency group the core install does
# not carry. Either way, reaching one at import time is the defect.
_HEAVY = {
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


def _import_time_modules(tree):
    """{module: line} for imports that run when the file is read.

    Anything inside a function or a class body is skipped: that runs when
    something calls it, which is how an optional dependency is reached.
    """
    found = {}
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        else:
            stack.extend(ast.iter_child_nodes(node))
            continue
        for dotted in names:
            found.setdefault(dotted.split(".")[0], node.lineno)
    return found


def test_the_sweep_reads_the_imports_that_are_there():
    """Control. The check below is an absence, and a walk that understood
    nothing would report the same absence."""
    seen = {}
    for source in sources(*_SHIPPED):
        seen.update(_import_time_modules(parsed(source)))

    assert "flask" in seen, sorted(seen)
    assert "sqlite3" in seen, sorted(seen)
    assert len(seen) > 25, sorted(seen)


def test_nothing_heavy_is_imported_when_a_module_is_read():
    """Each of these has cost this program a start-up already."""
    eager = {}
    for source in sources(*_SHIPPED):
        for module, line in _import_time_modules(parsed(source)).items():
            if module in _HEAVY:
                eager[f"{source.name}:{line}"] = (module, _HEAVY[module])

    assert not eager, (
        f"{eager} -- imported while the module is being read, so every start "
        f"pays for it whether the AI layer is switched on or not. Move the "
        f"import inside the function that needs it."
    )


def test_an_eager_heavy_import_would_be_caught():
    """Control for the check above: it has to fail for the shape it exists
    to catch, and pass for the shape that is fine."""
    eager = _import_time_modules(ast.parse("import os\nimport torch\n"))
    lazy = _import_time_modules(ast.parse("import os\n\n\ndef go():\n    import torch\n    return torch\n"))

    assert "torch" in eager
    assert "torch" not in lazy, "an import inside a function is the fix, not the defect"
    assert "os" in lazy, "the walk stopped reading too early"
