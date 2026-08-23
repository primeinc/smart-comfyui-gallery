"""SG8xx: repository hygiene -- what a clone hands out.

These rules ask git (the index, its line-ending classification, a
checkout in each autocrlf mode), read pyproject.toml against
requirements.txt, and read pytest.ini. They run as `python -m sglint
--repo` (`just repo-check`), never inside the test suite: a test never
spawns a program. Every rule takes its `git` runner as an argument so
the controls can hand it a fake and watch the rule fire.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import tomllib
import typing

from .rules import REPO_ROOT, Finding

Git = typing.Callable[..., "subprocess.CompletedProcess[str]"]

#: Personal launch scripts people make for themselves. The rule outlives
#: the filenames, so the names stay reserved and ignored.
PERSONAL_LAUNCHERS = ("run_smartgallery.bat", "run_exhibition.bat", "run_smartgallery.sh", "run_exhibition.sh")
#: Follows whatever the checkout does -- `* text=auto` with no eol rule of
#: its own -- so it says which autocrlf mode is in force.
BELLWETHER = "pyproject.toml"


def real_git(*args: str, cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is not on PATH")
    return subprocess.run(
        [git, *args], cwd=str(cwd or REPO_ROOT), capture_output=True, text=True, timeout=900, check=False
    )


def _lines(done: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in done.stdout.splitlines() if line.strip()]


def rule_index(git: Git = real_git, root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG800 the sweep cannot see the repository; SG801 a tracked path
    .gitignore matches (the rule does nothing while the file ships);
    SG802 a personal launcher is tracked; SG803 a personal launcher is
    not ignored."""
    found: list[Finding] = []
    at = root / ".gitignore"
    tracked = _lines(git("ls-files"))
    if len(tracked) <= 100 or "db/schema.sql" not in tracked:
        return [
            Finding(
                at, 1, 0, "SG800", f"git lists {len(tracked)} tracked files; the sweep is not seeing the repository"
            )
        ]
    ignored_on_disk = _lines(git("ls-files", "-i", "-o", "--exclude-standard"))
    if not ignored_on_disk:
        found.append(
            Finding(at, 1, 0, "SG800", "no ignored paths on disk; the tracked-but-ignored check cannot be validated")
        )
    found.extend(
        Finding(at, 1, 0, "SG801", f"{path} is tracked although .gitignore matches it; `git rm --cached {path}`")
        for path in _lines(git("ls-files", "-i", "-c", "--exclude-standard"))
    )
    found.extend(
        Finding(
            at,
            1,
            0,
            "SG802",
            f"{name} is tracked; the README has each person make their own and it would overwrite theirs",
        )
        for name in PERSONAL_LAUNCHERS
        if name in set(tracked)
    )
    found.extend(
        Finding(at, 1, 0, "SG803", f".gitignore does not match {name}; someone's paths can be committed by accident")
        for name in PERSONAL_LAUNCHERS
        if git("check-ignore", "--no-index", "-q", name).returncode != 0
    )
    return found


def committed_line_endings(git: Git = real_git) -> dict[str, str]:
    """{path: eolinfo} for the bytes in the index: `git ls-files --eol`
    reports i/<eolinfo> (git-ls-files.adoc:198-213)."""
    endings: dict[str, str] = {}
    for line in git("ls-files", "--eol").stdout.splitlines():
        fields, _tab, path = line.partition("\t")
        if not path:
            continue
        for field in fields.split():
            if field.startswith("i/"):
                endings[path] = field[2:]
    return endings


def rule_line_endings(git: Git = real_git, root: pathlib.Path = REPO_ROOT, *, checkouts: bool = True) -> list[Finding]:
    """SG804 a committed file holds CRLF (everyone gets those bytes);
    SG805 no .gitattributes (the policy is whatever the cloner has set);
    SG806 a checkout in one autocrlf mode does not behave as that mode,
    so the policy is not in force."""
    found: list[Finding] = []
    at = root / ".gitattributes"
    endings = committed_line_endings(git)
    if len(endings) <= 100 or sum(1 for k in endings.values() if k == "lf") <= 50:
        found.append(
            Finding(at, 1, 0, "SG800", "the index reading finds too few lf files; the parse misses i/<eolinfo>")
        )
    found.extend(
        Finding(at, 1, 0, "SG804", f"{path} is committed with {kind} line endings; everyone gets those bytes")
        for path, kind in sorted(endings.items())
        if kind in {"crlf", "mixed"}
    )
    if not at.exists():
        found.append(
            Finding(at, 1, 0, "SG805", "no .gitattributes: what a checkout does to the tree is the cloner's setting")
        )
    if checkouts:
        for autocrlf, expect_crlf in (("true", True), ("false", False)):
            with tempfile.TemporaryDirectory() as scratch:
                target = pathlib.Path(scratch)
                done = git("-c", f"core.autocrlf={autocrlf}", "checkout-index", "-a", "-f", f"--prefix={target}/")
                if done.returncode != 0:
                    found.append(
                        Finding(
                            at, 1, 0, "SG806", f"checkout-index failed under autocrlf={autocrlf}: {done.stderr[:200]}"
                        )
                    )
                    continue
                data = (target / BELLWETHER).read_bytes()
                crlf = data.count(b"\r\n")
                bare = data.count(b"\n") - crlf
                if expect_crlf and not (crlf > 0 and bare == 0):
                    found.append(
                        Finding(
                            at, 1, 0, "SG806", f"autocrlf=true checkout is not converting ({crlf} CRLF, {bare} bare)"
                        )
                    )
                if not expect_crlf and not (bare > 0 and crlf == 0):
                    found.append(
                        Finding(
                            at, 1, 0, "SG806", f"autocrlf=false checkout converts on its own ({crlf} CRLF, {bare} bare)"
                        )
                    )
    return found


def _shape(requirement) -> tuple[str, str, str]:
    """Everything about a dependency that decides what gets installed.

    Extras count: `litestar[pydantic]` and `litestar` install different
    packages, so a file that carries the extra and a file that does not do
    not agree, however identical their specifiers.
    """
    extras = ",".join(sorted(requirement.extras))
    return str(requirement.specifier), str(requirement.marker) if requirement.marker else "", extras


def rule_requirements(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG807 requirements.txt and pyproject disagree on a dependency's
    specifier or marker; SG808 the AI layer looks optional again (a
    dependency group beyond dev, or a second requirements file)."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    found: list[Finding] = []
    with open(root / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)
    declared = {}
    for entry in project["project"]["dependencies"]:
        requirement = Requirement(entry)
        declared[canonicalize_name(requirement.name)] = _shape(requirement)
    listed: dict[str, tuple[str, str, str]] = {}
    at = root / "requirements.txt"
    with open(at, encoding="utf-8") as fh:
        for line in fh:
            bare = line.split("#", 1)[0].strip()
            if bare:
                requirement = Requirement(bare)
                listed[canonicalize_name(requirement.name)] = _shape(requirement)
    found.extend(
        Finding(at, 1, 0, "SG807", f"requirements.txt lacks the pyproject dependency {name}")
        for name in sorted(set(declared) - set(listed))
    )
    found.extend(
        Finding(at, 1, 0, "SG807", f"{name}: pyproject declares {declared[name]}, requirements.txt says {listed[name]}")
        for name in sorted(set(declared) & set(listed))
        if listed[name] != declared[name]
    )
    groups = set(project.get("dependency-groups", {})) - {"dev"}
    if groups:
        found.append(
            Finding(
                root / "pyproject.toml",
                1,
                0,
                "SG808",
                f"dependency groups {sorted(groups)}: the AI layer is core, not a group",
            )
        )
    if (root / "requirements-ai.txt").exists():
        found.append(
            Finding(root / "requirements-ai.txt", 1, 0, "SG808", "requirements-ai.txt is back; the AI layer is core")
        )
    return found


def rule_pytest_path(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG809 pytest.ini no longer puts the repository root on sys.path,
    so only `python -m pytest` would work, by accident."""
    ini = (root / "pytest.ini").read_text(encoding="utf-8")
    entries = [line.split("=", 1)[1].strip() for line in ini.splitlines() if line.strip().startswith("pythonpath")]
    if not entries or "." not in entries[0].split():
        return [
            Finding(
                root / "pytest.ini",
                1,
                0,
                "SG809",
                "pytest.ini must set `pythonpath = .` so `uv run pytest` and a bare `pytest` import the application",
            )
        ]
    return []


def rule_commit_stamp(git: Git = real_git, root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG810 the browser report's provenance stamp is not honest about a
    dirty tree: against a scratch repository it must name the commit
    alone when clean, `-dirty` for a change or an untracked file, and
    `unknown` outside any repository."""
    import importlib.util

    where = root / "benchmarks" / "browser_report.py"
    spec = importlib.util.spec_from_file_location("browser_report_under_lint", where)
    if spec is None or spec.loader is None:
        return [Finding(where, 1, 0, "SG810", "the browser report driver cannot be loaded")]
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    found: list[Finding] = []
    with tempfile.TemporaryDirectory() as scratch:
        repo = pathlib.Path(scratch)
        for argv in (
            ("init", "--initial-branch=main"),
            ("add", "-A"),
        ):
            if argv[0] == "add":
                (repo / "witness.txt").write_text("as committed\n", encoding="utf-8")
            git(*argv, cwd=repo)
        git("-c", "user.email=stamp@lint", "-c", "user.name=stamp", "commit", "-q", "-m", "state", cwd=repo)
        clean = driver._commit_stamp(repo)
        if clean in ("", "unknown") or clean.endswith("-dirty") or not all(c in "0123456789abcdef" for c in clean):
            found.append(Finding(where, 1, 0, "SG810", f"a clean tree is stamped {clean!r}, not its commit alone"))
        (repo / "witness.txt").write_text("changed since\n", encoding="utf-8")
        if not driver._commit_stamp(repo).endswith("-dirty"):
            found.append(Finding(where, 1, 0, "SG810", "a changed file does not dirty the stamp"))
        (repo / "witness.txt").write_text("as committed\n", encoding="utf-8")
        (repo / "brand_new.py").write_text("print()\n", encoding="utf-8")
        if not driver._commit_stamp(repo).endswith("-dirty"):
            found.append(
                Finding(where, 1, 0, "SG810", "an untracked file does not dirty the stamp (the incident's exact shape)")
            )
    with tempfile.TemporaryDirectory() as empty:
        if driver._commit_stamp(pathlib.Path(empty)) != "unknown":
            found.append(Finding(where, 1, 0, "SG810", "a directory that is no repository is guessed at"))
    return found


def run() -> list[Finding]:
    found: list[Finding] = []
    for rule in (rule_index, rule_line_endings, rule_commit_stamp):
        found.extend(rule())
    found.extend(rule_requirements())
    found.extend(rule_pytest_path())
    return sorted(found, key=lambda f: (str(f.path), f.line, f.col, f.code))
