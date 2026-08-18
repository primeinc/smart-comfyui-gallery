"""The compose files have to name the same variables their instructions do.

`compose-exhibit.yaml` read `${EXHIBITION_PASS}` while the example .env at
the bottom of the same file said `EXHIBIT_PASS`. Anyone who followed the
instructions in the file got an empty substitution, so the container was
launched with

    --exhibition --port 8190 --admin-pass

-- a flag with nothing after it. That is not a warning: argparse exits 2
with a bare usage message, docker_init.bash reports a failure, and
`restart: unless-stopped` turns it into a crash loop. Nothing in any of
that output mentions a .env file or a variable name.

Both variables are now mandatory (`${VAR:?err}`, which the Compose spec
defines as exiting with that message when the variable is unset OR empty --
compose-spec/12-interpolation.md). So a missing password now stops
`docker compose up` and names what to set, instead of starting a container
that cannot launch.

These tests are text-level on purpose: the name agreement is the bug, and
checking it must not depend on a YAML parser being installed.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_COMPOSE = sorted(_REPO_ROOT.glob("compose*.y*ml"))

# ${NAME}, ${NAME:-default}, ${NAME:?err} ...
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")
# the example .env lines the file carries in its own comments
_DOCUMENTED = re.compile(r"^#\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _references(text):
    return dict(_REFERENCE.findall(text))


def _documented(text):
    return set(_DOCUMENTED.findall(text))


def test_there_are_compose_files_to_check():
    """Control for the sweep: a glob that matched nothing would make every
    test below pass by checking nothing at all."""
    assert len(_COMPOSE) >= 2, [str(p) for p in _COMPOSE]
    assert {"compose.yaml", "compose-exhibit.yaml"} <= {p.name for p in _COMPOSE}


@pytest.mark.parametrize("path", _COMPOSE, ids=lambda p: p.name)
def test_every_variable_used_is_the_one_documented(path):
    """The bug: the file asked for EXHIBITION_PASS and told the reader to
    set EXHIBIT_PASS."""
    text = path.read_text(encoding="utf-8")
    used = set(_references(text))
    documented = _documented(text)

    assert used == documented, (
        f"{path.name} uses {sorted(used)} but its own instructions document "
        f"{sorted(documented)}. Anyone following the file gets an empty "
        f"substitution."
    )


@pytest.mark.parametrize("path", _COMPOSE, ids=lambda p: p.name)
def test_every_variable_is_mandatory(path):
    """An unset variable must stop compose with a message, not expand to
    nothing and hand the app a flag with no value."""
    for name, rest in _references(path.read_text(encoding="utf-8")).items():
        assert rest.startswith((":?", "?")), (
            f"{path.name}: ${{{name}{rest}}} expands to nothing when unset. "
            f"Use ${{{name}:?message}} so compose stops and says what to set."
        )


def test_the_checks_catch_the_file_as_it_was():
    """Control for both matchers. Without this they could be passing because
    the regexes match nothing, and the next mismatch would sail through."""
    was_shipped = (
        "      CLI_ARGS: --exhibition --port 8190 --admin-pass ${EXHIBITION_PASS}\n"
        "# Example:\n"
        "# MAIN_PASS=your_main_password\n"
        "# EXHIBIT_PASS=your_exhibition_password\n"
    )

    used = _references(was_shipped)
    assert set(used) == {"EXHIBITION_PASS"}
    assert _documented(was_shipped) == {"MAIN_PASS", "EXHIBIT_PASS"}
    assert set(used) != _documented(was_shipped), "the mismatch went unnoticed"
    assert used["EXHIBITION_PASS"] == "", "it was not mandatory either"

    fixed = "${EXHIBITION_PASS:?set it in .env}\n# EXHIBITION_PASS=x\n"
    assert set(_references(fixed)) == _documented(fixed)
    assert _references(fixed)["EXHIBITION_PASS"].startswith(":?")


@pytest.mark.parametrize("path", _COMPOSE, ids=lambda p: p.name)
def test_the_compose_files_are_valid_yaml(path):
    """The mandatory-variable message contains a comma and a colon, so the
    value has to stay quoted to remain one scalar."""
    yaml = pytest.importorskip(
        "yaml",
        reason="PyYAML is not a declared dependency; the name checks "
        "above are the ones that matter and they need nothing",
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert document.get("services"), path.name
    for service in document["services"].values():
        environment = service.get("environment")
        if isinstance(environment, dict) and "CLI_ARGS" in environment:
            assert isinstance(environment["CLI_ARGS"], str)


@pytest.mark.parametrize("path", _COMPOSE, ids=lambda p: p.name)
def test_the_password_does_not_travel_inside_cli_args(path):
    """docker_init.bash runs `python smartgallery.py ${CLI_ARGS}` unquoted,
    so the shell splits that string on spaces. A passphrase set as
    MAIN_PASS="correct horse battery" reached the gallery as `correct` --
    measured, not assumed -- and the remaining words were dropped without
    an error, so the operator could not log in with what they had set.

    ADMIN_PASSWORD is the documented equivalent and crosses as one
    environment value, which nothing splits."""
    yaml = pytest.importorskip("yaml", reason="PyYAML is not a declared dependency")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    for name, service in document["services"].items():
        environment = service.get("environment")
        if not isinstance(environment, dict):
            continue
        assert "--admin-pass" not in (environment.get("CLI_ARGS") or ""), (
            f"{path.name}: {name} passes the password inside CLI_ARGS, which is word-split. Use ADMIN_PASSWORD."
        )
        if "--force-login" in (environment.get("CLI_ARGS") or "") or "--exhibition" in (
            environment.get("CLI_ARGS") or ""
        ):
            assert environment.get("ADMIN_PASSWORD"), (
                f"{path.name}: {name} forces a login but sets no ADMIN_PASSWORD, so it cannot start"
            )
