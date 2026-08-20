"""A mistyped yes/no setting must not silently mean "no".

CONFIGURATION.md promises that "an unparseable number or yes/no word warns
on the console and falls back to the default". The gallery's own env_flag
does exactly that. The AI layer had a second, older helper that tested
membership of the truthy set and nothing else:

    return raw.strip().lower() in ("1", "true", "yes", "on")

so every misspelling meant False. `ENABLE_AI_DAM=ture` switched off Similar,
Faces, Review and the search palette -- the opposite of the documented
default -- with nothing on screen to say why, which is indistinguishable
from the feature being broken.

The same helper reads AI_DAM_AUTO_PROVISION and AI_DAM_EPHEMERAL_INDEX.
"""

from __future__ import annotations

import pytest

import smartgallery_ai


@pytest.mark.parametrize("typo", ["ture", "yess", "enabled", "y e s", "truthy", "-1"])
def test_a_misspelt_yes_keeps_a_true_default(monkeypatch, typo):
    """The regression: these all read as False."""
    monkeypatch.setenv("ENABLE_AI_DAM", typo)

    assert smartgallery_ai._env_bool("ENABLE_AI_DAM", "true") is True, (
        f"{typo!r} silently turned a default-on setting off"
    )


@pytest.mark.parametrize("typo", ["ture", "maybe", "0.0"])
def test_a_misspelt_value_keeps_a_false_default(monkeypatch, typo):
    monkeypatch.setenv("AI_DAM_EPHEMERAL_INDEX", typo)

    assert smartgallery_ai._env_bool("AI_DAM_EPHEMERAL_INDEX") is False


def test_it_says_so(monkeypatch, caplog):
    """Falling back quietly would still leave someone wondering why the
    setting did nothing."""
    monkeypatch.setenv("ENABLE_AI_DAM", "ture")

    with caplog.at_level("WARNING"):
        smartgallery_ai._env_bool("ENABLE_AI_DAM", "true")

    assert "ENABLE_AI_DAM" in caplog.text, caplog.text
    assert "ture" in caplog.text, caplog.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
    ],
)
def test_the_real_values_still_parse(monkeypatch, value, expected):
    """The counterpart -- a helper that always returned the default would
    pass every test above."""
    monkeypatch.setenv("ENABLE_AI_DAM", value)

    assert smartgallery_ai._env_bool("ENABLE_AI_DAM", "true") is expected


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_still_means_the_default(monkeypatch, blank):
    monkeypatch.setenv("ENABLE_AI_DAM", blank)

    assert smartgallery_ai._env_bool("ENABLE_AI_DAM", "true") is True


def test_the_layer_stays_on_when_the_switch_is_misspelt(monkeypatch, tmp_path):
    """End to end: the config the app actually builds."""
    monkeypatch.setenv("ENABLE_AI_DAM", "ture")
    monkeypatch.setenv("AI_DAM_AUTO_PROVISION", "flase")

    config = smartgallery_ai.AIConfig.from_env(str(tmp_path), str(tmp_path / "g.sqlite"))

    assert config.enabled is True, "a typo disabled the whole AI layer"
    assert config.auto_provision is True, "a typo disabled provisioning"


def test_a_deliberate_off_is_still_honoured(monkeypatch, tmp_path):
    """Nobody may lose the ability to switch the layer off."""
    monkeypatch.setenv("ENABLE_AI_DAM", "false")

    config = smartgallery_ai.AIConfig.from_env(str(tmp_path), str(tmp_path / "g.sqlite"))
    assert config.enabled is False
