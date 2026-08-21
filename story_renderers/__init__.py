"""Deterministic narration Adapters for the Renderer Seam.

`claims` turns one already-validated Claim into wording through a
CLOSED registry keyed by claim kind; `formatting` owns how counts,
days, percentages and name lists are spelled. Neither touches a
database, a model, or a template: they receive frozen values and
return strings. Jinja, elsewhere, lays the resulting StoryRender out.
"""
