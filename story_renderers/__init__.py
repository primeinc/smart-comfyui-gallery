"""Deterministic narration Adapters for the Renderer Seam.

`claims` turns one already-validated Claim into wording through a
CLOSED registry keyed by claim kind; `formatting` owns how counts,
days, percentages and name lists are spelled. Neither touches a
database, a model, or a template: they receive frozen values and
return strings. Jinja, elsewhere, lays the resulting StoryRender out.

POLICY_VERSION is the render policy -- ONE token for everything in this
package that decides output bytes: claim wording, which kinds a profile
surfaces, how a count, a day, a percent or a list of names is spelled.
Bump it for EVERY observable change, cosmetic or semantic: a render
request is content-addressed, and two code versions that would produce
different bytes for one request identity would make "same request, one
document" a lie.
"""

POLICY_VERSION = 7
