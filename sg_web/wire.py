"""The one policy every JSON contract at the HTTP seam obeys -- and nothing else.

The contracts themselves live with the modules whose routes they belong to,
the way AlbumEntry sits above the album routes it is the body of. What is
shared is a single decision, stated once here.

`extra="forbid"` works in both directions. A request body carrying a field
nobody asked for is a 400 at the seam instead of a key some handler silently
ignored. A response built from a row that grew a column fails where the
contract is written instead of shipping a shape the browser was never
promised.

`strict=True` means the seam translates rather than coerces. It is not
uniform strictness for its own sake: int widens to float even here, because
that loses nothing, but the integer 1 is NOT a boolean. SQLite stores a flag
as 0 or 1 and the browser is promised true or false, so that conversion is a
line of Python somebody can read rather than a rule pydantic applied because
the value happened to look convertible. Storage details stop at this seam.

This is deliberately not a place to accumulate types. A shape that crosses the
wire for one module is part of that module's interface, and deleting the
module should take its contract with it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Wire(BaseModel):
    """A JSON shape that crosses the HTTP seam, in either direction."""

    model_config = ConfigDict(extra="forbid", strict=True)
