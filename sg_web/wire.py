"""The one policy every JSON contract at the HTTP seam obeys -- and nothing else.

The contracts themselves live with the modules whose routes they belong to,
the way AlbumEntry sits above the album routes it is the body of. What is
shared is a single decision, stated once here.

`extra="forbid"` works in both directions. A request body carrying a field
nobody asked for is a 400 at the seam instead of a key some handler silently
ignored. A response built from a row that grew a column fails where the
contract is written instead of shipping a shape the browser was never
promised.

`strict=True` means the seam translates rather than coerces. Litestar
decodes a body with `model_validate(value, strict=...)` and that argument
beats this config, so the flag is only live while the application registers
`PydanticInitPlugin(validate_strict=True)` -- which `plugins()` below does. It is not
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

import typing

from litestar.openapi.spec import Schema
from litestar.plugins.pydantic import PydanticDIPlugin, PydanticInitPlugin, PydanticSchemaPlugin
from litestar.typing import FieldDefinition
from pydantic import BaseModel, ConfigDict, RootModel

if typing.TYPE_CHECKING:
    from litestar._openapi.schema_generation.schema import SchemaCreator
    from litestar.plugins import PluginProtocol


class Wire(BaseModel):
    """A JSON shape that crosses the HTTP seam, in either direction."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _WireSchema(PydanticSchemaPlugin):
    """The OpenAPI document says what the seam does.

    Litestar builds a component from a model's fields, required set, title
    and examples, and never looks at `extra`
    (litestar/plugins/pydantic/plugins/schema.py for_pydantic_model). The
    document that came out therefore admitted a key the server answers 400
    for, and the browser's generated types were built from that document.
    """

    @classmethod
    @typing.override
    def for_pydantic_model(cls, field_definition: FieldDefinition, schema_creator: SchemaCreator) -> Schema:
        model = field_definition.annotation
        if isinstance(model, type) and issubclass(model, RootModel):
            # A RootModel's JSON is its root, not `{"root": ...}`. Litestar
            # builds a component from a model's FIELDS, and a RootModel has
            # exactly one called `root`, so the document described a wrapper
            # object the server does not accept -- and the browser would
            # have been generated against it.
            held = model.model_fields["root"]
            inner = schema_creator.for_field_definition(FieldDefinition.from_annotation(held.annotation))
            return inner if isinstance(inner, Schema) else Schema(one_of=[inner])
        schema = super().for_pydantic_model(field_definition=field_definition, schema_creator=schema_creator)
        if isinstance(model, type) and issubclass(model, BaseModel) and model.model_config.get("extra") == "forbid":
            schema.additional_properties = False
        return schema


def plugins() -> list[PluginProtocol]:
    """The three pydantic plugins, registered by hand.

    `PydanticPlugin` would build its own schema plugin, so the parts are
    named individually; naming all three leaves every branch of Litestar's
    own default registration inert (litestar/app.py:561-572).
    """
    return [PydanticInitPlugin(validate_strict=True), _WireSchema(), PydanticDIPlugin()]
