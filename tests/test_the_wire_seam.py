"""The two facts about the JSON seam that only running it can show.

sg_web/wire.py is integration code: it decides how litestar validates a
body and how it describes one. Both decisions are invisible to ruff, to
pyright and to sglint -- they live in arguments litestar passes and in a
schema it builds -- and both have been wrong here in ways nothing else
caught.

`just api check` compares the generated document with the committed one,
so it sees the document CHANGE. It cannot see the document being wrong,
because it has nothing to compare the document against except itself.
That is what these hold.
"""

from __future__ import annotations

from typing import Annotated, Literal

from litestar import Litestar, post
from litestar.testing import TestClient
from pydantic import Field, RootModel

from sg_web import wire
from sg_web.wire import Wire


class Fine(Wire):
    kind: Literal["fine"]
    weight: int


class Coarse(Wire):
    kind: Literal["coarse"]
    lumps: int


class Either(RootModel[Annotated[Fine | Coarse, Field(discriminator="kind")]]):
    pass


@post("/probe", sync_to_thread=False)
def probe(data: Either) -> dict:
    return {"got": type(data.root).__name__}


def _served() -> Litestar:
    return Litestar(route_handlers=[probe], plugins=[*wire.plugins()])


def test_a_root_model_body_crosses_unwrapped_in_both_directions():
    """A RootModel's JSON is its root, and the document has to say so.

    Litestar builds a component from a model's FIELDS, and a RootModel has
    exactly one, called `root` -- so the document described
    `{"root": {...}}`, a wrapper the server does not accept, and
    openapi-typescript generated the browser against it. Nothing else
    would have noticed: the server was right, the document was wrong, and
    the drift gate only compares the document with itself.
    """
    app = _served()

    held = app.openapi_schema.to_schema()["paths"]["/probe"]["post"]["requestBody"]
    schema = held["content"]["application/json"]["schema"]
    assert "oneOf" in schema, schema
    assert {one["$ref"].rsplit("/", 1)[-1] for one in schema["oneOf"]} == {"Fine", "Coarse"}
    assert "root" not in str(schema), "the document must not describe the python wrapper"

    with TestClient(app=app) as client:
        told = client.post("/probe", json={"kind": "fine", "weight": 3})
        assert told.status_code == 201, told.text
        assert told.json() == {"got": "Fine"}, "the discriminator picked the arm"

        wrapped = client.post("/probe", json={"root": {"kind": "fine", "weight": 3}})
        assert wrapped.status_code == 400, "the wrapper the document used to describe is not accepted"


def test_a_contract_refuses_a_field_its_arm_does_not_name():
    """extra="forbid" reaches the arms of a discriminated union, and the
    refusal says which arm and which field.

    This is the whole reason a union beats one model holding both sets of
    fields: `{"kind": "fine", "lumps": 2}` is a mistake, and accepting it
    silently would report success for a request that meant something the
    server did not do.
    """
    with TestClient(app=_served()) as client:
        strange = client.post("/probe", json={"kind": "fine", "weight": 3, "lumps": 2})

        assert strange.status_code == 400, strange.text
        assert "fine.lumps" in strange.text, strange.text


def test_the_seam_translates_rather_than_coerces():
    """Wire's strict=True is only live while the application registers
    PydanticInitPlugin(validate_strict=True), because litestar passes
    `strict=` to model_validate and an explicit argument beats the model's
    own config. Lax, the 1 below arrives as the weight 1."""
    with TestClient(app=_served()) as client:
        coerced = client.post("/probe", json={"kind": "fine", "weight": "3"})

        assert coerced.status_code == 400, coerced.text
