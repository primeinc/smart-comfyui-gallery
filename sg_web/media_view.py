"""One media item, one address, three ways of looking at it.

`/i/{slug}` is the item's ONLY address, whatever its kind -- image,
animated image, video, audio, document are peers here exactly as they
are in the ResultSet; a picture-shaped module would reintroduce the
special-case worldview the schema deleted. The path is the entity; the
query string is the browsing context -- the GalleryQuery the viewer is
walking, carried whole so a reload, a pasted link or a fresh tab can
reconstruct previous/next without any client state owning what "next"
means.

The presentations of that one address, negotiated deterministically
(and declared with `Vary: Accept, HX-Request`, because caches punish
ambiguity):

    Accept names application/json      -> the MediaSurface itself
    else HX-Request: true              -> the lightbox fragment
    else Accept names text/html        -> the full detail page
    else (wildcard, machine default)   -> the MediaSurface itself

All from ONE view assembly. There is no /lightbox/ URL and no second
definition of the item.

Previous and next come from `resultset.neighborhood` -- locate plus
the window around it, so the arrows and the strip beneath the picture
are one read of one answer. Position in the
answer being walked, never a folder walk: a bare `/i/{slug}` uses the
default GalleryQuery (the whole library, newest first), so the arrows
always mean something and always mean what the grid meant. The context
carries the ResultSet currency as concurrency evidence and the
computed return-to-results URL, so closing a directly-opened item goes
back to its results page instead of wherever the browser had been.
The `X-SG-Expect` header works as the rail's `expect` does: a context
from a superseded generation is refused with 409, never silently
swapped under the gallery still on screen.
"""

from __future__ import annotations

import pathlib
import time
from typing import Literal

from litestar import MediaType, Request, get
from litestar.datastructures import State
from litestar.exceptions import HTTPException, NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromPath, FromQuery
from litestar.response import Redirect, Response, Template

from db import authored, connect, derived, naming, pages, places, resultset, settings
from db.resultset import MediaKind, canonical
from sg_web import home
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import presented
from sg_web.wire import Wire


def view(
    conn, models_dir: str, file_id: int, slug: str, query: resultset.GalleryQuery, now: float, actor_id: int
) -> MediaSurface:
    """The MediaSurface: everything every presentation shows, assembled
    once, inside ONE database snapshot.

    Ordering inside the snapshot is load-bearing: the ResultSet context
    comes FIRST, because its currency read (the monitor connection)
    must precede the read that pins this connection's snapshot -- a
    metadata read first would pin the snapshot before the currency was
    taken, and a commit in the gap would label pre-commit data with a
    post-commit currency: exactly the mislabeling the caller's 409
    comparison exists to catch. The context's currency is therefore
    always present -- from `neighborhood` when the item is in the answer,
    from `describe` (same projection, same snapshot) when it is not.
    """
    with resultset.snapshot(conn):
        # `neighborhood` IS locate plus the window around it, from one
        # projection: the ordinal, the arrows and the strip beneath the
        # picture are then the same answer at the same generation. Two
        # calls would be two chances to describe two different walks.
        found = resultset.neighborhood(conn, models_dir, query, file_id, now, actor_id=actor_id)
        if found is not None:
            generation, asked, answer = found["currency"], found["qs"], found["answer"]
        else:
            shape = resultset.describe(conn, models_dir, query, now, actor_id=actor_id)
            generation, asked, answer = shape["currency"], shape["qs"], shape["answer"]
        told = pages.picture(conn, file_id)
        if told is None:
            raise NotFoundException(f"/i/{slug} has no file row")
        return _assembled(conn, file_id, slug, found, generation, asked, answer, told, actor_id)


def _assembled(
    conn, file_id: int, slug: str, found, generation: str, asked: str, answer: str, told, actor_id: int
) -> MediaSurface:
    name, folder, width, height, duration, asked_w, checkpoint, missing_since, prompt, seed, fields, kind, read = told
    back = f"/g?{asked}" if asked else "/g"
    if found is not None and found["page"] > 1:
        back += ("&" if asked else "?") + f"page={found['page']}"
    loras = pages.file_loras(conn, file_id)
    recipe = pages.generation_of(conn, file_id)
    # Every prompt role in one read (db/prompts.py ROLES), so the panel can
    # show what was TYPED beside what the sampler actually saw -- they
    # differ exactly when something expanded wildcards, which is when the
    # person branching the picture wants the one they can edit.
    typed = pages.prompt_texts(conn, file_id)
    # A recipe is the reason there is a Creation at all: a photograph was
    # taken, and giving it an empty prompt/checkpoint/seed block would be
    # a section the page renders for every picture that has none.
    made = (
        prompt is not None
        or checkpoint is not None
        or seed is not None
        or bool(loras)
        or asked_w is not None
        or bool(recipe)
    )
    return MediaSurface(
        slug=slug,
        name=name,
        present=missing_since is None,
        stage=_stage(slug, kind, width, height, duration),
        context=BrowsingContext(
            qs=asked,
            in_answer=found is not None,
            return_url=back,
            currency=generation,
            answer=answer,
            ordinal=found["ordinal"] if found else None,
            page=found["page"] if found else None,
            total=found["total"] if found else None,
            previous=found["previous"] if found else None,
            next=found["next"] if found else None,
            first=found["first"] if found else None,
            last=found["last"] if found else None,
            filmstrip=_filmstrip(found, asked) if found else None,
        ),
        when=_when(conn, file_id),
        where=where_of(conn, file_id),
        faces=_faces(conn, file_id),
        said=_said(conn, file_id, actor_id),
        said_first=derived.said_first(conn, [file_id], prefer=settings.value(conn, "caption_model")).get(file_id),
        creation=(
            Creation(
                tool=recipe.get("tool"),
                prompt=prompt,
                original=typed.get("original"),
                negative=typed.get("negative"),
                original_negative=typed.get("original_negative"),
                checkpoint=checkpoint,
                loras=[Weighted(name=name, weight=weight) for name, weight in loras],
                seed=seed,
                steps=recipe.get("steps"),
                cfg=recipe.get("cfg"),
                denoise=recipe.get("denoise"),
                clip_skip=recipe.get("clip_skip"),
                sampler=recipe.get("sampler"),
                scheduler=recipe.get("scheduler"),
                asked_for_width=asked_w,
                asked_for_height=recipe.get("height"),
            )
            if made
            else None
        ),
        file=FileFacts(folder=folder, read=read, fields=fields),
        lineage=Lineage(
            copies=[
                Copy(slug=s, name=n, distance=d, is_best=bool(b)) for s, n, d, b in pages.dupe_copies(conn, file_id)
            ],
            parents=[Relative(slug=s, name=n, kind=k) for s, n, k in pages.parents(conn, file_id)],
            children=[Relative(slug=s, name=n, kind=k) for s, n, k in pages.children(conn, file_id)],
        ),
        params=[ParamRow(source=source, key=key, value=value) for source, key, value in pages.fields_of(conn, file_id)],
        place_choices=PlaceChoices(
            named=[PlaceNamed(name=name, kind=kind) for name, kind in pages.places_named(conn)],
            kinds=list(places.KINDS),
        ),
        # The actor's authored facts ride the SAME snapshot as everything
        # else -- a favorite committed mid-request must not appear under
        # the metadata of the generation before it.
        authored=_authored(conn, file_id, actor_id),
        viewing=_viewing(conn),
    )


def _filmstrip(found: dict, asked: str) -> Filmstrip:
    """The window the ResultSet returned, as addresses.

    Every href carries the walked question, spelled by the server that
    already knows its canonical form. A strip handing the browser slugs
    would be asking it to rebuild browsing state from parts, which is
    how a second, disagreeing ordering gets born.
    """
    from vision import thumbs

    return Filmstrip(
        first_ordinal=found["first_ordinal"],
        last_ordinal=found["last_ordinal"],
        total=found["total"],
        items=[
            FilmstripItem(
                slug=near["slug"],
                name=near["name"],
                kind=near["kind"],
                ordinal=near["ordinal"],
                href=f"/i/{near['slug']}" + (f"?{asked}" if asked else ""),
                # None for a kind with no picture to take. The raster
                # routes refuse audio and documents outright
                # (`_variant_bytes`: "a {kind} has no {variant}"), so a
                # walk through a mixed library used to emit a 404 for
                # every such member and draw a broken image where one
                # was. The strip says the kind instead.
                thumb=thumbs.asset_url(near["sha"], near["slug"], medium=near["kind"]),
            )
            for near in found["items"]
        ],
    )


def _viewing(conn) -> Viewing:
    """The run's viewer preferences, read on the same snapshot as the rest.

    Validated rather than trusted: `settings.put` refuses a value outside
    the registry's choices, so the column cannot hold anything else -- but
    a database edited by hand can, and the seam is where a bad value
    becomes a 500 here instead of a modifier that silently does nothing in
    somebody's browser.
    """
    held = settings.value(conn, "viewer_wheel_modifier")
    if held not in settings.WHEEL_MODIFIERS:
        raise ValueError(f"viewer_wheel_modifier is {held!r}, not one of {', '.join(settings.WHEEL_MODIFIERS)}")
    return Viewing(wheel_modifier=held)


def _authored(conn, file_id: int, actor_id: int) -> AuthoredState:
    """This actor's own facts about this picture.

    Spelled field by field rather than validated from `asdict`: the
    dataclass holds its collections as a TUPLE (immutable state nobody
    edits in place), and the seam is strict, so the translation from
    storage's shape to the wire's is a line of Python somebody can read
    -- which is what sg_web/wire.py asks for.
    """
    held = authored.media_state(conn, file_id, actor_id)
    return AuthoredState(
        favorite=held.favorite,
        rating=held.rating,
        collections=[CollectionSummary(slug=one["slug"], name=one["name"]) for one in held.collections],
        tags=[TagSummary(tag=one["tag"], label=one["label"]) for one in held.tags],
    )


#: The place vocabulary is the schema's: db/schema.sql constrains place.kind
#: with CHECK, and db/places.py KINDS is the same list. Stating it here as a
#: Literal carries the closed set across the wire, so a body naming a kind no
#: place can ever be is a 400 at the seam rather than a row the database
#: refuses later, and the browser gets a union instead of `string`.
PlaceKind = Literal["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"]


class Where(Wire):
    """Where the picture happened, as the current interpretation holds it."""

    id: int
    slug: str
    name: str
    kind: PlaceKind
    #: a person's word, or nothing yet -- GPS alone names no place
    basis: str | None
    #: the place and every ancestor, leaf first: "Lisbon, Portugal"
    chain: list[str]
    #: the gallery's query string for everything there
    qs: str
    #: the same place on the timeline, as a whole address
    timeline: str


def where_of(conn, file_id: int) -> Where | None:
    """Where the picture happened: the place by address, its kind, the
    basis, and the links to everything there. None when nobody has said."""
    import urllib.parse

    from db import facets

    row = pages.media_place(conn, file_id)
    if row is None:
        return None
    place_id, slug, name, kind, basis = row
    spelled = urllib.parse.urlencode([("f", facets.spell(facets.facet("place.id", "eq", str(place_id))))])
    return Where(
        id=place_id,
        slug=slug,
        name=name,
        kind=kind,
        basis=basis,
        chain=[one["name"] for one in places.chain(conn, place_id)],
        qs=spelled,
        timeline=f"/timeline?{spelled}",
    )


class Pixels(Wire):
    """A size in real pixels, as the picture is SEEN."""

    width: int
    height: int


def _source(width: int | None, height: int | None) -> Pixels | None:
    """The source's pixels, as displayed. Recorded upright already.

    NOT turned again here. A phone stores its sensor's frame with a tag
    saying which quarter turn it needs, and ingest resolves that ONCE at
    the point of record: db/ingest.py, `if found.orientation in
    TRANSPOSED: UPDATE file SET width = height, height = width`. Every
    renderer downstream agrees with the row -- the preview is rendered
    from `oriented.for_model` (sg_web/app.py:810), and a browser turns
    the original itself because `image-orientation: from-image` is the
    initial value. A swap here would be the second one, and would file
    every portrait photograph as landscape again.

    None when nothing recorded a size -- video, whose dimensions wait on
    the same probe `duration` does, and anything ingest has not read.
    """
    if not width or not height:
        return None
    return Pixels(width=width, height=height)


def _contained(source: Pixels, edge: int) -> Pixels:
    """What `ImageOps.contain` makes of this size against a square box.

    The renderer's own arithmetic, restated so the server can say what
    the browser will receive instead of the browser measuring it
    (python-pillow/Pillow@bb1d8e8 src/PIL/ImageOps.py:272-299). Note that
    `contain` RESIZES rather than shrinks: it computes the fitted size
    and calls `resize` unconditionally, so a 400x300 source becomes a
    1440x1080 preview. A viewer that promoted to the original on
    "displayed pixels exceed the preview's" would fetch FEWER pixels for
    every small picture in the library; `promotable` below is the
    comparison that actually holds.
    """
    if source.width >= source.height:
        return Pixels(width=edge, height=round(source.height / source.width * edge))
    return Pixels(width=round(source.width / source.height * edge), height=edge)


class ImageStage(Wire):
    """A still picture: the one kind a viewer zooms into.

    `src` is painted first because it is small and cached; `original` is
    the file itself. Both are rendered upright by the browser, so both
    agree with `source`.
    """

    kind: Literal["image"]
    #: the derived preview -- fast, cached, and what the grid already warmed
    src: str
    #: the bytes on disk, promoted to when the source has more to give
    original: str
    #: the file's own pixels, upright; None when ingest has not read it
    source: Pixels | None
    #: what `src` delivers -- stated, not measured (see `_contained`)
    shown: Pixels | None
    #: whether `original` holds more pixels than `src`. False for a source
    #: smaller than the preview box, which the preview UPSCALED.
    promotable: bool


class AnimatedStage(Wire):
    """A picture that moves: painted from the original, because a preview
    is one frame of it. Its poster is that frame, for a strip or a card."""

    kind: Literal["animated_image"]
    src: str
    original: str
    poster: str
    source: Pixels | None


class VideoStage(Wire):
    """A clip: the element seeks over the original through Range, so the
    preview is only ever its poster."""

    kind: Literal["video"]
    src: str
    original: str
    poster: str
    source: Pixels | None
    #: seconds, when the container states one
    duration: float | None


class SoundStage(Wire):
    """Audio: nothing to show, something to play."""

    kind: Literal["audio"]
    src: str
    original: str
    duration: float | None


class DocumentStage(Wire):
    """A document: the browser's own PDF renderer is the viewer.

    No raster preview exists to take (sg_web/app.py `_variant_bytes` says
    so), but /media types the bytes `application/pdf` from the sniff, so
    an embedded frame displays the file itself."""

    kind: Literal["document"]
    #: what the embedded frame is fed -- the original, typed by its bytes
    src: str
    original: str


#: What the viewer paints, per kind.
#:
#: A plain union, as CollectionDocument is: litestar builds a union's schema
#: itself and never asks pydantic, so `Field(discriminator=...)` would be an
#: annotation lying about the document. Every arm states `kind` as a
#: single-valued Literal, which is what the browser narrows on -- and what
#: makes `source`, `shown` and `promotable` reachable ONLY where they mean
#: something, with no assertion at the call site.
Stage = ImageStage | AnimatedStage | VideoStage | SoundStage | DocumentStage


def _stage(slug: str, kind: MediaKind, width, height, duration) -> Stage:
    """The display facts for one file, from the row already read.

    No query of its own: everything here is arithmetic over the picture
    row and the variant policy, so a viewer never measures what the
    server can state.
    """
    from vision import thumbs

    source = _source(width, height)
    original, poster, preview = f"/media/{slug}", f"/preview/{slug}", f"/preview/{slug}"
    if kind == "image":
        shown = _contained(source, thumbs.EDGES["preview"]) if source else None
        return ImageStage(
            kind="image",
            src=preview,
            original=original,
            source=source,
            shown=shown,
            promotable=bool(source and shown and source.width > shown.width),
        )
    if kind == "animated_image":
        return AnimatedStage(kind="animated_image", src=original, original=original, poster=poster, source=source)
    if kind == "video":
        return VideoStage(
            kind="video", src=original, original=original, poster=poster, source=source, duration=duration
        )
    if kind == "audio":
        return SoundStage(kind="audio", src=original, original=original, duration=duration)
    return DocumentStage(kind="document", src=original, original=original)


class Person(Wire):
    """Somebody the primary clustering puts in this picture."""

    slug: str
    name: str | None
    href: str
    #: how many detected faces here are theirs
    faces: int


class FaceScan(Wire):
    """One detector's pass over these bytes -- so "nobody here" is
    distinguishable from "nobody looked"."""

    model_id: str
    model_version: str
    faces: int
    at: float


class Faces(Wire):
    people: list[Person]
    looked: list[FaceScan]


#: What a model can say about a picture. `derived_annotation.kind` carries
#: a CHECK of exactly this list (db/schema.sql).
#:
#: Named by table rather than by line: this said :1468, which is inside
#: `derived_face_instance` -- a different table about a different thing.
SaidKind = Literal["caption", "description", "alt_text", "tag", "ocr", "title"]


class Said(Wire):
    """One thing a model said, with the evidence to weigh it by."""

    id: int
    kind: SaidKind
    text: str
    confidence: float | None
    model_id: str
    model_version: str
    region_id: int | None
    sample_id: int | None
    #: where in a clip it was said, when it was said of a frame
    offset_ms: int | None
    #: what THIS actor said about it: right, wrong, unsure, or nothing.
    #: Never a machine's confidence -- `confidence` above is that, and
    #: the two being one field is how a model's certainty and a person's
    #: judgement come to be averaged together.
    verdict: Literal["right", "wrong", "unsure"] | None = None
    #: said of bytes this file no longer has
    stale: bool


class WhenSession(Wire):
    """A session this picture belongs to, and the ways in to it."""

    id: int
    kind: str
    start: float
    end: float
    pictures: int
    qs: str
    story: str | None
    timeline: str


class When(Wire):
    """The picture's place on the human timeline, with its evidence.

    Two clock domains, never fused: a wall clock when that is what was
    claimed, an instant only when one is knowable (`domain` says which).
    """

    moment: float
    local_at: float | None
    instant_at: float | None
    tz_offset_min: int | None
    domain: Literal["wall", "instant"]
    precision: str
    basis: str
    certainty: float
    supports: list[str]
    conflicts: list[str]
    origin: str
    local_day: str
    day_qs: str
    timeline: str
    sessions: list[WhenSession]


class Copy(Wire):
    """Another body of the same picture."""

    slug: str
    name: str
    distance: int
    is_best: bool


class Relative(Wire):
    """A picture this one came from, or made."""

    slug: str
    name: str
    kind: str


class Lineage(Wire):
    """Where this picture sits among its own copies and derivations."""

    copies: list[Copy]
    parents: list[Relative]
    children: list[Relative]


class ParamRow(Wire):
    """One parsed field, exactly as it was read."""

    source: str
    key: str
    value: str


class Weighted(Wire):
    """One model applied at a strength. The strength is the point: a LoRA
    named without its weight is not enough to make the picture again."""

    name: str
    weight: float | None


class Creation(Wire):
    """How the picture was MADE, when a recipe was recorded. None on a
    photograph, which was taken rather than generated.

    This carries the whole REPRODUCTION RECIPE and not a summary of it.
    The generation row has held seed, steps, cfg, denoise, clip_skip,
    sampler, scheduler and size since it was written; the viewer showed
    five fields and none of those, so the one question a person actually
    has about a generated picture -- how do I make this again, or make it
    slightly differently -- could not be answered from the page that
    exists to answer it.
    """

    #: What made it: "swarm", "comfy", "a1111"...
    tool: str | None
    #: The prompt as the sampler saw it, after wildcards and substitution.
    prompt: str | None
    #: What was typed, when the tool recorded both. Different from
    #: `prompt` exactly when something expanded it, which is when a person
    #: branching the picture wants the one they can edit.
    original: str | None
    negative: str | None
    original_negative: str | None
    checkpoint: str | None
    loras: list[Weighted]
    seed: int | None
    steps: int | None
    cfg: float | None
    denoise: float | None
    clip_skip: int | None
    sampler: str | None
    scheduler: str | None
    #: what the recipe ASKED for, which the file need not have obeyed
    #: (db/schema.sql: generation.width is the request, file.width the fact)
    asked_for_width: int | None
    asked_for_height: int | None


class FileFacts(Wire):
    """The file as a file."""

    folder: str
    #: whether this metadata was read from the CURRENT bytes
    read: Literal["current", "stale", "never"]
    #: how many parsed fields there are, before anybody asks for them
    fields: int


class FilmstripItem(Wire):
    """One member of the walk near the picture on screen, as an address.

    Deliberately not the gallery's ResultItem: that carries a uuid for
    selection and a model caption for the grid's hover, and a strip of
    64px squares needs neither. `href` arrives whole, carrying the walked
    question, because reconstructing a browsing URL from a slug is the
    one thing the browser must never be asked to do here.
    """

    slug: str
    name: str
    kind: MediaKind
    #: position in the WHOLE answer, not in this window
    ordinal: int
    href: str
    #: None for a kind with no picture to take -- audio, documents. The
    #: strip is still a walk through THIS answer, and those files are
    #: members of it, so the cell is drawn saying its kind rather than
    #: dropped: a walk that skips its own members is a different walk.
    thumb: str | None


class Filmstrip(Wire):
    """The local stretch of the walk, in answer order.

    A window around the current item, not a second gallery: the rail on
    the results page already owns long-distance travel through the same
    answer, and this owns the few pictures either side of the one being
    looked at. It knows nothing about pages -- a window straddling a page
    boundary is not a special case here (db/resultset.py neighborhood),
    and it is the ANSWER's order, never a folder's.
    """

    first_ordinal: int
    last_ordinal: int
    total: int
    items: list[FilmstripItem]


class BrowsingContext(Wire):
    """The walk this address was opened inside.

    The ResultSet's, never a folder's: the arrows mean what the grid
    meant, and the currency is the concurrency evidence a mounted
    overlay is checked against.
    """

    qs: str
    in_answer: bool
    return_url: str
    currency: str
    answer: str
    #: present only when the item is IN the answer being walked
    ordinal: int | None
    page: int | None
    total: int | None
    previous: str | None
    next: str | None
    #: the two ENDS of this answer, so a walk that WRAPS has an address
    #: to wrap to. Whether it wraps is how a person arranged their viewer
    #: (frontend/src/workspace.ts) -- it changes no membership and is in
    #: no fingerprint; what the server owes is where the ends are.
    first: str | None
    last: str | None
    #: the few members either side of this one, or None when the item is
    #: not in the answer being walked -- the query defines the walk, and
    #: there is no other stretch of walk to invent for it
    filmstrip: Filmstrip | None


class PlaceNamed(Wire):
    name: str
    kind: PlaceKind


class PlaceChoices(Wire):
    """What the "where did this happen?" form offers: the places already
    named in this library, and the whole closed vocabulary of kinds."""

    named: list[PlaceNamed]
    kinds: list[PlaceKind]


class CollectionSummary(Wire):
    """One collection a file is filed in: its address and its name."""

    slug: str
    name: str


class TagSummary(Wire):
    """One keyword on a file: the normalised identity a filter is built
    from, and the spelling to put on the screen."""

    tag: str
    label: str


class AuthoredState(Wire):
    """What this actor has written down about one file, as opposed to
    what was derived from it.

    db/authored.py's MediaAuthoredState says `collections: tuple[dict,
    ...]` with the keys in a comment. This is the same fact with the
    comment promoted into the type, because the browser is given this
    one. Lives here rather than beside the write routes because the
    surface carries it too, and sg_web/media_authored.py already depends
    on this module for Where and PlaceKind -- one owner, and the arrow
    already pointed this way.
    """

    favorite: bool
    rating: int | None
    collections: list[CollectionSummary]
    #: Shared rather than this actor's, unlike everything above it: a
    #: keyword is a fact about the picture that everybody reads, where a
    #: rating is one person's opinion and two people may hold different
    #: ones. It rides in the actor's state anyway because it is authored
    #: and this is where the surface already looks for authored things.
    tags: list[TagSummary]


class Viewing(Wire):
    """How this run has asked the viewer to behave.

    NOT a fact about the picture -- a fact about the person looking at
    it, and the only one the viewer cannot decide for itself. Everything
    else it does (zoom, pan, which panel is open, whether the chrome is
    hidden) is what somebody is doing right now and is never stored;
    this is here because "which key walks the library" has three
    still-valid answers and no way to derive one.
    """

    wheel_modifier: settings.WheelModifier


class MediaSurface(Wire):
    """One media item, as every presentation of `/i/{slug}` reads it.

    Grouped by what a reader is looking FOR, not by which table a column
    came from: the stage is what to paint, `when`/`where`/`faces`/`said`
    are the human context that changes what the picture MEANS, and
    creation, file, lineage and params are the provenance behind it.
    """

    slug: str
    name: str
    #: whether the bytes are on disk right now
    present: bool
    stage: Stage
    context: BrowsingContext
    when: When | None
    where: Where | None
    faces: Faces
    said: list[Said]
    #: the one caption a strip or a bar shows, by the preferred model
    said_first: str | None
    creation: Creation | None
    file: FileFacts
    lineage: Lineage
    params: list[ParamRow]
    place_choices: PlaceChoices
    authored: AuthoredState
    viewing: Viewing


def faces_of(conn, file_id: int) -> Faces:
    """Who is in the picture, for a surface that just changed it.

    The same read the page assembles from, exported so a write route
    answers with what the database now holds rather than with what the
    caller asked for -- the two differ the moment anything else has a
    say, and a browser that drew the difference would be inventing state.
    """
    return _faces(conn, file_id)


def _faces(conn, file_id: int) -> Faces:
    """Who is in the picture, by the primary clustering, and whether any
    detector has looked at its current bytes -- so the page can tell
    "nobody here" from "nobody looked"."""
    return Faces(
        people=[
            Person(slug=slug, name=name, href=f"/p/{slug}", faces=int(count))
            for slug, name, count in pages.media_people(conn, file_id)
        ],
        looked=[
            FaceScan(model_id=model_id, model_version=version, faces=int(faces), at=at)
            for model_id, version, faces, at in pages.media_face_scans(conn, file_id)
        ],
    )


def _said(conn, file_id: int, actor_id: int | None = None) -> list[Said]:
    """What models have said, and what this actor said back.

    Translated at the seam: SQLite answers the staleness comparison with
    0 or 1, and the browser is promised a boolean (sg_web/wire.py).

    `verdict` rides along because a control that opens blank over a
    judgement somebody already made asks them to make it again -- and
    the second click would then be read as a change of mind.
    """
    return [
        Said.model_validate(
            {
                **row,
                "stale": bool(row["stale"]),
                "verdict": authored.standing_verdict(
                    conn, file_id, row["kind"], row["model_id"], row["model_version"], actor_id
                ),
            }
        )
        for row in derived.said_about(conn, file_id)
    ]


def _when(conn, file_id: int) -> When | None:
    """The picture's place on the human timeline, with the evidence
    behind it (db/pages.py MEDIA_WHEN), and the current sessions it
    belongs to -- each a link to the timeline, the day, the session's
    pictures and its story. None while the file is uninterpreted."""
    import json
    import urllib.parse

    from db import facets

    row = pages.media_when(conn, file_id)
    if row is None:
        return None
    local_at, instant_at, tz, basis, certainty, supports, conflicts, precision, origin, moment, local_day = row
    day_qs = urllib.parse.urlencode(
        [("f", facets.spell(facets.facet("context.local_day", "eq", local_day))), ("sort", "moment")]
    )
    day_lo, day_hi = pages.binned("day", moment, moment)
    return When(
        moment=moment,
        local_at=local_at,
        instant_at=instant_at,
        tz_offset_min=tz,
        domain="wall" if local_at is not None else "instant",
        precision=precision,
        basis=basis,
        certainty=certainty,
        supports=json.loads(supports) if supports else [],
        conflicts=json.loads(conflicts) if conflicts else [],
        origin=origin,
        local_day=local_day,
        day_qs=day_qs,
        timeline="/timeline?" + urllib.parse.urlencode({"bin": "hour", "start": day_lo, "end": day_hi}),
        sessions=[
            WhenSession(
                id=event_id,
                kind=kind,
                start=start,
                end=end,
                pictures=pictures,
                qs=urllib.parse.urlencode(
                    [("f", facets.spell(facets.facet("event.id", "eq", str(event_id)))), ("sort", "moment")]
                ),
                story=f"/stories/renders/{render_id}" if render_id is not None else None,
                # the session's hour window on the timeline: where its
                # story is told (the timeline owns the tell button)
                timeline="/timeline?" + urllib.parse.urlencode(pages.hour_window_qs(start, end)),
            )
            for event_id, kind, start, end, pictures, render_id in pages.media_sessions(conn, file_id)
        ],
    )


@get(
    "/i/{slug:str}",
    sync_to_thread=True,
    responses={
        200: ResponseSpec(
            data_container=MediaSurface,
            media_type=MediaType.JSON,
            description="the media surface, for a machine or the browser's viewer",
            # deterministic, and a page of them doubles the document for
            # nothing the generated types read
            generate_examples=False,
        )
    },
)
def media_page(
    state: State,
    request: Request,
    slug: FromPath[str],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    sort: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
) -> Template | Response | Redirect:
    """One media item at its address, presented for whoever is asking.

    The overlay's currency expectation arrives OUT-OF-BAND in the
    `X-SG-Expect` header -- never in the URL, which stays the canonical
    context the browser may push, share or reload. The facets ride the
    URL like every other part of the question: a picture opened from a
    place's link walks that place, not the library."""
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
    )
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "file", slug)
        if found is None:
            raise NotFoundException(f"no file at /i/{slug}")
        file_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, file_id)
            if live is not None:
                # The context survives the rename: the address moved, the
                # walk the viewer was on did not.
                asked = canonical(query)
                return Redirect(path=f"/i/{live[1]}" + (f"?{asked}" if asked else ""), status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, file_id, slug, query, time.time(), state.actor_id)
        conn.commit()  # a semantic context may have minted registry rows
        # Compared AFTER assembly, against the currency the view was
        # actually located in: a commit landing mid-request would
        # otherwise pass a pre-assembly check and hand back arrows from
        # a newer answer under the old mounted gallery.
        expected = request.headers.get("x-sg-expect")
        if expected is not None and told.context.currency != expected:
            raise HTTPException(status_code=409, detail="the result set has changed; redraw the gallery")
    finally:
        connect.close(conn)
    return presented(request, told, page="media.html", fragment="_media_lightbox.html", name="item")
