"""Real photographs from Wikimedia Commons, chosen for metadata diversity.

The corpus needs files written by many camera bodies across many years. Two
earlier candidates failed to supply that and are recorded in
`docs/CORPUS_SOURCES.md`: `Cheliosoops/EXIF` ships EXIF-stripped 768x512
resizes, and `images9/flickr-cc-by` ships URL indexes whose images Flickr
re-encodes without EXIF.

Commons works because the API reports a file's EXIF before the bytes are
fetched. Selection targets diversity instead of hoping for it: query, read
Make/Model/DateTimeOriginal, keep the file only if its (maker, model, year) is
new, then download.

Every file keeps its Commons title, SHA-1, license and author, so the corpus
can be redistributed with attribution.

API contract read from the live server on 2026-08-25 via
`action=paraminfo&modules=query+imageinfo|query+categorymembers`, and from
`../refs/wikimedia/mediawiki`. Params used: iiprop, iilimit, cmtitle, cmtype,
cmlimit, cmsort, cmstart, cmend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import time

import httpx

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "commons"
LOCKFILE = REPO / "tests" / "commons.lock.json"

API = "https://commons.wikimedia.org/w/api.php"

#: Hosts this module is allowed to open. Commons serves its API from the
#: first and its files from the second, and nothing here has any business
#: opening anything else -- a URL arrives from a JSON response, so the
#: scheme and host are checked rather than assumed (ruff S310).
ALLOWED = ("https://commons.wikimedia.org/", "https://upload.wikimedia.org/")

#: What a per-file or per-category failure is allowed to be. Anything else
#: is a defect in this module rather than a fact about Commons, and is left
#: to propagate (ruff BLE001).
TROUBLE = (OSError, RuntimeError, ValueError, KeyError)

#: Commons rate-limits requests that do not carry a contact, and returned
#: HTTP 429 to a version of this module that carried none. The `+url` form
#: is the contact -- the Wikimedia UA policy accepts a reachable page in
#: place of an address.
AGENT = "smart-comfyui-gallery-corpus/1.0 (+https://github.com/primeinc/smart-comfyui-gallery)"

#: The makers to walk. Commons names these categories inconsistently -- some
#: are `Taken with X`, some `Photos taken with X` -- so the name is resolved at
#: run time rather than written down here and quietly missing.
MAKERS: tuple[str, ...] = (
    "Canon",
    "Nikon",
    "Sony",
    "Panasonic",
    "Olympus",
    "Fujifilm",
    "Pentax",
    "Samsung",
    "Apple Inc.",
    "Google",
    "Leica",
    "Kodak",
    "Casio",
    "Motorola",
    "Ricoh",
)

#: How a maker's category might be spelled, most common first. Commons is
#: not consistent about this and the list is what was observed, not a rule.
FORMS: tuple[str, ...] = (
    "Category:Photos taken with {}",
    "Category:Photos taken with {} cameras",
    "Category:Taken with {}",
    "Category:Photographs taken with {}",
)

#: Skip anything larger than this. A single 40 MB panorama buys one row in the
#: diversity table and spends what twenty ordinary photographs would.
MOST_BYTES = 12_000_000

#: Commons asks API clients to stay under one request per second.
PAUSE = 0.25


@dataclasses.dataclass(frozen=True)
class Picture:
    """One Commons file, with the provenance needed to redistribute it."""

    title: str
    url: str
    sha1: str
    bytes: int
    mime: str
    make: str
    model: str
    taken: str
    license: str
    artist: str

    @property
    def year(self) -> str:
        return self.taken[:4]

    @property
    def name(self) -> str:
        """A filename that keeps the Commons title recoverable.

        Commons titles legally contain characters Windows refuses, and one
        `[Errno 22] Invalid argument` per run is a file silently missing from
        the corpus. Only the reserved set is replaced, so the title still reads
        back.
        """
        held = self.title.removeprefix("File:")
        for bad in '<>:"/\\|?*':
            held = held.replace(bad, "_")
        return held.strip(". ")[:180]


def _fetched(url: str, timeout: int, params: dict[str, str] | None = None) -> httpx.Response:
    """GET a Commons address, with 429 backoff.

    The file URLs arrive inside a JSON response rather than from this file,
    so the host is checked before anything is fetched.

    Backs off on 429. Commons rate-limited a run that paced itself at a
    fixed interval: it kept 34 files and recorded 159 rate-limit failures,
    which read as six makers having no photographs when they had only been
    asked too fast. With the backoff on both the API calls and the
    downloads: 243 files, no rate-limit failures.
    """
    if not url.startswith(ALLOWED):
        raise ValueError(f"refusing to open {url!r}: not a Commons address")
    answer = httpx.get(url, params=params, headers={"User-Agent": AGENT}, timeout=timeout)
    for attempt in range(5):
        if answer.status_code != 429:
            break
        time.sleep(2 ** (attempt + 1))
        answer = httpx.get(url, params=params, headers={"User-Agent": AGENT}, timeout=timeout)
    answer.raise_for_status()
    time.sleep(PAUSE)
    return answer


def _get(params: dict[str, str]) -> dict:
    """One API call. Raises rather than returning an empty result, because an
    empty result that means "the request failed" reads as "there is nothing
    there" and silently shrinks the corpus.
    """
    answer = _fetched(API, 60, {**params, "format": "json", "formatversion": "2"})
    got = answer.json()
    if "error" in got:
        raise RuntimeError(f"{got['error'].get('code')}: {got['error'].get('info')}")
    return got


def category_for(maker: str) -> str | None:
    """The Commons category holding this maker's photographs, or None.

    None means every spelling in `FORMS` was asked for and none existed -- not
    that the request failed, which raises instead.
    """
    for form in FORMS:
        name = form.format(maker)
        got = _get({"action": "query", "titles": name, "prop": "categoryinfo"})
        pages = (got.get("query") or {}).get("pages") or []
        if pages and not pages[0].get("missing"):
            return name

    # A category can hold files without having a description page, and then
    # `categoryinfo` calls it missing. `Category:Taken with Casio` is one:
    # it has members and reported missing, so Casio and Kodak were recorded
    # as makers Commons does not photograph. Ask what categories actually
    # exist under the prefix instead of what pages do.
    for form in FORMS:
        prefix = form.format(maker).removeprefix("Category:")
        got = _get({"action": "query", "list": "allcategories", "acprefix": prefix, "aclimit": "1"})
        found = (got.get("query") or {}).get("allcategories") or []
        if found:
            first = found[0]
            name = first if isinstance(first, str) else first.get("category", "")
            if name:
                return f"Category:{name}"
    return None


def models_in(category: str, most: int = 40) -> list[str]:
    """The per-model subcategories under a maker's category.

    A maker category is mostly a container: `Photos taken with Canon` held 23
    subcategories and 9 loose files when this was written, so the files are one
    level down and each subcategory names a body.
    """
    got = _get(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "subcat",
            "cmlimit": str(most),
        }
    )
    return [one["title"] for one in (got.get("query") or {}).get("categorymembers") or []]


def _exif(info: dict) -> dict[str, str]:
    """EXIF as a plain dict. The API returns a list of {name, value} pairs."""
    return {one["name"]: one["value"] for one in (info.get("metadata") or [])}


def _extra(info: dict) -> dict[str, str]:
    """License and author. The API nests each under a "value" key."""
    held = info.get("extmetadata") or {}
    return {key: str(value.get("value", "")) for key, value in held.items()}


def candidates(category: str, most: int = 60) -> list[Picture]:
    """Files in `category` that report a camera, a model and a capture date.

    Files without those three cannot serve the diversity the corpus needs, so
    they are dropped here rather than downloaded and sorted out later.
    """
    listed = _get(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": str(most),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|metadata|extmetadata",
        }
    )
    out: list[Picture] = []
    for page in (listed.get("query") or {}).get("pages") or []:
        info = (page.get("imageinfo") or [{}])[0]
        if not info.get("url"):
            continue
        tags, extra = _exif(info), _extra(info)
        make, model = str(tags.get("Make", "")).strip(), str(tags.get("Model", "")).strip()
        taken = str(tags.get("DateTimeOriginal", "")).strip()
        if not (make and model and taken):
            continue
        if int(info.get("size") or 0) > MOST_BYTES:
            continue
        out.append(
            Picture(
                title=page["title"],
                url=info["url"].split("?")[0],
                sha1=info.get("sha1", ""),
                bytes=int(info.get("size") or 0),
                mime=info.get("mime", ""),
                make=make,
                model=model,
                taken=taken,
                license=extra.get("LicenseShortName", "unknown"),
                artist=extra.get("Artist", ""),
            )
        )
    return out


def choose(found: list[Picture], seen: set[tuple[str, str, str]]) -> list[Picture]:
    """Keep only files whose (maker, model, year) has not been kept yet.

    `seen` is carried across categories by the caller, so a body that appears
    in two categories is not downloaded twice.
    """
    keep: list[Picture] = []
    for one in sorted(found, key=lambda p: (p.model, p.taken)):
        key = (one.make.lower(), one.model.lower(), one.year)
        if key in seen:
            continue
        seen.add(key)
        keep.append(one)
    return keep


def download(one: Picture, into: pathlib.Path) -> pathlib.Path:
    """Fetch a file and check it against the SHA-1 the API reported.

    A file whose bytes do not match what the API described is not the file the
    provenance row describes, so it is removed rather than kept.
    """
    into.mkdir(parents=True, exist_ok=True)
    target = into / one.name
    if target.is_file() and target.stat().st_size == one.bytes:
        return target
    raw = _fetched(one.url, 300).content
    # SHA-1 because that is what the API reports, and this compares the
    # bytes to what it described. Not a security boundary -- the transport
    # is TLS and the checksum is an integrity check against a truncated or
    # substituted download.
    got = hashlib.sha1(raw, usedforsecurity=False).hexdigest()
    if one.sha1 and got != one.sha1:
        raise RuntimeError(f"{one.title}: sha1 {got} but the API said {one.sha1}")
    target.write_bytes(raw)
    return target


def fetch(models_per_maker: int = 12, files_per_model: int = 8) -> dict:
    """Walk maker -> model -> files, keep the diverse ones, download them.

    Every maker that yields nothing is written to `trouble` with the reason.
    A maker missing from `files` and absent from `trouble` would mean it was
    never asked for, which is the failure this records against.
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[Picture] = []
    trouble: list[dict[str, str]] = []
    walked: list[dict[str, object]] = []

    for maker in MAKERS:
        try:
            category = category_for(maker)
        except TROUBLE as why:
            trouble.append({"maker": maker, "why": f"lookup failed: {why}"})
            continue
        if category is None:
            trouble.append({"maker": maker, "why": "no category under any known spelling"})
            continue
        try:
            models = models_in(category)[:models_per_maker]
        except TROUBLE as why:
            trouble.append({"maker": maker, "category": category, "why": str(why)})
            continue

        before = len(kept)
        # The maker category holds a few loose files as well as the model
        # subcategories, so both are asked for.
        for where in [category, *models]:
            try:
                found = candidates(where, files_per_model)
            except TROUBLE as why:
                trouble.append({"category": where, "why": str(why)})
                continue
            for one in choose(found, seen):
                try:
                    download(one, IMAGES)
                except TROUBLE as why:
                    trouble.append({"title": one.title, "why": str(why)})
                    continue
                kept.append(one)
        walked.append({"maker": maker, "category": category, "models": len(models), "kept": len(kept) - before})

    held = {
        "what": "Real photographs from Wikimedia Commons, one per (maker, model, year).",
        "source": {"api": API, "makers": list(MAKERS), "walked": walked},
        "files": [dataclasses.asdict(one) for one in kept],
        "trouble": trouble,
    }
    LOCKFILE.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held


def available() -> bool:
    return IMAGES.is_dir() and any(IMAGES.iterdir())


if __name__ == "__main__":
    got = fetch()
    makers = sorted({one["make"] for one in got["files"]})
    # Only the four-digit ones. A blank `taken` sorts last and printed
    # "1964..None" as the span of the corpus, which is a wrong number in
    # the one place somebody reads to decide the corpus is broad enough.
    years = sorted(one["taken"][:4] for one in got["files"] if one["taken"][:4].isdigit())
    print(f"{len(got['files'])} files into {IMAGES}")
    print(f"makers: {len(makers)} distinct: {', '.join(makers)}")
    print(f"years:  {years[0]}..{years[-1]} ({len(set(years))} distinct)" if years else "years:  none")
    if got["trouble"]:
        print(f"trouble: {len(got['trouble'])} entries, see {LOCKFILE}")
