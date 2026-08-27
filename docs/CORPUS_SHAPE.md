# The shape

What the corpus must span, and what it must contain that is not media in
good standing. Written before acquisition, so coverage has a denominator
that means something.

Every axis below is READ OUT OF THE APPLICATION. None is typed by hand, so
none can be quietly shortened to match what was easy to find. Where a
vocabulary is a declared constant it is named; where it is a schema
constraint the table and column are named.

## 1. The shape

Seven axes. A corpus spans the shape when every value of every axis is
reached by a real file, or is BLOCKED_EXTERNALLY with evidence.

| axis | values | declared by |
|---|---|---|
| kind | 5 | `db/scan.py KIND_BY_SUFFIX` values |
| suffix | 99 | `db/scan.py KIND_BY_SUFFIX` keys |
| generator dialect | 8 | `metaparse/adapters.py MARKER_ADAPTERS + HEURISTIC_ADAPTERS` |
| time basis | 6 | `db/context.py TIME_BASES` |
| time precision | 7 | `db/schema.sql derived_media_context.time_precision` |
| location basis | 2 | `db/context.py LOCATION_BASES` |
| origin | 4 | `db/context.py ORIGINS` |

Spelled out, because these are the ones that get forgotten:

```text
kind             image  animated_image  video  audio  document
time basis       capture  embedded  filename  folder  btime  mtime
time precision   decade  year  month  day  minute  second  subsecond
location basis   gps  authored
origin           captured  generated  mixed  imported
dialect          A1111/Forge  SwarmUI  ComfyUI  NovelAI  Fooocus
                 InvokeAI  Easy Diffusion  Draw Things
```

**The time-basis axis is the one this corpus kept missing.** `capture` is
the easy rung. `btime` and `mtime` are the fallback every real library leans
on, because most files people own have no EXIF at all, and a corpus made of
camera output never exercises them. `filename` and `folder` are the rungs a
scanned or exported file lands on.

**Four values used to be listed here and no code could produce any of
them** -- `time_basis:first_seen`, `time_precision:hour`,
`location_basis:sidecar`, `location_basis:inferred`. They were removed from
the application in v44 (`db/migrate.py @step(43)`), not from this page
alone: a vocabulary that names unreachable values makes a reader believe a
query for one could return something, and it made this document describe a
shape no corpus could ever fill.

**`location_basis:authored` is not reachable by any file, and is not a
gap.** A person says where a picture happened; no bytes carry it. The
ledger measures it by performing the assertion (`tests/needs.py
INTERACTIONS`), which is a stronger check than a corpus file would be --
if `set_place` stops producing it, the measurement goes UNSATISFIED.

### EXIF writer variation, within the capture rung

Reaching `capture` once is not spanning it. `db/capture.py Capture` carries
21 fields and the branches that fill them differ by writer:

```text
tz_offset_min       present / absent      a 2001 body writes no zone
subsec_ms           present / absent      pre-2005 bodies have no subsecond
maker_tz_offset_min present / absent      the zone hidden in a MakerNote
orientation         1..8                  8 is a real sideways photograph
gps_lat/lon/alt     present / absent      and both hemispheres
body_serial         present / absent
lens                present / absent
camera              many makers, many models, many DECADES
unrecorded/homeless non-empty              tags the schema has no column for
unreadable          non-null              the reader's own refusal, recorded
```

## 2. What the shape is NOT

A corpus of files in good standing tests the happy path and nothing else.
These are not defects in the corpus; they are REQUIRED members of it, and a
corpus missing them is the one that lets a library die on the first odd file
somebody owns.

| negative case | why it must be present |
|---|---|
| no metadata at all | the ordinary fate of anything through a chat window or a screenshot |
| truncated to its header | a half-finished download, a bad card |
| bytes corrupt mid-file | a dying disk |
| metadata that LIES | EXIF date disagreeing with filename and mtime |
| conflicting dates across rungs | what `time_conflicts` exists to record |
| zero bytes | a file that exists and holds nothing |
| suffix disagreeing with content | a PNG named `.jpg`; a JPEG named `.jfif` |
| a declared suffix the readers refuse | e.g. LibRaw's answer to a stub |
| duplicate: same bytes, two names | the checksum case |
| duplicate: same capture, different bytes | RAW beside its out-of-camera JPEG; a checksum cannot see it |
| the same photo re-encoded | a resize or a re-save, which pHash must catch and sha256 cannot |
| a file the application declares unsupported | so refusal is exercised deliberately |
| non-ASCII and shell-hostile filenames | quotes, asterisks, RTL scripts, 200 characters |
| a very large file | the memory path |
| media with an alpha channel / odd bit depth | the decode path that is not 8-bit RGB |

**The shape is also not a format conformance suite.** 194 files at one file
per format, mostly undecodable, tests the readers and produces a gallery of
broken tiles. Both are needed and they are not the same artifact.

**And it is not a benchmark.** Nothing here is chosen for how long it takes
to process.

## 3. How a need closes

```text
DISCOVERED -> UNSATISFIED -> PARTIAL -> SATISFIED
                          -> BLOCKED_EXTERNALLY   (evidence required)
```

`DEFERRED`, `TODO`, `SKIPPED`, `SYNTHESIZED INSTEAD` and `TESTS PASS WITHOUT
IT` are not terminal states. `docs/CORPUS_CONTRACT.md` holds the rule; this
file holds the enumeration it applies to.

Measured by `tests/needs.py`, frozen in `tests/needs.lock.json`, and gated by
a test that fails when a declared value is reached by nothing — so the
number cannot be satisfied by writing a larger report.
