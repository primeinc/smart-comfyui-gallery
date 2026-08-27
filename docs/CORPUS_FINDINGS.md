# Corpus findings

Running `../sg-corpus` through the real ingestion path.
Contract: `docs/CORPUS_CONTRACT.md`.

```text
entry point   python -m tests.corpus_report C:/Users/will/dev/sg-corpus
repo          7cf254e99c6f  (branch frontend-typescript)
corpus        1064 files, 901 scan items
measured      2026-08-25
```

Pinned refs used as evidence:

```text
letmaik/rawpy                    326494be83cb
PyAV-Org/PyAV                    040da79f2ef3
exiftool/exiftool                2200871d9cef  (tag 13.59)
comfyanonymous/ComfyUI_examples  f9431bb000ce
```

Corpus bytes are never edited to clear a finding.

## Run — clean, after APP-1/2/3

`just corpus against C:/Users/will/dev/sg-corpus`, 2026-08-25 16:14:59-16:19:30.

```text
scanned      912 added, 912 hashed   (1064 files on disk)

jobs         #2 hash      901 items, 18 failed
             #3 scan      912 items, 15 failed
             #4 context   912 items,  0 failed
             #5 hash      901 items, 20 failed
             #6 hash        1 item,   0 failed

kinds        image 882, video 14, audio 9, animated_image 5, document 2
             with a recipe   247
             with a capture  596

producers    camera 143, workflow 89, checkpoint 49, lens 17, lora 10

duplicates   99 groups, 276 files

timeline     dated 912 of 912
             contested 269
             collapsed, whole library: 12 years, 4 years, 7 years, 5 years, 4 years
```

Every failure is a file the corpus holds on purpose. No item failure ended a
job. Before APP-1 the first job died at item 313 of 901.

Three runs were needed to reach this: each fix exposed the next escape.

```text
run 1  died at item 313   CanonRaw.cr2   LibRawFileUnsupportedError
run 2  died at item 348   Matroska.mkv   av.error.EOFError
run 3  died at item 370   QuickTime.heic builtins.EOFError (pillow_heif)
run 4  completed          20 item failures, 0 escapes
```

## Taxonomy

`ITEM_FAILURES` (introduced `7016dab`, `db/runner.py:36`) was
`(OSError, ValueError, RuntimeError, LookupError, sqlite3.Error)`.

Contract it enforces, `7016dab db/runner.py:9-15`: an unreadable file or
corrupt image is recorded on the item and the job carries on; anything else
propagates and ends the worker turn.

Three decoders raise outside that tuple. Each finding below is one of them.

## APP-1 — LibRawError ends the job

```text
status    FIXED
where     vision/decode.py open_still
class     LibRawError(Exception)          letmaik/rawpy@326494be83cb rawpy/_rawpy.pyx:346
escape    _thumbs_item -> run_next        7016dab db/runner.py:437,1924
blast     9 of 811 image-kind files; each alone ends a full-library scan
fix       LibRawError -> OSError at the decode boundary
```

| file | error |
|---|---|
| `exiftool/CanonRaw.cr2` | `LibRawFileUnsupportedError` |
| `exiftool/CanonRaw.cr3` | `LibRawIOError` |
| `exiftool/CanonRaw.crw` | `LibRawIOError` |
| `exiftool/FujiFilm.raf` | `LibRawIOError` |
| `exiftool/Minolta.mrw` | `LibRawIOError` |
| `exiftool/PhaseOne.iiq` | `LibRawIOError` |
| `exiftool/Sigma.x3f` | `LibRawIOError` |
| `exiftool/SigmaDP2.x3f` | `LibRawTooBigError` |
| `exiftool/Nikon.nef` | `LibRawDataError` |

`_raw_preview` (`vision/decode.py:229`) already caught `LibRawError`. The two
paths now agree. Files are ExifTool specimens truncated to metadata on
purpose — not a corpus defect.

## APP-2 — FFmpegError ends the job, depending on container

```text
status    FIXED
where     vision/decode.py frames_at
classes   FFmpegError(Exception)                        PyAV-Org/PyAV@040da79 av/error.pyi:9
          InvalidDataError(FFmpegError, ValueError)     av/error.pyi:27   covered
          EOFError(FFmpegError, builtins.EOFError)      av/error.pyi:59   NOT covered
blast     14 video-kind files: 8 decode, 6 fail
fix       every FFmpegError -> OSError at the decode boundary
```

`ASF.wmv` failed as one item. `Matroska.mkv` ended the job. Same fact about
the file, different outcome, decided by container format.

## APP-3 — builtin EOFError ends the job

```text
status    FIXED
where     db/runner.py ITEM_FAILURES  (+ EOFError)
raiser    pillow_heif 1.1.0 Image.load() -> db/oriented.py:137
message   Decoder plugin generated an error: Unexpected end of file
blast     one truncated HEIC ended a scan of 901 items
```

Raised outside `vision/decode.py`, so no decoder boundary exists to translate
it at. `EOFError` is raised where a decoder ran out of input: a fact about the
file, never a defect in the reader. Added to the tuple rather than patched at
a fourth site.

## APP-4 — `graph.text_of` stops at conditioning pass-through nodes

```text
status    FIXED
where     db/graph.py  text_of, read, _linked, _ONE_SIDE, _BOTH_SIDES, _ERASERS
input     92 prompt-bearing files, comfyanonymous/ComfyUI_examples@f9431bb000ce
tests     tests/test_the_schema_has_producers.py
          test_a_prompt_behind_a_controlnet_is_still_found
          test_a_node_conditioning_both_prompts_keeps_them_apart
          test_a_zeroed_conditioning_reports_no_words
          test_a_custom_sampler_takes_its_prompt_from_the_guider
```

Result over the 92:

```text
                       before   after
both populated             38      60
positive only              12      26
negative only              22       0
neither                    20       6
positive == negative         -       0
with a positive prompt     50      86
```

The 6 remaining carry no prompt to find: `sdxl_revision_zero_positive`
(zeroed by name), `stable_zero123_example` (no CLIPTextEncode at all), and
four edit-model workflows.

`metaparse.adapters.parse_file` returning `positive=''`/`params={}` is NOT the
defect — that is by design. `db/graph.py read()` is the extractor and does
populate a Recipe (sampler, seed, steps, cfg, negative). Checked and dismissed.

The defect is in `text_of`. Measured over the 92:

```text
both positive and negative   38
positive only                12
negative only                22   <-- positive lost
neither                      20
```

`text_of` follows only inputs NAMED `text, text_g, string, value, prompt,
populated_text` (`db/graph.py:163`). A conditioning pass-through node has none
of them, so the walk returns "" at the first one. `back()` (`db/graph.py:141`)
walks every input generically; `text_of` does not.

Worked example, `2_pass_pose_worship.png`:

```text
KSampler #3  negative -> ['7', 0]   CLIPTextEncode        resolves
KSampler #3  positive -> ['10', 0]  ControlNetApply       returns ""
                                    inputs: strength, conditioning,
                                            control_net, image
             ControlNetApply.conditioning -> ['6', 0]  CLIPTextEncode
                                                       (the lost prompt)
```

Sampler conditioning links landing on a node with no text input, all 92 files:

| count | node | class |
|---|---|---|
| 7 | `ConditioningCombine` | single |
| 7 | `unCLIPConditioning` | single |
| 6 | `StableCascade_StageB_Conditioning` | single |
| 5 | `ControlNetApply` | single |
| 1 | `FluxGuidance` | single |
| 1 | `ConditioningZeroOut` | single |
| 3+3 | `ControlNetApplyAdvanced` | slot-discriminated |
| 2+2 | `InstructPixToPixConditioning` | slot-discriminated |
| 2+2 | `ControlNetApplySD3` | slot-discriminated |
| 2+2 | `InpaintModelConditioning` | slot-discriminated |
| 1+1 | `StableZero123_Conditioning` | slot-discriminated |

Why it was not a one-line fix. The second class takes BOTH prompts and emits
slot 0 = positive, slot 1 = negative. `_link` discarded the slot index, so
"follow any conditioning input" routes a negative prompt into `positive` — a
wrong answer that looks like an answer.

The fix, in four parts:

1. `_linked` keeps `(node_id, slot)`; `_link` stays as the id-only caller.
2. `text_of` takes the slot it arrived on. A node holding both `positive` and
   `negative` follows the one the slot names; a node holding one chain follows
   any of `_ONE_SIDE`.
3. `_ERASERS`. `ConditioningZeroOut` DISCARDS its input — it is how a workflow
   says it has no negative prompt, by zeroing the POSITIVE conditioning. The
   first version of this fix walked through it and reported the positive
   prompt as the negative in `sd3_anime_example`,`sd3_controlnet_example` and
   `sd3.5_large_canny_controlnet_example`. Caught by the corpus, not by
   review.
4. `read` follows a custom sampler's `guider`. `SamplerCustomAdvanced` has no
   `positive` input at all; the chain is
   `guider -> BasicGuider -> FluxGuidance -> CLIPTextEncode`. A `BasicGuider`
   holds only the positive chain, so the negative is read only from a guider
   that has a `negative` input.

## APP-5 — a truncated video answered 500 instead of 404

```text
status    FIXED
where     vision/decode.py  open_still, frames_at
caught by tests/test_a_thumbnail_is_a_static_asset.py
          test_a_file_with_no_decodable_frame_is_a_404_not_a_500
```

Introduced by the first cut of APP-1/APP-2, not pre-existing. Both
translations raised `OSError`, which satisfied `ITEM_FAILURES` and broke the
thumbnail route: `sg_web/app.py:1216-1227` catches `ValueError` and answers
404, and anything else is an uncaught 500 with a traceback, once per grid
cell.

`ValueError` is the house convention for unrenderable media --
`vision/derive.py:257` already raises it for the same situation, and it is in
`ITEM_FAILURES` too. Both translations now raise `ValueError`.

## APP-6 — the timeline printed query strings at the reader

```text
status    FIXED
where     db/facets.py said(), sg_web/timeline_view.py _chips(),
          sg_web/templates/timeline.html
tests     tests/test_the_timeline_says_what_it_is_showing.py (3 of them)
```

The scope line rendered `one.spelled if one.spelled else key ~ "=" ~ value`.
Only a facet carries `spelled`, and it spells itself `tag:eq:beach` -- the
URL. So the page said `kind=image`, `favorite=True`, `tag:eq:beach` where it
meant `kind image`, `favorite yes`, `keyword beach`.

Two constraints had to hold at once:

- `TimelineScopePart.spelled` is asserted key-for-key by
  `tests/test_a_picture_has_a_place.py` and
  `tests/test_the_timeline_is_the_way_in.py` -- `None` for a scope,
  `place.id:eq:11` for a facet. Adding a field to the model broke their
  dict equality.
- The page must read in words.

Resolved by keeping the wire model exactly as it was and rendering the
sentences into the HTML context alone (`_chips`), from the same question the
model is built from. Neither existing test was edited.

`facets.said` rather than an import of `db.vocabulary` into the adapter:
`sglint SG402` does not let `sg_web/timeline_view.py` speak to the vocabulary,
and widening that allowlist is the move this repo has already had to revert
once. What a clause is called stays on the db side of the boundary.

## APP-7 — the app cannot date a photograph it only knows the year of

```text
status    FIXED
where     db/when.py folder_when + _YEARS, db/schema.sql (v43),
          db/context.py POLICY_VERSION 9, db/pages.py, db/events.py
found by  tests/needs.py, measuring the axis against the schema's vocabulary
proved by tests/scanned.py -- 10 folder shapes, 10 correct rungs
```

Two defects, one visible from the other.

**The year range was a digital camera's lifetime.** `_YEARS` was
`range(1990, 2101)`, and `_day_of` returns None outside it, so a folder or
filename dated `1964-08-12` produced nothing at all. Measured against the
corpus: six Commons photographs are from 1964, 1965, 1977, 1978, 1982 and
1989. Now `range(1826, 2101)` -- 1826 is the earliest surviving photograph,
and the lower bound only has to stop `1234` reading as a date.

**The folder ladder had no coarse rungs.** `folder_day` answered with a day
or with nothing:

```text
before                          after
1998/scan0042.jpg    -> mtime   -> 1998,    basis=folder, precision=year
2003-07/img12.jpg    -> mtime   -> 2003-07, basis=folder, precision=month
1970s/nan.jpg        -> mtime   -> 1970,    basis=folder, precision=decade
2013/02/x.jpg        -> mtime   -> 2013-02, basis=folder, precision=month
```

`folder_when` replaces it and reports its own precision; `folder_day` stays
as the narrow caller. Finest still wins, and the `2013/02/10` chain is read
at whatever depth it reaches.

Carried through every consumer, because a value nothing indexes is invisible:
`_SPAN` and `Verdict.refined_at` (`db/when.py`), the refinement window
(`db/context.py`), `_GRANULE` (`db/events.py`), `BINS` and `_FINE_ENOUGH`
(`db/pages.py` -- a `decade` bin was added, or a decade-dated scan would have
been absent from every zoom rather than shown at the resolution it is known
to). `_SEQUENCED` (`db/planning.py`) deliberately does NOT gain them: two
photographs both claimed to 1998 say nothing about which came first.

`POLICY_VERSION` 8 -> 9, because the interpretation changed meaning and
without the bump every already-interpreted row keeps its old verdict.

### The vocabulary is stated TWICE and I widened one

`time_precision` has two CHECK constraints, 70 lines apart:

```text
db/schema.sql:2034  derived_media_context.time_precision     -- the conclusion
db/schema.sql:2103  derived_media_occurrence.time_precision  -- the claim
```

Widening only the first made every context item fail with
`CHECK constraint failed: time_precision IN` -- a claim recorded at `year`
had nowhere to be written, while the table holding the conclusion would have
taken it. The job reported 1368 item failures and the corpus's coarse rungs
stayed invisible, which reads exactly like the corpus lacking the files.

The migration now rebuilds both. The DDL blocks are sliced from `schema.sql`
by terminator, and the occurrence table ends `) STRICT, WITHOUT ROWID;` --
searching for `) STRICT;` ran past it and captured two tables plus the
comments between them, which `conn.execute` refused as "only one statement
at a time". The failure was loud; a slice that had happened to end on a
statement boundary would not have been.

The constraint admits five values:

```sql
time_precision TEXT CHECK (time_precision IN
  ('day','hour','minute','second','subsecond'))
```

Every producer, enumerated:

| value | produced at |
|---|---|
| `day` | `db/when.py:380` (name stamp, no time part), `:417` (folder day) |
| `minute` | `db/when.py:284` (a filename carrying hour and minute) |
| `second` | `db/when.py:393`, `_wall` at `:192` |
| `subsecond` | `db/when.py:392`, `:459`, `:460`, `:593`, `:595` |
| `hour` | **nowhere** |

`hour` appears only where a value is CONSUMED -- `when.py:175`, `when.py:422`,
`context.py:510`, `events.py:62`, `pages.py:1341-1345`, `planning.py:96` --
never where one is assigned. `pages.py`'s `hour` bin admits claims of `hour`
precision that cannot exist.

**The real defect is coarser than the dead value.** `hour` being unproducible
is the visible end of it; the whole coarse half of the ladder is missing.

`folder_day` (`db/when.py:403-419`) parses a full `Y-M-D`, in one folder or
as a `2013/02/10` chain, and returns `None` for anything less. So:

```text
1998/scan0042.jpg          -> no claim; falls to mtime
2003-07/holiday/img12.jpg  -> no claim; falls to mtime
1970s/nan and grandad.jpg  -> no claim; falls to mtime
```

A scanned photograph in a `1998/` folder is dated by when somebody last
copied the file. The corpus proves this is not hypothetical: `commons` holds
photographs from **1964** across 37 distinct years, and exported and scanned
libraries are folder-dated by year or month constantly.

The vocabulary cannot express the answer either -- the CHECK admits
`day`, `hour`, `minute`, `second`, `subsecond` and has no `month`, `year` or
`decade` to put a coarse claim in.

An earlier version of this entry argued the CHECK was "a permissive domain,
not an obligation" and changed nothing. That is false in this repository:
`sglint/schema_rules.py` holds fifteen SG7xx rules making the schema an
executable contract, and `docs/BACKLOG.md` already calls a declared-but-
unexercised claim "an overclaim no corpus fixes -- only editing the claim
fixes the claim". It was a deferral wearing a rationale.

## APP-8 — the timeline answered 500 on any library holding a folder-dated scan

`GET /timeline` -> `500 KeyError: 'month'` at `sg_web/timeline_view.py:1031`.

`sg_web/timeline_view.py` kept its own `_SPAN = {"day": 86_400, "hour":
3_600, "minute": 60}` and indexed it with a `time_precision` read out of
`derived_media_context`. APP-7 taught the application to date a photograph
to its month, year or decade, and the first library containing one broke
the surface that draws time.

A copy does not fail when it goes stale. It fails on the one input the copy
never heard of, and only once somebody produces that input.

Fixed: `_SPAN = when.SPAN`. Not a wider copy -- the table itself. A lookup
keyed by a vocabulary another module owns has to BE that vocabulary.

Proven: the corpus report's `GET /timeline` line goes from
`"HTTP/1.1 500 Internal Server Error"` to `"HTTP/1.1 200 OK"`, and the
report prints `whole library ['12 years', '4 years', '7 years', '4 years']`.

## APP-9 — the granule facet excluded every coarse claim from every bin

`db/facets.py` restated the same width table as a SQL `CASE`, listing five
precisions and sending everything else to `2147483647`:

```sql
CASE mc.time_precision
  WHEN 'subsecond' THEN 0 WHEN 'second' THEN 1 WHEN 'minute' THEN 60
  WHEN 'hour' THEN 3600 WHEN 'day' THEN 86400 ELSE 2147483647 END
```

`context.granule` is asked as `lte:<bin width>`, so `decade`, `year` and
`month` compared as infinitely coarse: a folder-dated scan was excluded
from every bar link it belonged in. Silent -- the bar counted it and the
link did not return it.

Also wrong at the fine end: `subsecond` was 0 where `when.SPAN` says 0.001.

Fixed: the CASE is BUILT from `when.SPAN` at import. Found by the
adversarial-oracle magic-number audit (`MAGIC-FUCKUPS.json`), not by a test.

## APP-10 — four declared values no code could produce

`time_basis:first_seen`, `time_precision:hour`, `location_basis:sidecar`,
`location_basis:inferred`. Each was in a schema CHECK and a Python tuple,
and nothing in the application could write any of them.

`first_seen` is the clearest: the branch it named -- `judge_file` with no
name stamp, no dated folder, no mtime and no btime -- returns `None`, so it
writes no row at all. Two structures in the tree already disagreed with the
declaration: `derived_media_occurrence.basis` never listed it, and
`db/planning.py _BASES` held six where `TIME_BASES` held seven.

The precisions a `Verdict` can actually be constructed with, by extraction
rather than by reading: a stamped name gives day, second or subsecond; a
folder gives day, month, year or decade; a generator date gives second, day
or minute; the filesystem fallback gives subsecond. Never hour.

Removed in v44 (`db/migrate.py @step(43)`), which counts offending rows
before rebuilding rather than assuming there are none.

## APP-11 — two claimed suffixes have no decoder, and the gate never asked

`vision/decode.py RAW_SUFFIXES` was immich's list under a comment calling
it "LibRaw's coverage spelled as suffixes". immich's table maps a suffix to
the MIME types IT SERVES. Two entries were not LibRaw formats at all:

```text
.cin   immich: image/x-phantom-cin     Phantom Cine, a video format
.ari   immich: image/x-arriflex-ari    a cinema camera's format
```

Searched LibRaw@HEAD `src/`, `internal/`, `libraw/`: `cineon` 0 hits;
`phantom` 1, and it is `"DJI Phantom4 Pro/Pro+"` at `cameralist.cpp:310`;
`\bARRI\b|ARRIFLEX|\bAlexa\b` 0. Controlled in the same trees with the same
flags -- Sigma 102 hits in 7 files, Hasselblad 16 files including its own
decoder and model module, Phase One 15. A bare `arri` search returns 3 and
every one is the word `barrier`.

The corpus held a real CC0 Arri Alexa Mini frame and LibRaw answered
`Unsupported file format or not RAW file`. That read as a corpus problem
for as long as the claim stood. It was the claim.

And the gate never asked: `tests/test_the_corpus_spans_the_shape.py` tested
dialect, kind, time_basis, time_precision, location_basis and origin, and
never `suffix:` -- the largest axis. Four claimed suffixes had no corpus
file and failed nothing.

`.cap` (Phase One) and `.k25` (Kodak DC25 -- `cameralist.cpp:513`,
`identify.cpp:3256`) are real LibRaw formats. Neither is in
raw.pixls.us's CC0 archive (1870 samples; `.iiq` 35 as control), exiftool's
`t/images` (194 files; `PhaseOne.iiq` as control), or rawsamples.ch's Kodak
listing (`.KDC`/`.DCR` as control). Those searches are recorded as evidence
in `tests/needs.py BLOCKED`, the two needs are BLOCKED_EXTERNALLY, and the
gate is green in this -- the repository's intended -- state. It cannot rot
into an excuse: `test_an_excuse_cannot_outlive_its_gap` fails the moment a
corpus file reads either suffix while the row still stands, and
`tests/rawsamples.py` re-asks the archive on every fetch. A gate red by
design was tried first; it lasted one push before it taught `LEFTHOOK=0`.

## Not findings

### pyvips TIFF warnings

`Tag BitsPerSample entry count is 3, whereas it should be SamplesPerPixel=1`,
`Photometric tag is missing`, `error in tile 0 x 0`, logged at WARNING on the
truncated ExifTool TIFFs. Library reporting malformed input it then handles.
Files are malformed on purpose.

### JXL "Generic Error. Please build libjxl from source"

The message names a build problem; the build is fine. Checked by writing a
valid JXL with the same plugin and decoding it:

```text
valid.jxl   80 bytes   decodes -> (64, 48) RGB
JXL.jxl     22 bytes   RuntimeError
JXL2.jxl   395 bytes   RuntimeError
```

22 bytes is not a picture. Both are ExifTool specimens truncated to their
metadata, `tests/sourced.py INTENT` says so of `JXL.jxl` already, and failing
on them is correct. The message is `pillow-jxl-plugin`'s wording for any
decode failure, not a statement about this installation.

### `just test` collects nothing

`tests/conftest.py` marks the whole collection `slow`, so `-m "not slow"`
selects zero and `|| [ $? -eq 5 ]` turns "collected nothing" into success.
Deliberate and explained at `justfile:11-20`; `just test-slow` is the suite.

README described that lane as "the fast tests, one module per worker (~20s)",
which was false. Corrected.
