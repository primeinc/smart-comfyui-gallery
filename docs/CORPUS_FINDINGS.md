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
