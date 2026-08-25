# Corpus sources

Contract: `docs/CORPUS_CONTRACT.md`. Root: `../sg-corpus`.
Rejecting a source does not close the need it was a candidate for.

Per-file provenance: `tests/sourced.lock.json`, `tests/commons.lock.json`,
`tests/gathered.lock.json`. Each row carries a checksum, byte size, origin and
the reason the file is present.

## Accepted

| part | source | pin | license | files | bytes | kind |
|---|---|---|---|---|---|---|
| `exiftool` | `exiftool/exiftool` `t/images` | `2200871d9cef` (tag 13.59) | GPL-3 | 194 | 1.1 MB | real |
| `raw-dng` | `Temporarium/HDR_Photos_VAE_Training_DNG` | unpinned | cc0-1.0 | 35 | 427 MB | real |
| `commons` | Wikimedia Commons API | per-file `sha1` | 15 free licenses | 243 | 459 MB | real |
| `raw-canon` | local Canon EOS 5D Mark III shoot | n/a | author's own | 326 | 3.78 GB | real |
| `comfyui` | `comfyanonymous/ComfyUI_examples` | `f9431bb000ce` | permissive (repo LICENSE) | 100 | 136 MB | generated |
| `swarm` | local SwarmUI output | n/a | author's own | 115 | 165 MB | generated |
| `swarm-i2i` | local SwarmUI i2i output | n/a | author's own | 55 | 131 MB | generated |
| `generated-bare` | local ChatGPT image output | n/a | author's own | 6 | 16 MB | generated |

### What each part is for

```text
exiftool        194 writers, one file per format/maker. 134/194 do not decode;
                median dimension of those that do is 8x8. Read targets.
raw-dng         Google Pixel 3a DNG. 6/6 sampled carry camera, capture time, ISO.
commons         43 maker strings, 37 years (1964-2026).
raw-canon       CR2 + out-of-camera JPEG pairs, one 2013 shoot. Same capture,
                different bytes: a duplicate a checksum cannot see.
comfyui         92/100 carry BOTH `prompt` and `workflow` chunks. 8 carry none.
swarm           `parameters` chunk (SwarmUI). Generated video beside stills.
swarm-i2i       many outputs sharing one source image.
generated-bare  0 chunks, 0 EXIF. Measured, not assumed.
```

### commons — how diversity was obtained

The API reports a file's EXIF before its bytes are fetched
(`iiprop=metadata|extmetadata|sha1|size|mime|url`), so selection targets
`(maker, model, year)` and keeps only unseen triples. Downloaded bytes are
checked against the SHA-1 the API reported.

Maker strings are kept verbatim, including the drift the corpus exists to
carry:

```text
Asahi Optical Co., Ltd.   vs  Asahi Optical Co.,Ltd        one space
NIKON / Nikon             motorola / Motorola              case
samsung / Samsung Electronics / Samsung Techwin / SAMSUNG TECHWIN CO., LTD.
OLYMPUS OPTICAL CO.,LTD -> OLYMPUS IMAGING CORP. -> OLYMPUS CORPORATION
LEICA / LEICA CAMERA AG / Leica Camera AG
```

Licenses present: CC BY-SA 4.0 (48), CC BY-SA 3.0 (47), Public domain (33),
CC BY 2.0 (21), CC BY-SA 2.0 (18), CC0 (9), CC BY 3.0 (7), CC BY-SA 2.5 (3),
CC BY 4.0 (2), CC BY 1.0, GFDL, GFDL 1.2, Attribution, Copyrighted free use,
No restrictions (1 each). Every file has an `artist` field for attribution.

Kodak and Casio were recorded as having no category under any spelling. Two
causes, both fixed:

- `Photos taken with X cameras` is a fourth spelling `FORMS` did not have.
- A category can hold files without having a description page, and then
  `prop=categoryinfo` reports it `missing`. `Category:Taken with Casio` is
  one. `category_for` now falls back to `list=allcategories` with the prefix,
  which asks what categories exist rather than what pages do.

Result: 194 files/39 makers -> 243 files/43 makers, and `trouble` is empty.

## Excluded on purpose

`tests/gathered.py SOURCES` is an allowlist. The same local tree holds
face-recognition and KYC photograph sets of real people
(`Black_People_Face_Recognition`, `caucasian-people-kyc-photo-dataset`,
`people`, `people_detection`, `real_people_3000`, `IndoorSceneRecognition`).
Not licensed for redistribution, never copied into the corpus. A set added to
that tree later is excluded by default rather than swept in.

## Rejected

### `Cheliosoops/EXIF`

```text
revision  3a4f287ef4e1d5e62e159ad769ad94d57f7090bc   modified 2025-01-15
size      16 files, 35.9 GB advertised, no declared license
probed    2026-08-25 by downloading 1.9 GB
verdict   wrong content after inspection
```

| file | bytes | sha256 |
|---|---|---|
| `CANON.zip` | 1126356945 | `bc58e0a8e7cac62d365e596d7a02374f1e6bd25dd382cd63d1750b4ffec552b7` |
| `Nikon.zip` | 778263551 | `a4c75ebb6b520bc253babb4b7959c6cc382763086cb6f1fd11adec63b82224bb` |

Nested zips: CANON.zip = 277 inner, Nikon.zip = 201 inner under
`Nikon_train/` and `Nikon_test/`, named `AutoMode_Scene###`.

Sampled every ~23rd inner archive across both, n=23:

```text
Exif APP1 present      0/23
file header            ffd8ffe0 (JFIF APP0) in 23/23
dimensions             768x512 Canon 12/12, 770x512 Nikon 11/11
PIL getexif() entries  0
```

ML scene split, metadata stripped and images resized. `db.capture.read()`
returning all-`None` on these is correct; no application change warranted.
Remaining 34 GB not downloaded. Local bytes deleted.

### `images9/flickr-cc-by`

```text
size      14.0 GB single tar, cc0-1.0 (metadata only)
probed    2026-08-25 by HTTP range request, first 1 MB
verdict   contains no images
```

Tar holds per-timerange zips of Flickr API JSON. One shard
(`1202849709_1202849787.json`, 126824 bytes) held 262 photo records.

Fields present across all 262: `url_m`, `url_s` (262 each), `url_h` (69).
Absent: `url_o`, `originalsecret`, `originalformat` — no original-size URL.

Fetched `url_h` for photo 2260666569 to test whether Flickr's resized variants
keep EXIF:

```text
bytes 277312   header ffd8ffed (APP13 Photoshop IRB)
Exif APP1 present   False
PIL getexif()       0 entries
dimensions          1600x1060
```

Flickr strips EXIF from re-encoded sizes. Usable as a real-photograph index,
not as a source of camera metadata. Not downloaded.
