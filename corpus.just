# A sample library on disk: real media, and a generated shape around it.
#
# TWO HALVES, because neither does the other's job.
#
# FETCHED is real media from pinned public sources -- one file per format
# and per camera maker, written by the bodies and tools that really write
# them. It is what reaches the readers: a 2001 Nikon with no subsecond, a
# Pixel 8 with a real OffsetTimeOriginal, a Pentax at Orientation=8, a
# Canon CR3 that PyAV opens and cannot describe. A generator emits what
# its own library emits, so it can only ever test one writer; these are
# many. Measured: they take the readers from 24.2% to 31.3%.
#
# GENERATED is the shape a real library has in TIME -- decades, unevenly,
# with holes. The fetched files carry whatever dates they carry and no
# arrangement at all, so they cannot exercise a timeline. This half can,
# and it is honest about being painted rectangles.
#
# The fetched half is NOT vendored: ExifTool is GPL-3, its images are
# collected from mixed sources, and several carry real coordinates. They
# are downloaded here, outside the repository, and never committed.

set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

#: Where a corpus goes unless somebody says otherwise. Beside the repo
#: rather than inside it: it is hundreds of files that must never be
#: mistaken for the tree, and `.gitignore` is a weaker promise than a
#: different parent. `SG_CORPUS` overrides it for the tests.
default_root := justfile_directory() / '..' / 'sg-corpus'

# What each part holds, so a person can pick before waiting.
[doc('What each scale and source holds, without fetching or writing it')]
scales:
    @echo 'REAL (measured 2026-08-25; see docs/CORPUS_SOURCES.md)'
    @echo '  exiftool        194 files   1.1 MB   GPL-3      194 writers; 134 do not decode'
    @echo '  commons         194 files   224 MB   15 free    39 makers, 112 models, 1964-2026'
    @echo '  raw-canon       326 files  3.78 GB   own        5D Mark III, CR2+JPEG pairs'
    @echo '  raw-dng          35 files   427 MB   cc0-1.0    Google Pixel DNG  (--raw)'
    @echo
    @echo 'GENERATED (real generator metadata, not painted rectangles)'
    @echo '  comfyui         100 files   136 MB   permissive 92 carry prompt+workflow'
    @echo '  swarm           115 files   165 MB   own        parameters chunk, plus video'
    @echo '  swarm-i2i        55 files   131 MB   own        many outputs, one source'
    @echo '  generated-bare    6 files    16 MB   own        no chunk, no EXIF'
    @echo
    @echo 'GENERATED TIME SHAPE (painted rectangles, honestly)'
    @echo '  small       ~85 files            a fixture; seconds'
    @echo '  medium     ~325 files            enough to look at; under a minute'
    @echo '  wide       ~900 files            enough to feel; a few minutes (video encodes)'

# The real media, and nothing else. Idempotent: present means no network.
[doc('Fetch the real media only (194 files, 1.1 MB) into DIR')]
[script]
fetch DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -c "from tests import sourced; print(sourced.fetch())"

# Both halves. SCALE is small|medium|wide; pass --raw for the 407 MB set.
[doc('Fetch the real media and write the generated time shape into DIR')]
[script]
write SCALE='medium' DIR=default_root RAW='':
    cd "$(git rev-parse --show-toplevel)"
    scale="$1"
    dir="$2"
    raw="${3:-}"
    mkdir -p "$dir"

    echo "== fetching real media"
    SG_CORPUS="$dir" {{ python }} -c "from tests import sourced; print('  ', sourced.fetch())"

    if [ "$raw" = "--raw" ]; then
      echo "== fetching the RAW set (407 MB)"
      {{ python }} - "$dir" <<'PY'
    import pathlib, sys
    from huggingface_hub import snapshot_download
    into = pathlib.Path(sys.argv[1]) / "raw-dng"
    into.mkdir(parents=True, exist_ok=True)
    snapshot_download("Temporarium/HDR_Photos_VAE_Training_DNG", repo_type="dataset",
                      local_dir=str(into), max_workers=8)
    print("  ", into)
    PY
    fi

    echo "== writing the generated time shape"
    {{ python }} -m tests.corpus "$dir/generated" --scale "$scale" --force

    echo
    echo "add it:  POST /roots {\"path\": \"$(cd "$dir" && pwd -W 2>/dev/null || pwd)\"}"
    echo "or open  /operations  and add the root there"

# Fetched, written, scanned, and read back -- the only proof that a
# corpus is a LIBRARY rather than a directory. Reports what the
# application made of it: kinds, recipes, cameras, duplicate groups, and
# how much of the timeline is empty enough to collapse.
[doc('Build a small corpus, scan it, and report what the library made of it')]
[script]
prove:
    cd "$(git rev-parse --show-toplevel)"
    {{ python }} -m tests.corpus_report

# Real photographs, chosen for metadata diversity. The API reports a
# file's EXIF before its bytes are fetched, so this keeps one file per
# (maker, model, year) rather than downloading and sorting out after.
[doc('Fetch real photographs from Wikimedia Commons into DIR')]
[script]
commons DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.commons

# Media this machine and the reference mirrors already hold: ComfyUI's
# own prompt+workflow chunks, SwarmUI output, generated stills carrying
# nothing at all, and a Canon shoot's CR2/JPEG pairs.
[doc('Copy local and reference media into DIR')]
[script]
gather DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.gathered

# One file per registered generator. Six of the eight have no obtainable
# real specimen -- 925 candidate images across four upstream mirrors carry
# no generator metadata at all -- so these are written from this repo's own
# writer ports and labelled written. See tests/dialects.py.
[doc('Write one file per generator adapter into DIR')]
[script]
dialects DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.dialects

# One file per container the app claims and no real source writes.
[doc('Encode one file per otherwise-unreachable container into DIR')]
[script]
encoded DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.encoded

# Photographs whose only date is the folder they sit in -- the coarse
# rungs, which every camera-and-generator corpus misses entirely.
[doc('Write folder-dated undated photographs into DIR')]
[script]
scanned DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.scanned

# Real RAW, one per suffix, CC0, from the darktable/RawTherapee archive.
[doc('Fetch one real RAW per uncovered suffix into DIR')]
[script]
rawsamples DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.rawsamples

# What the application declares it supports, against what the corpus reads.
# The frozen list the contract's coverage rule needs: a need may reach
# SATISFIED or BLOCKED_EXTERNALLY, and never DEFERRED.
[doc('Measure the corpus at DIR against every suffix and dialect the app claims')]
[script]
needs DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m tests.needs

# The measurement above REPORTS. This one FAILS.
#
# A number written into a lockfile can be satisfied by writing a bigger
# number. `tests/test_the_corpus_spans_the_shape.py` re-measures and asserts
# every declared value is reached by a real file or BLOCKED_EXTERNALLY in
# `tests/needs.py BLOCKED` with evidence -- and asserts the reverse too, so
# an excuse cannot outlive the gap it was written for.
[doc('Fail if the corpus does not span the shape the application declares')]
[script]
spans DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    SG_CORPUS="$1" {{ python }} -m pytest tests/test_the_corpus_spans_the_shape.py -m slow

# Every part, in order. What `docs/CORPUS_CONTRACT.md` calls the corpus.
[doc('Assemble the whole corpus into DIR: fetched, commons, gathered, generated')]
[script]
all SCALE='medium' DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    scale="$1"
    dir="$2"
    mkdir -p "$dir"
    echo "== real media, pinned"
    SG_CORPUS="$dir" {{ python }} -c "from tests import sourced; print('  ', sourced.fetch())"
    echo "== real photographs, Wikimedia Commons"
    SG_CORPUS="$dir" {{ python }} -m tests.commons
    echo "== local and reference media"
    SG_CORPUS="$dir" {{ python }} -m tests.gathered
    echo "== one file per generator"
    SG_CORPUS="$dir" {{ python }} -m tests.dialects
    echo "== one real RAW per uncovered suffix"
    SG_CORPUS="$dir" {{ python }} -m tests.rawsamples
    echo "== one encode per otherwise-unreachable container"
    SG_CORPUS="$dir" {{ python }} -m tests.encoded
    echo "== folder-dated photographs, the coarse rungs"
    SG_CORPUS="$dir" {{ python }} -m tests.scanned
    echo "== the generated time shape"
    {{ python }} -m tests.corpus "$dir/generated" --scale "$scale" --force
    echo "== what it covers"
    SG_CORPUS="$dir" {{ python }} -m tests.needs

# The contract's runtime proof: the REAL corpus through the real
# ingestion path, unmodified. Failures are findings about the
# application (docs/CORPUS_FINDINGS.md), never a reason to edit a file.
[doc('Scan an EXISTING corpus at DIR with the application and report')]
[script]
against DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    {{ python }} -m tests.corpus_report "$1"
