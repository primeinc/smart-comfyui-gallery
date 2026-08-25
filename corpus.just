# A library shaped like a real one, on disk.
#
# Twenty `write_library` functions in the suite each hand-roll their own
# media and not one spans more than a fortnight, so every surface that
# has to survive a LIBRARY was only exercised against eight files in one
# afternoon. This writes the other thing: two decades, uneven, mixed
# modalities, several generator dialects, and the disagreements a real
# folder is full of.
#
# Deterministic in `seed`, so the same arguments write the same bytes.

set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

#: Where a corpus goes unless somebody says otherwise. Beside the repo
#: rather than inside it: it is hundreds of files that must never be
#: mistaken for the tree, and `.gitignore` is a weaker promise than a
#: different parent.
default_root := justfile_directory() / '..' / 'sg-corpus'

# What the scales hold, so a person can pick before waiting.
[doc('What each scale writes, without writing it')]
scales:
    @echo 'small   ~70 files    a fixture; seconds'
    @echo 'medium  ~310 files   enough to look at; under a minute'
    @echo 'wide    ~900 files   enough to feel; a few minutes (video encodes)'

# Write one. SCALE is small|medium|wide.
[doc('Write a decades-wide sample library to DIR (default ../sg-corpus)')]
[script]
write SCALE='medium' DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    scale="$1"
    dir="$2"
    mkdir -p "$dir"
    {{ python }} -m tests.corpus "$dir" --scale "$scale" --force
    echo
    echo "add it:  POST /roots {\"path\": \"$(cd "$dir" && pwd -W 2>/dev/null || pwd)\"}"
    echo "or open  /operations  and add the root there"

# The same, plus real media mixed in and re-dated across the range.
#
# The synthetic files are small painted rectangles: they exercise every
# seam and lie about every cost. One real 24-megapixel raw tells the
# thumbnail, decode and embed paths something no number of generated
# ones will.
[doc('Write a corpus with real media from SOURCE mixed in and re-dated')]
[script]
mixed SOURCE SCALE='medium' DIR=default_root:
    cd "$(git rev-parse --show-toplevel)"
    source="$1"
    scale="$2"
    dir="$3"
    test -d "$source" || { echo "no such directory: $source" >&2; exit 1; }
    mkdir -p "$dir"
    {{ python }} -m tests.corpus "$dir" --scale "$scale" --real "$source" --force

# Written, scanned, and read back -- the only proof that a corpus is a
# LIBRARY rather than a directory. Reports what the application made of
# it: kinds, recipes, cameras, duplicate groups, and how much of the
# timeline is empty enough to collapse.
[doc('Write a small corpus, scan it, and report what the library made of it')]
[script]
prove:
    cd "$(git rev-parse --show-toplevel)"
    {{ python }} -m tests.corpus_report
