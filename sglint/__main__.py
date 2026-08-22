"""`python -m sglint`: the repository's structural rules as a linter.

Prints one `path:line:col: CODE message` per finding and exits 1 when
there are any; prints nothing and exits 0 on a clean tree. `--repo`
adds the repository-hygiene rules (SG8xx: the git index, line endings,
the requirements file, the test command) -- they ask git, so they are
their own lane (`just repo-check`). `--explain` lists the families.
"""

from __future__ import annotations

import sys

from . import rules


def main(argv: list[str]) -> int:
    if "--explain" in argv:
        print(rules.__doc__)
        return 0
    found = rules.run()
    if "--repo" in argv:
        from . import repo_rules

        found = sorted([*found, *repo_rules.run()], key=lambda f: (str(f.path), f.line, f.col, f.code))
    for one in found:
        print(one.spelled())
    if found:
        print(f"{len(found)} finding(s)", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
