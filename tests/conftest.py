"""Shared pytest behaviour for the greenfield suite.

No hooks. Tests own their databases (in-memory or under tmp_path), the
application under test is `db` + `sg_web` + `vision` + `metaparse`, and
nothing here points environment variables at anything -- a suite that
needs its environment arranged before import is a suite whose subject
reads configuration at import time, and that defect died with the
application that had it. No test starts a program (sglint SG006): the
checks that need one (git, a checkout) are `python -m sglint --repo`.
"""
