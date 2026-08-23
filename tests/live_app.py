"""The application a browser test's server process serves.

Litestar's own subprocess runner (`litestar.testing.client.subprocess_client.run_app`)
starts `litestar --app tests.live_app:create_app run` on a free port; the
home directory comes from the one variable the harness sets before it
launches (`SG_TEST_HOME`). The product reads no environment; this module
is the harness's, not the product's.
"""

from __future__ import annotations

import os

from litestar import Litestar

from sg_web.app import build_app


def create_app() -> Litestar:
    return build_app(os.environ["SG_TEST_HOME"])
