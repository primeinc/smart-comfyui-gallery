from __future__ import annotations

import warnings
from typing import Literal

Action = Literal["default", "error", "ignore", "always", "module", "once"]

VENDOR_TOLERATED: tuple[tuple[Action, type[Warning], str, str], ...] = (
    ("ignore", FutureWarning, r"insightface\.utils\.face_align", r".*`estimate` is deprecated.*"),
    ("ignore", UserWarning, r"torchvision\.models\._utils", r".*'pretrained' is deprecated.*"),
    ("ignore", UserWarning, r"torchvision\.models\._utils", r".*Arguments other than a weight enum.*"),
    # transformers' sam3_video applies @torch.jit.script at import
    # (models/sam3_video/modeling_sam3_video.py:1845), which torch 2.9
    # deprecates.
    #
    # torch warns with no stacklevel, so the warning is attributed to
    # torch.jit._script itself -- the only module a filter can pin; the
    # message pins the one deprecation tolerated.
    ("ignore", DeprecationWarning, r"torch\.jit\._script", r".*`torch\.jit\.script` is deprecated.*"),
)


def arm() -> None:
    warnings.resetwarnings()
    warnings.filterwarnings("error")
    for action, category, module, message in VENDOR_TOLERATED:
        warnings.filterwarnings(action, category=category, module=module, message=message)
