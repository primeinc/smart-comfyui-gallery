from __future__ import annotations

import warnings
from typing import Literal

Action = Literal["default", "error", "ignore", "always", "module", "once"]

VENDOR_TOLERATED: tuple[tuple[Action, type[Warning], str, str], ...] = (
    ("ignore", FutureWarning, r"insightface\.utils\.face_align", r".*`estimate` is deprecated.*"),
    ("ignore", UserWarning, r"torchvision\.models\._utils", r".*'pretrained' is deprecated.*"),
    ("ignore", UserWarning, r"torchvision\.models\._utils", r".*Arguments other than a weight enum.*"),
)


def arm() -> None:
    warnings.resetwarnings()
    warnings.filterwarnings("error")
    for action, category, module, message in VENDOR_TOLERATED:
        warnings.filterwarnings(action, category=category, module=module, message=message)
