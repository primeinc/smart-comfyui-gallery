"""Generation-metadata parsing for images from any known SD/diffusion tool.

Marker-based detection first (tool-unique chunk names), popularity-ordered
heuristics second, opt-in stealth-pnginfo (LSB) last. See adapters.py for the
registry and the upstream format references.

    from metaparse import parse_file, render_report
    parsed = parse_file(path, allow_stealth=True)
    if parsed:
        print(parsed.tool, parsed.positive)
        print(render_report(parsed))
"""

from .adapters import parse_file, parse_raw, parse_stealth_text, parse_infotext
from .containers import RawMetadata, load_raw
from .model import ParsedMetadata
from .render import render_report

__all__ = [
    "parse_file", "parse_raw", "parse_stealth_text", "parse_infotext",
    "RawMetadata", "load_raw", "ParsedMetadata", "render_report",
]
