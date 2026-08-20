"""What EXIF's enumerated values mean, so a facet can be read by a person.

Most of what a camera records about a frame is a small integer standing for a
phrase. Stored raw, `Flash = 89` is unsearchable and unreadable: nobody types
89, and nobody recognises it. Stored as the phrase alone it stops being
comparable. `file_param` has both columns, so both go in -- the label in
`value_text`, the code in `value_num`.

Every table below is transcribed from exiftool, which is the working
reference for what these values mean in practice rather than in the
specification -- it carries the non-standard values real cameras emit
(Samsung's SceneCaptureType 4, Apple's CustomRendered 2-8) that a
spec-derived table would reject as invalid.

Source: refs/exiftool/exiftool/lib/Image/ExifTool/Exif.pm, cited per table.
Transcribed 2026-08-19; these are stable published tables, but a value seen
in the wild and missing here should be added from that file rather than
guessed at.
"""

from __future__ import annotations

from PIL import ExifTags

# Exif.pm:175-209. Flash is a bitfield printed in hex, which is why the codes
# look arbitrary in decimal: 0x59 = 89 = auto, fired, red-eye reduction.
_FLASH = {
    0x00: "No Flash",
    0x01: "Fired",
    0x05: "Fired, Return not detected",
    0x07: "Fired, Return detected",
    0x08: "On, Did not fire",
    0x09: "On, Fired",
    0x0D: "On, Return not detected",
    0x0F: "On, Return detected",
    0x10: "Off, Did not fire",
    0x14: "Off, Did not fire, Return not detected",
    0x18: "Auto, Did not fire",
    0x19: "Auto, Fired",
    0x1D: "Auto, Fired, Return not detected",
    0x1F: "Auto, Fired, Return detected",
    0x20: "No flash function",
    0x30: "Off, No flash function",
    0x41: "Fired, Red-eye reduction",
    0x45: "Fired, Red-eye reduction, Return not detected",
    0x47: "Fired, Red-eye reduction, Return detected",
    0x49: "On, Red-eye reduction",
    0x4D: "On, Red-eye reduction, Return not detected",
    0x4F: "On, Red-eye reduction, Return detected",
    0x50: "Off, Red-eye reduction",
    0x58: "Auto, Did not fire, Red-eye reduction",
    0x59: "Auto, Fired, Red-eye reduction",
    0x5D: "Auto, Fired, Red-eye reduction, Return not detected",
    0x5F: "Auto, Fired, Red-eye reduction, Return detected",
}

# Exif.pm:139-172
_LIGHT_SOURCE = {
    0: "Unknown",
    1: "Daylight",
    2: "Fluorescent",
    3: "Tungsten (Incandescent)",
    4: "Flash",
    9: "Fine Weather",
    10: "Cloudy",
    11: "Shade",
    12: "Daylight Fluorescent",
    13: "Day White Fluorescent",
    14: "Cool White Fluorescent",
    15: "White Fluorescent",
    16: "Warm White Fluorescent",
    17: "Standard Light A",
    18: "Standard Light B",
    19: "Standard Light C",
    20: "D55",
    21: "D65",
    22: "D75",
    23: "D50",
    24: "ISO Studio Tungsten",
    25: "Daylight",
    26: "Day White",
    27: "Cool White",
    28: "White",
    29: "Warm White",
    30: "Daylight LED",
    31: "Day White LED",
    32: "Cool White LED",
    33: "White LED",
    34: "Warm White LED",
    255: "Other",
}

# Exif.pm:291-300
_ORIENTATION = {
    1: "Horizontal (normal)",
    2: "Mirror horizontal",
    3: "Rotate 180",
    4: "Mirror vertical",
    5: "Mirror horizontal and rotate 270 CW",
    6: "Rotate 90 CW",
    7: "Mirror horizontal and rotate 90 CW",
    8: "Rotate 270 CW",
}

#: tag -> {code: label}. Cited individually; see the module docstring.
LABELS: dict[int, dict[int, str]] = {
    ExifTags.Base.Flash: _FLASH,  # Exif.pm:2413-2420
    ExifTags.Base.LightSource: _LIGHT_SOURCE,  # Exif.pm:2402-2407
    ExifTags.Base.Orientation: _ORIENTATION,  # Exif.pm:683-689
    ExifTags.Base.MeteringMode: {  # Exif.pm:2392-2401
        0: "Unknown",
        1: "Average",
        2: "Center-weighted average",
        3: "Spot",
        4: "Multi-spot",
        5: "Multi-segment",
        6: "Partial",
        255: "Other",
    },
    ExifTags.Base.ExposureProgram: {  # Exif.pm:2108-2119
        0: "Not Defined",
        1: "Manual",
        2: "Program AE",
        3: "Aperture-priority AE",
        4: "Shutter speed priority AE",
        5: "Creative (Slow speed)",
        6: "Action (High speed)",
        7: "Portrait",
        8: "Landscape",
        9: "Bulb",
    },
    ExifTags.Base.WhiteBalance: {0: "Auto", 1: "Manual"},  # Exif.pm:2874
    ExifTags.Base.ExposureMode: {  # Exif.pm:2863
        0: "Auto",
        1: "Manual",
        2: "Auto bracket",
    },
    ExifTags.Base.SceneCaptureType: {  # Exif.pm:2900-2906
        0: "Standard",
        1: "Landscape",
        2: "Portrait",
        3: "Night",
        4: "Other",  # non-standard, some Samsung models
    },
    ExifTags.Base.GainControl: {  # Exif.pm:2913-2919
        0: "None",
        1: "Low gain up",
        2: "High gain up",
        3: "Low gain down",
        4: "High gain down",
    },
    ExifTags.Base.Contrast: {0: "Normal", 1: "Low", 2: "High"},  # Exif.pm:2925
    ExifTags.Base.Saturation: {0: "Normal", 1: "Low", 2: "High"},  # Exif.pm:2936
    ExifTags.Base.Sharpness: {0: "Normal", 1: "Soft", 2: "Hard"},  # Exif.pm:2947
    ExifTags.Base.SubjectDistanceRange: {  # Exif.pm:2963-2968
        0: "Unknown",
        1: "Macro",
        2: "Close",
        3: "Distant",
    },
    ExifTags.Base.SensingMethod: {  # Exif.pm:2477-2486
        1: "Monochrome area",
        2: "One-chip color area",
        3: "Two-chip color area",
        4: "Three-chip color area",
        5: "Color sequential area",
        6: "Monochrome linear",
        7: "Trilinear",
        8: "Color sequential linear",
    },
    ExifTags.Base.CustomRendered: {  # Exif.pm:2844-2854
        0: "Normal",
        1: "Custom",
        2: "HDR (no original saved)",
        3: "HDR (original saved)",
        4: "Original (for HDR)",
        6: "Panorama",
        7: "Portrait HDR",
        8: "Portrait",
    },
    ExifTags.Base.ColorSpace: {  # Exif.pm:2685-2693
        1: "sRGB",
        2: "Adobe RGB",
        0xFFFD: "Wide Gamut RGB",
        0xFFFE: "ICC Profile",
        0xFFFF: "Uncalibrated",
    },
    ExifTags.Base.ResolutionUnit: {1: "None", 2: "inches", 3: "cm"},  # Exif.pm:874
    ExifTags.Base.FocalPlaneResolutionUnit: {  # Exif.pm:2438-2447
        1: "None",
        2: "inches",
        3: "cm",
        4: "mm",
        5: "um",
    },
}


def label_for(tag, value):
    """The phrase this code stands for, or None if it is not enumerated.

    An unknown code returns None rather than a guess: a value missing from
    these tables is a table to extend, and inventing a label for it would
    hide that.
    """
    table = LABELS.get(tag)
    if table is None:
        return None
    try:
        return table.get(int(value))
    except (TypeError, ValueError):
        return None
