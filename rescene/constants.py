"""Shared constants: defect label normalization and defect families.

Real-IAD stores defect classes as short codes inside JSON metadata and file
names. The mapping below covers the codes documented in the Real-IAD paper /
release; unknown codes pass through unchanged (lower-cased) so the pipeline
never crashes on a new dataset variant.
"""

# Real-IAD defect code -> human readable label (best effort; unknown codes pass through)
REAL_IAD_DEFECT_CODES = {
    "OK": "good",
    "NG": "defect",
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing parts",
    "YW": "foreign objects",
    "ZW": "contamination",
}

# keyword -> defect family. Checked in order; first match wins.
DEFECT_FAMILY_KEYWORDS = [
    (("scratch", "scratches", "abrasion", "cut", "line"), "line_damage"),
    (("crack", "fracture", "break", "broken", "damage", "chip"), "breakage"),
    (("missing", "absent", "loss", "lack", "incomplete", "hole", "pit"), "missing_part"),
    (("contamination", "dirt", "stain", "smudge", "zw", "soil", "spot"), "contamination"),
    (("deformation", "bent", "bend", "warp", "twist", "dent"), "deformation"),
    (("foreign", "particle", "fiber", "hair", "extra"), "foreign_object"),
    (("bubble", "blister", "porosity"), "surface_bubble"),
    (("color", "discolor", "fade", "burn", "oxid", "corrosion", "rust"), "discoloration"),
    (("glue", "overflow", "residue", "excess"), "material_excess"),
]

DEFAULT_DEFECT_FAMILY = "generic_defect"

# a defect family -> short natural language phrase used by the instruction generator
DEFECT_FAMILY_PHRASES = {
    "line_damage": "scratch-like",
    "breakage": "damage-like",
    "missing_part": "missing-like",
    "contamination": "contamination-like",
    "deformation": "deformation-like",
    "foreign_object": "foreign-object-like",
    "surface_bubble": "bubble-like",
    "discoloration": "discoloration-like",
    "material_excess": "excess-material-like",
    DEFAULT_DEFECT_FAMILY: "anomaly-like",
}

DATASET_IDS = {
    "real_iad": 0,
    "viaduct": 1,
}

# view slot budget used by episode construction and the model wrapper.
# Both Real-IAD and VIADUCT use five camera viewpoints.
MAX_VIEWS = 5


def normalize_defect_label(raw: str) -> str:
    """Map a raw defect code/folder name to a readable label."""
    if raw is None:
        return "defect"
    raw = str(raw).strip()
    if raw.upper() in REAL_IAD_DEFECT_CODES:
        return REAL_IAD_DEFECT_CODES[raw.upper()]
    label = raw.replace("_", " ").replace("-", " ").strip().lower()
    return label if label else "defect"


def defect_family_of(label: str) -> str:
    """Infer a coarse defect family from a (normalized) defect label."""
    if not label:
        return DEFAULT_DEFECT_FAMILY
    label_l = label.lower()
    for keywords, family in DEFECT_FAMILY_KEYWORDS:
        for kw in keywords:
            if kw in label_l:
                return family
    return DEFAULT_DEFECT_FAMILY
