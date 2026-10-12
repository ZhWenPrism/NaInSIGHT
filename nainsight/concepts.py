"""Canonical concept registry shared by all site adapters, the event store and the knowledge graph.

Every raw item (laboratory test, vital sign, fluid chart entry, medication) is mapped to one of the
canonical codes below by a site adapter.  Medications are mapped to an *ingredient*; each ingredient
belongs to exactly one drug *concept*, and drug concepts are the intervention nodes of the
medication-effect graph.  Laboratory concepts and urine output map to the graph *state* nodes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    code: str
    modality: str                 # lab | vital | io | drug
    unit: str
    lo: float | None = None       # plausibility window (values outside are masked as invalid)
    hi: float | None = None
    node: str | None = None       # knowledge-graph node


LAB_CONCEPTS = (
    Concept("LAB_NA", "lab", "mmol/L", 100.0, 180.0, "NA"),
    Concept("LAB_NA_BG", "lab", "mmol/L", 100.0, 180.0, "NA"),
    Concept("LAB_K", "lab", "mmol/L", 1.5, 9.0, "K"),
    Concept("LAB_CL", "lab", "mmol/L", 60.0, 150.0, "CL"),
    Concept("LAB_GLU", "lab", "mmol/L", 1.0, 50.0, "GLU"),
    Concept("LAB_UREA", "lab", "mmol/L", 0.3, 80.0, "UREA"),
    Concept("LAB_CREA", "lab", "umol/L", 10.0, 1000.0, "CREA"),
    Concept("LAB_ALB", "lab", "g/L", 10.0, 70.0, "ALB"),
    Concept("LAB_TCO2", "lab", "mmol/L", 5.0, 50.0),
    Concept("LAB_CA", "lab", "mmol/L", 1.0, 4.0),
    Concept("LAB_HGB", "lab", "g/L", 30.0, 250.0),
    Concept("LAB_WBC", "lab", "1e9/L", 0.1, 100.0),
    Concept("LAB_PLT", "lab", "1e9/L", 5.0, 1500.0),
    Concept("LAB_OSM", "lab", "mOsm/kg", 200.0, 400.0),
    Concept("LAB_UNA", "lab", "mmol/L", 5.0, 400.0),
    Concept("LAB_USG", "lab", "ratio", 1.000, 1.060),
)

VITAL_CONCEPTS = (
    Concept("VS_HR", "vital", "bpm", 20.0, 250.0),
    Concept("VS_SBP", "vital", "mmHg", 50.0, 260.0),
    Concept("VS_DBP", "vital", "mmHg", 20.0, 160.0),
    Concept("VS_MAP", "vital", "mmHg", 30.0, 200.0),
    Concept("VS_TEMP", "vital", "degC", 32.0, 43.0),
    Concept("VS_RR", "vital", "/min", 4.0, 60.0),
    Concept("VS_SPO2", "vital", "%", 50.0, 100.0),
    Concept("VS_GLU_POC", "vital", "mmol/L", 1.0, 40.0),
    Concept("VS_PUPIL_L", "vital", "mm", 1.0, 9.0),
    Concept("VS_PUPIL_R", "vital", "mm", 1.0, 9.0),
)

IO_CONCEPTS = (
    Concept("IO_URINE", "io", "mL", 0.0, 8000.0, "URINE"),
    Concept("IO_INTAKE", "io", "mL", 0.0, 8000.0),
    Concept("IO_DRAIN_CSF", "io", "mL", 0.0, 3000.0),
    Concept("IO_DRAIN_WOUND", "io", "mL", 0.0, 3000.0),
    Concept("IO_GI_LOSS", "io", "mL", 0.0, 4000.0),
)

_DRUG_CODES = (
    "MANNITOL", "HYPERTONIC_SALINE", "NORMAL_SALINE", "HYPOTONIC_SALINE", "BALANCED_CRYSTALLOID",
    "DEXTROSE_SALINE", "DEXTROSE_WATER", "ALBUMIN", "DEXAMETHASONE", "GLUCOCORTICOID_OTHER",
    "LOOP_DIURETIC", "THIAZIDE", "K_SPARING_DIURETIC", "DESMOPRESSIN", "VASOPRESSIN_ANALOG",
    "TOLVAPTAN", "POTASSIUM_CHLORIDE", "CARBAMAZEPINE", "OXCARBAZEPINE", "VALPROATE",
    "LEVETIRACETAM", "OTHER_AED", "SSRI_SNRI", "ANTIPSYCHOTIC", "OPIOID", "PPI",
)
DRUG_CONCEPTS = tuple(Concept(c, "drug", "ingredient-specific", node=c) for c in _DRUG_CODES)

ALL_CONCEPTS = LAB_CONCEPTS + VITAL_CONCEPTS + IO_CONCEPTS + DRUG_CONCEPTS
CONCEPTS = {c.code: c for c in ALL_CONCEPTS}

# ---------------------------------------------------------------------------------------------
# Knowledge-graph nodes: interventions (drug concepts) followed by physiological state nodes.
STATE_NODES = ("NA", "K", "CL", "GLU", "UREA", "CREA", "ALB", "URINE")
DRUG_NODES = _DRUG_CODES
GRAPH_NODES = DRUG_NODES + STATE_NODES
NODE_INDEX = {n: i for i, n in enumerate(GRAPH_NODES)}
STATE_INDEX = {n: i for i, n in enumerate(STATE_NODES)}
# laboratory concept that defines the value of each state node in the development cohort
STATE_SOURCE = {"NA": "LAB_NA", "K": "LAB_K", "CL": "LAB_CL", "GLU": "LAB_GLU", "UREA": "LAB_UREA",
                "CREA": "LAB_CREA", "ALB": "LAB_ALB", "URINE": "IO_URINE"}

# ---------------------------------------------------------------------------------------------
# Ingredient -> (drug concept, canonical dose unit, conversion kind)
#   kind: fluid (mL, carries a sodium load) | mass_g | mass_mg | mass_ug | iu
_ING = {
    "mannitol": ("MANNITOL", "g", "mass_g"),
    "sodium_chloride": ("NORMAL_SALINE", "mL", "fluid"),        # concept refined by concentration
    "hypertonic_saline": ("HYPERTONIC_SALINE", "mL", "fluid"),
    "normal_saline": ("NORMAL_SALINE", "mL", "fluid"),
    "half_normal_saline": ("HYPOTONIC_SALINE", "mL", "fluid"),
    "ringer": ("BALANCED_CRYSTALLOID", "mL", "fluid"),
    "ringer_lactate": ("BALANCED_CRYSTALLOID", "mL", "fluid"),
    "dextrose_saline": ("DEXTROSE_SALINE", "mL", "fluid"),
    "dextrose_water": ("DEXTROSE_WATER", "mL", "fluid"),
    "free_water": ("DEXTROSE_WATER", "mL", "fluid"),
    "albumin": ("ALBUMIN", "g", "mass_g"),
    "dexamethasone": ("DEXAMETHASONE", "mg", "mass_mg"),
    "methylprednisolone": ("GLUCOCORTICOID_OTHER", "mg", "mass_mg"),
    "hydrocortisone": ("GLUCOCORTICOID_OTHER", "mg", "mass_mg"),
    "prednisone": ("GLUCOCORTICOID_OTHER", "mg", "mass_mg"),
    "prednisolone": ("GLUCOCORTICOID_OTHER", "mg", "mass_mg"),
    "betamethasone": ("GLUCOCORTICOID_OTHER", "mg", "mass_mg"),
    "furosemide": ("LOOP_DIURETIC", "mg", "mass_mg"),
    "torasemide": ("LOOP_DIURETIC", "mg", "mass_mg"),
    "bumetanide": ("LOOP_DIURETIC", "mg", "mass_mg"),
    "hydrochlorothiazide": ("THIAZIDE", "mg", "mass_mg"),
    "indapamide": ("THIAZIDE", "mg", "mass_mg"),
    "chlorthalidone": ("THIAZIDE", "mg", "mass_mg"),
    "metolazone": ("THIAZIDE", "mg", "mass_mg"),
    "spironolactone": ("K_SPARING_DIURETIC", "mg", "mass_mg"),
    "eplerenone": ("K_SPARING_DIURETIC", "mg", "mass_mg"),
    "amiloride": ("K_SPARING_DIURETIC", "mg", "mass_mg"),
    "triamterene": ("K_SPARING_DIURETIC", "mg", "mass_mg"),
    "desmopressin": ("DESMOPRESSIN", "ug", "mass_ug"),
    "vasopressin": ("VASOPRESSIN_ANALOG", "IU", "iu"),
    "pituitrin": ("VASOPRESSIN_ANALOG", "IU", "iu"),
    "terlipressin": ("VASOPRESSIN_ANALOG", "mg", "mass_mg"),
    "tolvaptan": ("TOLVAPTAN", "mg", "mass_mg"),
    "potassium_chloride": ("POTASSIUM_CHLORIDE", "g", "mass_g"),
    "carbamazepine": ("CARBAMAZEPINE", "mg", "mass_mg"),
    "oxcarbazepine": ("OXCARBAZEPINE", "mg", "mass_mg"),
    "valproate": ("VALPROATE", "mg", "mass_mg"),
    "levetiracetam": ("LEVETIRACETAM", "mg", "mass_mg"),
}
for _name in ("phenobarbital", "phenytoin", "lamotrigine", "topiramate", "lacosamide", "zonisamide",
              "perampanel", "gabapentin", "pregabalin"):
    _ING[_name] = ("OTHER_AED", "mg", "mass_mg")
for _name in ("sertraline", "paroxetine", "fluoxetine", "citalopram", "escitalopram", "venlafaxine",
              "duloxetine", "mirtazapine"):
    _ING[_name] = ("SSRI_SNRI", "mg", "mass_mg")
for _name in ("olanzapine", "quetiapine", "risperidone", "haloperidol"):
    _ING[_name] = ("ANTIPSYCHOTIC", "mg", "mass_mg")
for _name in ("morphine", "fentanyl", "sufentanil", "remifentanil", "hydromorphone", "oxycodone",
              "tramadol", "pethidine"):
    _ING[_name] = ("OPIOID", "mg", "mass_mg")
for _name in ("omeprazole", "esomeprazole", "pantoprazole", "lansoprazole", "rabeprazole"):
    _ING[_name] = ("PPI", "mg", "mass_mg")
INGREDIENTS: dict[str, tuple[str, str, str]] = dict(_ING)

# glucocorticoid potency relative to prednisone (hydrocortisone 20 = prednisone 5 = methylprednisolone 4
# = dexamethasone 0.75 mg)
PREDNISONE_EQUIVALENT = {"hydrocortisone": 0.25, "prednisone": 1.0, "prednisolone": 1.0,
                         "methylprednisolone": 1.25, "dexamethasone": 5.0 / 0.75, "betamethasone": 5.0 / 0.75}

# sodium content of crystalloids (mmol/L) when the label carries no explicit concentration
FLUID_SODIUM_MMOL_PER_L = {"normal_saline": 154.0, "half_normal_saline": 77.0, "dextrose_saline": 154.0,
                           "ringer": 147.0, "ringer_lactate": 130.0, "dextrose_water": 0.0, "free_water": 0.0}

ROUTES = ("IV", "PO", "NG", "IM", "SC", "IT", "OTHER")
MODALITIES = ("special", "lab", "vital", "io", "drug")
PHASES = ("preop", "intraop", "postop")
TONICITY = ("", "hypertonic", "isotonic", "hypotonic", "sodium_free")


def concept_node(code: str) -> str | None:
    c = CONCEPTS.get(code)
    return None if c is None else c.node
