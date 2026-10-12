"""Dose normalisation for medication events.

* every ingredient is converted to its canonical unit (``concepts.INGREDIENTS``) using the label
  strength or concentration found in the product text (e.g. ``20% 250 mL``, ``10 mL:1 g``, ``0.1 mg x 30``);
* crystalloids additionally carry a sodium load (mmol) and a tonicity class, and unlabelled sodium
  chloride is assigned to the hyper-/iso-/hypotonic concept from its parsed concentration;
* glucocorticoids carry a prednisone-equivalent dose;
* a dose that cannot be converted stays NaN - it is never imputed as zero.  Downstream, doses are
  discretised into *within-ingredient* quantile levels, so no cross-drug potency equivalence is assumed.
Conversion factors are computed once per unique (ingredient, unit, product text) triple.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from ..concepts import FLUID_SODIUM_MMOL_PER_L, INGREDIENTS, PREDNISONE_EQUIVALENT

NACL_G_PER_MOL = 58.44
_N = r"(\d+(?:\.\d+)?)"
_MASS = r"(g|mg|ug|mcg|μg)"
RE_PCT = re.compile(_N + r"\s*%")
RE_VOL_PER_MASS = re.compile(_N + r"\s*ml\s*[:/]\s*" + _N + r"\s*" + _MASS)          # 10ml:1g
RE_MASS_PER_VOL = re.compile(_N + r"\s*" + _MASS + r"\s*[:/]\s*" + _N + r"\s*ml")      # 1.5g:10ml
RE_MASS_THEN_VOL = re.compile(_N + r"\s*" + _MASS + r"\s+" + _N + r"\s*ml")            # 400mg 4ml
RE_UNIT_STRENGTH = re.compile(_N + r"\s*" + _MASS + r"(?![a-z/:])")                    # 0.5g*30, 40mg
RE_IU_PER_VOL = [re.compile(_N + r"\s*ml\s*[:/]?\s*" + _N + r"\s*(?:iu|u)\b"),           # 1ml 6iu
                 re.compile(_N + r"\s*(?:iu|u)\s*[:/]?\s*" + _N + r"\s*ml")]            # 6iu 1ml
RE_CONTAINER_ML = re.compile(_N + r"\s*ml")

DEFAULT_UNIT_ALIASES = {
    "ml": "mL", "l": "L", "g": "g", "gm": "g", "mg": "mg", "ug": "ug", "mcg": "ug", "μg": "ug",
    "iu": "IU", "u": "IU", "unit": "IU", "units": "IU", "meq": "mEq", "mmol": "mmol",
    "tab": "unit_dose", "tablet": "unit_dose", "cap": "unit_dose", "capsule": "unit_dose",
    "vial": "unit_dose", "amp": "unit_dose", "bag": "unit_dose", "bottle": "unit_dose",
}
_MASS_TO_MG = {"g": 1000.0, "mg": 1.0, "ug": 1e-3, "mcg": 1e-3, "μg": 1e-3}
_KIND_TARGET_MG = {"mass_g": 1000.0, "mass_mg": 1.0, "mass_ug": 1e-3}


def normalise_text(s: pd.Series) -> pd.Series:
    return (s.astype("string").fillna("").str.normalize("NFKC").str.lower()
            .str.replace("×", "*", regex=False).str.replace(r"\s+", " ", regex=True).str.strip())


def normalise_unit(s: pd.Series, aliases: dict | None = None) -> pd.Series:
    table = dict(DEFAULT_UNIT_ALIASES)
    table.update({str(k).lower(): v for k, v in (aliases or {}).items()})
    u = s.astype("string").fillna("").str.normalize("NFKC").str.strip()
    return u.str.lower().map(table).fillna(u)


# ------------------------------------------------------------------------------ text parsers
def mass_concentration_mg_per_ml(text: str) -> float | None:
    m = RE_VOL_PER_MASS.search(text)
    if m:
        ml, amt, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        return amt * _MASS_TO_MG[unit] / ml if ml > 0 else None
    for rx in (RE_MASS_PER_VOL, RE_MASS_THEN_VOL):
        m = rx.search(text)
        if m:
            amt, unit, ml = float(m.group(1)), m.group(2), float(m.group(3))
            return amt * _MASS_TO_MG[unit] / ml if ml > 0 else None
    m = RE_PCT.search(text)
    if m:
        pct = float(m.group(1))
        if 0 < pct <= 100:
            return pct * 10.0                               # x % w/v = x g / 100 mL = 10x mg/mL
    return None


def unit_strength_mg(text: str) -> float | None:
    m = RE_UNIT_STRENGTH.search(text)
    return float(m.group(1)) * _MASS_TO_MG[m.group(2)] if m else None


def iu_per_ml(text: str) -> float | None:
    m = RE_IU_PER_VOL[0].search(text)
    if m and float(m.group(1)) > 0:
        return float(m.group(2)) / float(m.group(1))
    m = RE_IU_PER_VOL[1].search(text)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def sodium_chloride_percent(text: str) -> float | None:
    m = RE_PCT.search(text)
    if m:
        return float(m.group(1))
    c = mass_concentration_mg_per_ml(text)
    return c / 10.0 if c else None


# ------------------------------------------------------------------------------ conversion table
def _fluid_profile(ingredient: str, text: str, defaults: dict) -> tuple[str, float, str]:
    """-> (concept, sodium mmol per mL, tonicity)."""
    concept = INGREDIENTS[ingredient][0]
    if ingredient in ("sodium_chloride", "hypertonic_saline"):
        pct = sodium_chloride_percent(text)
        if pct is None:
            pct = float(defaults.get("hypertonic_saline_pct", 10.0) if ingredient == "hypertonic_saline"
                        else defaults.get("sodium_chloride_pct", 0.9))
        na_per_ml = pct * 10.0 / NACL_G_PER_MOL
        if pct >= 1.5:
            return "HYPERTONIC_SALINE", na_per_ml, "hypertonic"
        if pct >= 0.8:
            return "NORMAL_SALINE", na_per_ml, "isotonic"
        return "HYPOTONIC_SALINE", na_per_ml, "hypotonic"
    na_l = FLUID_SODIUM_MMOL_PER_L.get(ingredient, 0.0)
    tonicity = "sodium_free" if na_l == 0 else ("isotonic" if na_l >= 120 else "hypotonic")
    return concept, na_l / 1000.0, tonicity


def _container_volume_ml(text: str) -> float | None:
    vols = [float(v) for v in RE_CONTAINER_ML.findall(text)]
    return max(vols) if vols else None


def conversion_row(ingredient: str, unit: str, text: str, defaults: dict) -> dict:
    concept, canon_unit, kind = INGREDIENTS[ingredient]
    out = {"concept": concept, "dose_unit": canon_unit, "factor": np.nan, "na_per_ml": np.nan,
           "tonicity": "", "pred_eq": PREDNISONE_EQUIVALENT.get(ingredient, np.nan)}
    if kind == "fluid":
        concept, na_per_ml, tonicity = _fluid_profile(ingredient, text, defaults)
        out.update(concept=concept, na_per_ml=na_per_ml, tonicity=tonicity)
        if unit == "mL":
            out["factor"] = 1.0
        elif unit == "L":
            out["factor"] = 1000.0
        elif unit == "unit_dose":
            vol = _container_volume_ml(text)
            out["factor"] = vol if vol else np.nan
        return out
    if kind in _KIND_TARGET_MG:
        target = _KIND_TARGET_MG[kind]
        if unit in _MASS_TO_MG:
            out["factor"] = _MASS_TO_MG[unit] / target
        elif unit == "mL":
            c = mass_concentration_mg_per_ml(text)
            out["factor"] = c / target if c else np.nan
        elif unit == "unit_dose":
            s = unit_strength_mg(text)
            out["factor"] = s / target if s else np.nan
        return out
    if kind == "iu":
        if unit == "IU":
            out["factor"] = 1.0
        elif unit == "mL":
            c = iu_per_ml(text)
            out["factor"] = c if c else np.nan
    return out


def normalise_doses(df: pd.DataFrame, unit_aliases: dict | None = None, defaults: dict | None = None) -> pd.DataFrame:
    """Input columns: ingredient, value, unit, text.  Adds concept, dose_std, dose_unit, volume_ml,
    na_mmol, tonicity, pred_eq_mg."""
    defaults = defaults or {}
    out = df.copy()
    out["unit_n"] = normalise_unit(out["unit"], unit_aliases)
    out["text_n"] = normalise_text(out["text"])
    keys = out[["ingredient", "unit_n", "text_n"]].drop_duplicates()
    table = pd.DataFrame([conversion_row(i, u, t, defaults) for i, u, t in keys.itertuples(index=False)])
    table.index = keys.index
    keys = pd.concat([keys, table], axis=1)
    out = out.merge(keys, on=["ingredient", "unit_n", "text_n"], how="left")
    value = pd.to_numeric(out["value"], errors="coerce")
    out["dose_std"] = value * out["factor"]
    out.loc[~(out["dose_std"] > 0), "dose_std"] = np.nan
    is_fluid = out["na_per_ml"].notna()
    out["volume_ml"] = np.where(out["unit_n"].eq("mL"), value, np.where(is_fluid, out["dose_std"], np.nan))
    out["na_mmol"] = np.where(is_fluid, out["dose_std"] * out["na_per_ml"], np.nan)
    mg = out["dose_std"] * out["dose_unit"].map({"mg": 1.0, "g": 1000.0, "ug": 1e-3}).astype(float)
    out["pred_eq_mg"] = mg * out["pred_eq"]
    return out.drop(columns=["unit_n", "text_n", "factor", "na_per_ml", "pred_eq"])
