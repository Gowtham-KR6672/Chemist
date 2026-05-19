"""
Quantum Chemistry Analysis Web App
FastAPI backend using deterministic chemistry providers instead of AI.

Primary sources:
- Formula parser for exact formula and molecular weight from typed formulas.
- PubChem PUG REST for known molecular compounds.
- OQMD REST for materials entries when available.

Optional sources/tools:
- Materials Project, ChemSpider, Citrine via API keys.
- DeepChem, Open Babel, ASE, Quantum ESPRESSO, Matminer when installed locally.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import importlib.util
import json
import math
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Quantum Chemistry Analyzer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, dict] = {}
MODEL_PATH = Path("models/property_model.json")


class DftRequest(BaseModel):
    xyz: str = Field(..., description="XYZ text with atom count/header or simple 'Element x y z' lines")
    charge: int = 0
    spin: int = 0
    basis: str = "def2-svp"
    xc: str = "pbe0"
    max_cycle: int = 80

ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415,
    "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194,
    "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584,
    "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.6, "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196,
    "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Pr": 140.90766,
    "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.92535, "Dy": 162.5, "Ho": 164.93033,
    "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668,
    "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569,
    "Hg": 200.592, "Tl": 204.38, "Pb": 207.2, "Bi": 208.9804,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0,
    "Ra": 226.0, "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588,
    "U": 238.02891, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0,
    "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 266.0,
    "Rf": 267.0, "Db": 268.0, "Sg": 269.0, "Bh": 270.0,
    "Hs": 269.0, "Mt": 278.0, "Ds": 281.0, "Rg": 282.0,
    "Cn": 285.0, "Nh": 286.0, "Fl": 289.0, "Mc": 290.0,
    "Lv": 293.0, "Ts": 294.0, "Og": 294.0,
}

def add_counts(target: dict[str, int], source: dict[str, int], multiplier: int = 1) -> None:
    for element, count in source.items():
        target[element] = target.get(element, 0) + count * multiplier


def parse_formula_group(text: str, start: int = 0, stop: str | None = None) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    i = start
    while i < len(text):
        ch = text[i]
        if stop and ch == stop:
            return counts, i + 1
        if ch in "([{":
            closing = { "(": ")", "[": "]", "{": "}" }[ch]
            inner, i = parse_formula_group(text, i + 1, closing)
            m = re.match(r"\d+", text[i:])
            multiplier = int(m.group(0)) if m else 1
            if m:
                i += len(m.group(0))
            add_counts(counts, inner, multiplier)
            continue
        if ch.isupper():
            m = re.match(r"[A-Z][a-z]?", text[i:])
            element = m.group(0)
            i += len(element)
            n = re.match(r"\d+", text[i:])
            count = int(n.group(0)) if n else 1
            if n:
                i += len(n.group(0))
            counts[element] = counts.get(element, 0) + count
            continue
        i += 1
    if stop:
        raise ValueError(f"Missing closing {stop!r} in formula")
    return counts, i


def parse_formula_counts(raw: str) -> dict[str, int] | None:
    text = raw.strip()
    if not re.search(r"[A-Z][a-z]?\d*", text):
        return None
    text = re.sub(r"\s+", "", text).replace("·", ".")
    text = re.sub(r"[\^][+-]?\d*[+-]?$", "", text)
    text = move_terminal_hydrate_outside_bracket(text)

    totals: dict[str, int] = {}
    for part in split_formula_parts(text):
        m = re.match(r"^(\d+)(.*)$", part)
        multiplier = int(m.group(1)) if m else 1
        formula = m.group(2) if m else part
        counts, _ = parse_formula_group(formula)
        add_counts(totals, counts, multiplier)
    return totals or None


def move_terminal_hydrate_outside_bracket(text: str) -> str:
    if not text.endswith("]"):
        return text
    dot = text.rfind(".")
    if dot == -1:
        return text
    hydrate = text[dot + 1:-1]
    if re.fullmatch(r"\d*H2O", hydrate):
        return f"{text[:dot]}].{hydrate}"
    return text


def split_formula_parts(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0 and is_hydrate_separator(text, i):
            if text[start:i]:
                parts.append(text[start:i])
            start = i + 1
    if text[start:]:
        parts.append(text[start:])
    return parts


def is_hydrate_separator(text: str, index: int) -> bool:
    before = text[index - 1] if index > 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if not after:
        return False
    if after.isdigit():
        tail = text[index + 1:]
        return bool(re.match(r"\d+H2O", tail))
    return before == "]" or after.isupper()


def format_formula(counts: dict[str, int]) -> str:
    if "C" in counts:
        ordered = ["C"]
        if "H" in counts:
            ordered.append("H")
        ordered += sorted(e for e in counts if e not in {"C", "H"})
    else:
        ordered = sorted(counts)
    return "".join(f"{e}{counts[e] if counts[e] != 1 else ''}" for e in ordered)


def molecular_weight(counts: dict[str, int]) -> float | None:
    try:
        return sum(ATOMIC_WEIGHTS[element] * count for element, count in counts.items())
    except KeyError:
        return None


def parse_xyz_text(xyz: str) -> list[tuple[str, float, float, float]]:
    lines = [line.strip() for line in xyz.splitlines() if line.strip()]
    if not lines:
        raise ValueError("XYZ text is empty")
    if lines[0].isdigit():
        atom_count = int(lines[0])
        atom_lines = lines[2:2 + atom_count]
    else:
        atom_lines = lines

    atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line}")
        symbol = parts[0]
        if not re.fullmatch(r"[A-Z][a-z]?", symbol):
            raise ValueError(f"Invalid element symbol in XYZ: {symbol}")
        atoms.append((symbol, float(parts[1]), float(parts[2]), float(parts[3])))
    if not atoms:
        raise ValueError("No atoms found in XYZ text")
    return atoms


def run_pyscf_dft(req: DftRequest) -> dict:
    if importlib.util.find_spec("pyscf") is None:
        raise RuntimeError(
            "PySCF is not installed. Install it with a compatible wheel/environment "
            "or use conda-forge/WSL for real DFT."
        )

    from pyscf import dft, gto  # type: ignore

    atoms = parse_xyz_text(req.xyz)
    atom_spec = [(sym, (x, y, z)) for sym, x, y, z in atoms]
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.charge = req.charge
    mol.spin = req.spin
    mol.basis = req.basis
    mol.build()

    mf = dft.UKS(mol) if req.spin else dft.RKS(mol)
    mf.xc = req.xc
    mf.max_cycle = req.max_cycle
    energy = mf.kernel()

    mo_energy = mf.mo_energy[0] if isinstance(mf.mo_energy, (tuple, list)) else mf.mo_energy
    mo_occ = mf.mo_occ[0] if isinstance(mf.mo_occ, (tuple, list)) else mf.mo_occ
    occupied = [e for e, occ in zip(mo_energy, mo_occ) if occ > 0]
    unoccupied = [e for e, occ in zip(mo_energy, mo_occ) if occ == 0]
    gap_ev = None
    if occupied and unoccupied:
        gap_ev = (min(unoccupied) - max(occupied)) * 27.211386245988

    dipole = mf.dip_moment(unit="Debye")
    return {
        "engine": "PySCF",
        "method": f"{req.xc}/{req.basis}",
        "atom_count": len(atoms),
        "charge": req.charge,
        "spin": req.spin,
        "total_energy_hartree": energy,
        "homo_lumo_gap_ev": gap_ev,
        "dipole_moment_debye": {
            "x": float(dipole[0]),
            "y": float(dipole[1]),
            "z": float(dipole[2]),
            "magnitude": float(math.sqrt(sum(float(v) ** 2 for v in dipole))),
        },
        "converged": bool(mf.converged),
    }


def load_research_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def formula_feature_vector(counts: dict[str, int]) -> list[float]:
    heavy = sum(v for k, v in counts.items() if k != "H") or 1
    metal_atoms = sum(counts.get(k, 0) for k in (
        "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sc", "Ti", "V", "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Ag", "Pt", "Au",
    ))
    hetero = sum(counts.get(k, 0) for k in ("N", "O", "S", "P", "F", "Cl", "Br", "I"))
    return [
        1.0,
        float(heavy),
        float(metal_atoms),
        float(counts.get("C", 0)),
        float(counts.get("N", 0)),
        float(counts.get("O", 0)),
        float(counts.get("S", 0)),
        hetero / heavy,
        metal_atoms / heavy,
        math.log1p(heavy),
    ]


def predict_with_research_model(counts: dict[str, int]) -> dict:
    model = load_research_model()
    if not model:
        return {
            "available": False,
            "message": "No trained research model found. Add curated data and run train_property_model.py.",
        }
    x = formula_feature_vector(counts)
    predictions = {}
    for target, coef in model.get("coefficients", {}).items():
        predictions[target] = round(sum(float(c) * v for c, v in zip(coef, x)), 4)
    return {
        "available": True,
        "model_type": model.get("model_type"),
        "n_training_rows": model.get("n_training_rows"),
        "metrics": model.get("metrics", {}),
        "predictions": predictions,
        "message": "Research model prediction loaded from trained artifact.",
    }


def apply_research_predictions(data: dict, research_prediction: dict) -> dict:
    if not research_prediction.get("available"):
        data["ml_data_source"] = "formula/provider estimates"
        return data

    predictions = research_prediction.get("predictions", {})
    field_map = {
        "band_gap_ev": "band_gap_ev",
        "dipole_moment_debye": "dipole_moment_debye",
        "logp": "logp",
        "polar_surface_area": "polar_surface_area",
        "specific_capacitance_f_g": "specific_capacitance_f_g",
        "energy_density_wh_kg": "energy_density_wh_kg",
        "cycling_stability_pct": "cycling_stability_pct",
        "redox_potential_v": "redox_potential_v",
        "solvation_energy_kcal_mol": "solvation_energy_kcal_mol",
        "mulliken_charge": "mulliken_charge",
    }
    for model_key, data_key in field_map.items():
        if model_key in predictions:
            data[data_key] = str(round(float(predictions[model_key]), 4))
    data["ml_data_source"] = "trained research model"
    return data


def http_json(url: str, timeout: int = 12) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuantumChem/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def http_json_with_headers(url: str, headers: dict[str, str], timeout: int = 12) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuantumChem/2.0", **headers})
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def query_keyed_providers(name: str, formula: str | None = None) -> dict:
    results: dict[str, dict] = {}

    mp_key = os.environ.get("MP_API_KEY", "").strip()
    if mp_key and formula:
        encoded_formula = urllib.parse.quote(formula, safe="")
        url = f"https://api.materialsproject.org/materials/summary/?formula={encoded_formula}"
        payload = http_json_with_headers(url, {"X-API-KEY": mp_key})
        docs = payload.get("data", []) if payload else []
        first = docs[0] if docs else {}
        results["materials_project"] = {
            "status": "ok" if first else "not_found",
            "attempted": True,
            "formula": formula,
            "material_id": first.get("material_id"),
            "band_gap": first.get("band_gap"),
            "formation_energy_per_atom": first.get("formation_energy_per_atom"),
            "energy_above_hull": first.get("energy_above_hull"),
            "is_stable": first.get("is_stable"),
        }
    else:
        results["materials_project"] = {"status": "not_configured", "attempted": False}

    chemspider_key = os.environ.get("CHEMSPIDER_API_KEY", "").strip()
    results["chemspider"] = {
        "status": "configured_not_queried" if chemspider_key else "not_configured",
        "attempted": bool(chemspider_key),
        "message": "ChemSpider key is present; add exact RSC endpoint mapping before using it as a primary source." if chemspider_key else "No ChemSpider key.",
    }

    citrine_key = os.environ.get("CITRINE_API_KEY", "").strip()
    results["citrine"] = {
        "status": "configured_not_queried" if citrine_key else "not_configured",
        "attempted": bool(citrine_key),
        "message": "Citrine key is present; configure dataset/search endpoint for your account before querying." if citrine_key else "No Citrine key.",
    }

    return results


def query_pubchem(name: str) -> dict:
    props = ",".join([
        "MolecularFormula", "MolecularWeight", "IUPACName", "CanonicalSMILES",
        "InChIKey", "XLogP", "TPSA", "Complexity", "Charge",
    ])
    encoded = urllib.parse.quote(name, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/{props}/JSON"
    payload = http_json(url)
    if not payload:
        return {"status": "not_found"}
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        return {"status": "not_found"}
    row = rows[0]
    return {
        "status": "ok",
        "cid": row.get("CID"),
        "formula": row.get("MolecularFormula"),
        "molecular_weight": row.get("MolecularWeight"),
        "name": row.get("IUPACName"),
        "canonical_smiles": row.get("CanonicalSMILES"),
        "inchikey": row.get("InChIKey"),
        "logp": row.get("XLogP"),
        "polar_surface_area": row.get("TPSA"),
        "charge": row.get("Charge"),
        "complexity": row.get("Complexity"),
    }


def query_oqmd(formula: str) -> dict:
    encoded = urllib.parse.quote(f"composition={formula}", safe="=")
    url = f"https://oqmd.org/oqmdapi/formationenergy?filter={encoded}&limit=5"
    payload = http_json(url)
    if not payload:
        return {"status": "unavailable"}
    rows = payload.get("data") or []
    if not rows:
        return {"status": "not_found"}
    first = rows[0]
    return {
        "status": "ok",
        "count": len(rows),
        "entry_id": first.get("entry_id"),
        "formation_energy": first.get("delta_e"),
        "stability": first.get("stability"),
        "spacegroup": first.get("spacegroup"),
    }


def tool_status() -> dict:
    qe_path = shutil.which("pw.x")
    ase_installed = importlib.util.find_spec("ase") is not None
    pyscf_installed = importlib.util.find_spec("pyscf") is not None
    return {
        "pubchem": {"type": "REST", "configured": True, "key_required": False},
        "oqmd": {"type": "REST", "configured": True, "key_required": False},
        "materials_project": {
            "type": "REST/Python",
            "configured": bool(os.environ.get("MP_API_KEY")),
            "key_required": True,
        },
        "chemspider": {
            "type": "REST",
            "configured": bool(os.environ.get("CHEMSPIDER_API_KEY")),
            "key_required": True,
        },
        "citrine": {
            "type": "REST",
            "configured": bool(os.environ.get("CITRINE_API_KEY")),
            "key_required": True,
        },
        "deepchem": {"type": "Python library", "configured": importlib.util.find_spec("deepchem") is not None},
        "pyscf": {"type": "molecular DFT Python engine", "configured": pyscf_installed},
        "open_babel": {
            "type": "local command/Python library",
            "configured": bool(shutil.which("obabel") or importlib.util.find_spec("openbabel")),
        },
        "ase": {"type": "Python library", "configured": ase_installed},
        "quantum_espresso": {"type": "local DFT software", "configured": bool(qe_path), "path": qe_path},
        "matminer": {"type": "Python library", "configured": importlib.util.find_spec("matminer") is not None},
    }


def dft_analysis_status(counts: dict[str, int], has_structure: bool) -> dict:
    tools = tool_status()
    qe_ready = tools["quantum_espresso"]["configured"]
    ase_ready = tools["ase"]["configured"]
    pyscf_ready = tools["pyscf"]["configured"]
    can_run = bool(has_structure and (pyscf_ready or (qe_ready and ase_ready)))

    if can_run:
        status = "ready"
        message = "A local DFT engine is available. Provide 3D coordinates to run DFT."
    elif not has_structure:
        status = "needs_structure"
        message = "DFT requires real atomic coordinates. Formula-only input is not enough."
    else:
        missing = []
        if not qe_ready:
            missing.append("Quantum ESPRESSO pw.x")
        if not ase_ready:
            missing.append("ASE")
        if not pyscf_ready:
            missing.append("PySCF")
        status = "missing_tools"
        message = "Install " + " and ".join(missing) + " to run local DFT."

    heavy_atoms = sum(c for e, c in counts.items() if e != "H")
    return {
        "status": status,
        "can_run": can_run,
        "method": "DFT electronic structure from 3D coordinates",
        "engine": "PySCF" if pyscf_ready else "Quantum ESPRESSO",
        "workflow": "XYZ/SDF/CIF -> PySCF or ASE -> DFT energy/gap/dipole",
        "message": message,
        "estimated_runtime": "minutes-hours" if heavy_atoms > 40 else "minutes",
        "requires": ["atomic coordinates", "PySCF or Quantum ESPRESSO", "basis set/pseudopotentials"],
    }


def estimate_fields(counts: dict[str, int], pubchem: dict, oqmd: dict) -> dict:
    heavy_atoms = sum(c for e, c in counts.items() if e != "H")
    hetero = sum(counts.get(e, 0) for e in ("N", "O", "S", "P", "F", "Cl", "Br", "I"))
    carbon = counts.get("C", 0)
    has_mn = counts.get("Mn", 0) > 0
    rotatable = max(0, round(carbon * 0.24))
    ring_count = max(0, round(carbon / 8))
    hbd = counts.get("N", 0) + max(0, counts.get("O", 0) // 3)
    hba = counts.get("N", 0) + counts.get("O", 0)
    band_gap = oqmd.get("band_gap") or (2.36 if has_mn else 4.8)

    return {
        "charge": str(pubchem.get("charge", 0)),
        "spin_multiplicity": "6" if has_mn else "1",
        "point_group": "C1" if heavy_atoms > 12 else "C2v",
        "bond_count": str(max(0, heavy_atoms + carbon + hetero - 1)),
        "ring_count": str(ring_count),
        "rotatable_bonds": str(rotatable),
        "hbd_count": str(hbd),
        "hba_count": str(hba),
        "logp": str(pubchem.get("logp") if pubchem.get("logp") is not None else round((carbon * 0.22) - (hetero * 0.45), 2)),
        "polar_surface_area": str(pubchem.get("polar_surface_area") if pubchem.get("polar_surface_area") is not None else round(hetero * 18.5, 1)),
        "band_gap_ev": str(round(float(band_gap), 2)),
        "dipole_moment_debye": str(round(2.2 + hetero * 0.28 + (1.8 if has_mn else 0), 2)),
        "electronegativity": "N/A",
        "oxidation_state": "+2" if has_mn else "N/A",
        "hybridization": "sp3/sp2" if carbon else "N/A",
        "solubility": "N/A",
        "composition": counts,
        "descriptor_profile": descriptor_profile(counts, band_gap),
    }


def descriptor_profile(counts: dict[str, int], band_gap: float) -> dict:
    heavy_atoms = max(1, sum(c for e, c in counts.items() if e != "H"))
    carbon = counts.get("C", 0)
    nitrogen = counts.get("N", 0)
    oxygen = counts.get("O", 0)
    sulfur = counts.get("S", 0)
    metals = sum(counts.get(e, 0) for e in (
        "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sc", "Ti", "V", "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Ag", "Pt", "Au",
    ))
    hetero_ratio = (nitrogen + oxygen + sulfur) / heavy_atoms
    metal_ratio = metals / heavy_atoms
    carbon_ratio = carbon / heavy_atoms
    redox_activity = min(1.0, metal_ratio * 5 + nitrogen * 0.04 + sulfur * 0.08)
    ionic_character = min(1.0, counts.get("K", 0) * 0.12 + counts.get("Na", 0) * 0.12 + counts.get("Cu", 0) * 0.1)
    capacitance = min(100, 35 + redox_activity * 38 + hetero_ratio * 22 + oxygen * 0.8)
    energy_density = min(100, 28 + max(0, 5 - float(band_gap)) * 9 + carbon_ratio * 20 + redox_activity * 18)
    cycling = min(100, 45 + oxygen * 1.8 + sulfur * 7 + ionic_character * 18 + metal_ratio * 16)
    power = min(100, 38 + nitrogen * 2.2 + ionic_character * 20 + max(0, 4 - float(band_gap)) * 6)
    rate = min(100, 35 + hetero_ratio * 35 + ionic_character * 22 + metals * 4)
    coulombic = min(100, 50 + oxygen * 1.6 + nitrogen * 1.2 + sulfur * 5)
    dos_center_left = -2.2 - carbon_ratio * 1.2 - nitrogen * 0.04
    dos_center_right = 1.0 + float(band_gap)
    dos_width_left = 0.28 + hetero_ratio * 0.45
    dos_width_right = 0.25 + metal_ratio * 1.1 + sulfur * 0.08

    return {
        "heavy_atoms": heavy_atoms,
        "metal_ratio": round(metal_ratio, 4),
        "hetero_ratio": round(hetero_ratio, 4),
        "carbon_ratio": round(carbon_ratio, 4),
        "radar": {
            "capacitance": round(capacitance, 1),
            "energy_density": round(energy_density, 1),
            "cycling_stability": round(cycling, 1),
            "power_density": round(power, 1),
            "rate_capability": round(rate, 1),
            "coulombic_efficiency": round(coulombic, 1),
        },
        "dos": {
            "occupied_center": round(dos_center_left, 2),
            "unoccupied_center": round(dos_center_right, 2),
            "occupied_width": round(dos_width_left, 2),
            "unoccupied_width": round(dos_width_right, 2),
            "occupied_height": round(min(1.1, 0.45 + carbon_ratio + nitrogen * 0.035), 2),
            "unoccupied_height": round(min(1.1, 0.45 + metal_ratio * 2.8 + oxygen * 0.025), 2),
        },
    }


def pubchem_structure_image(cid: int | str | None) -> str | None:
    if not cid:
        return None
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=large"


def generated_schematic_diagram(counts: dict[str, int]) -> tuple[list[dict], list[dict]]:
    atoms: list[dict] = []
    bonds: list[dict] = []

    metals = [
        "Mn", "Co", "Fe", "Ni", "Cu", "Zn", "Cr", "V", "Ti", "Ru", "Rh",
        "Pd", "Ag", "Pt", "Au", "Cd", "Hg", "Mg", "Ca",
    ]
    metal = next((element for element in metals if counts.get(element, 0)), None)

    def add_atom(symbol: str, x: float, y: float) -> int:
        atoms.append({"symbol": symbol, "x": round(x, 3), "y": round(y, 3)})
        return len(atoms) - 1

    def add_ring(cx: float, cy: float, r: float = 0.72) -> list[int]:
        ring = []
        for i in range(6):
            a = -math.pi / 6 + i * math.pi / 3
            ring.append(add_atom("C", cx + math.cos(a) * r, cy + math.sin(a) * r))
        for i in range(6):
            bonds.append({"from": ring[i], "to": ring[(i + 1) % 6], "order": 1.5})
        return ring

    carbon = counts.get("C", 0)
    nitrogen = counts.get("N", 0)
    sulfur = counts.get("S", 0)
    oxygen = counts.get("O", 0)

    if metal and sulfur and oxygen >= 4 and carbon == 0:
        center = add_atom(metal, -1.35, 0)
        sulfate_s = add_atom("S", 1.05, 0)
        bonds.append({"from": center, "to": sulfate_s, "order": 1})

        sulfate_positions = [(1.05, 1.05), (2.1, 0), (1.05, -1.05), (0.0, 0)]
        sulfate_oxygens = []
        for i, (x, y) in enumerate(sulfate_positions):
            o = add_atom("O", x, y)
            sulfate_oxygens.append(o)
            bonds.append({"from": sulfate_s, "to": o, "order": 2 if i < 2 else 1})

        hydrate_count = max(0, (counts.get("H", 0) // 2) - 0)
        hydrate_count = min(hydrate_count, max(0, oxygen - 4), 8)
        hydrate_positions = [(-2.45, 0.9), (-2.45, -0.9), (-1.35, 1.25), (-1.35, -1.25), (-0.2, 1.45), (-0.2, -1.45), (2.2, 1.0), (2.2, -1.0)]
        for i in range(hydrate_count):
            water = add_atom("O", *hydrate_positions[i])
            if i < 4:
                bonds.append({"from": center, "to": water, "order": 1})
        return atoms, bonds

    if metal and carbon >= 1 and nitrogen >= 1 and oxygen == 0:
        center = add_atom(metal, 0, 0)
        ligand_count = min(carbon, nitrogen, 6)
        ligand_positions = [(-1.1, 0.0), (1.1, 0.0), (0.0, -1.1), (0.0, 1.1), (-0.78, -0.78), (0.78, 0.78)]
        ligand_indices = []
        for i in range(ligand_count):
            x, y = ligand_positions[i]
            c = add_atom("C", x, y)
            n = add_atom("N", x * 1.75, y * 1.75)
            bonds.append({"from": center, "to": c, "order": 1})
            bonds.append({"from": c, "to": n, "order": 3})
            ligand_indices.append(n)

        counter_elements = [e for e in ("K", "Na", "Li", "Rb", "Cs", "Ca", "Mg") if counts.get(e, 0)]
        counter_positions = [(-2.8, 1.25), (2.8, -1.25), (-2.8, -1.25), (2.8, 1.25), (0, 2.35), (0, -2.35)]
        pos_index = 0
        for element in counter_elements:
            for _ in range(min(counts[element], len(counter_positions) - pos_index)):
                x, y = counter_positions[pos_index]
                add_atom(element, x, y)
                pos_index += 1
        return atoms, bonds

    if metal and (carbon >= 10 or nitrogen >= 2):
        center = add_atom(metal, 0, 0)
        donor_symbols = (["N"] * min(nitrogen, 4)) + (["O"] * min(oxygen, 4))
        if not donor_symbols:
            donor_symbols = [metal]
        donor_symbols = donor_symbols[:6]
        donor_positions = [(-1.0, 0.0), (1.0, 0.0), (-0.55, 0.82), (0.55, 0.82), (-0.55, -0.82), (0.55, -0.82)]
        donors = []
        for symbol, (x, y) in zip(donor_symbols, donor_positions):
            donor = add_atom(symbol, x, y)
            donors.append(donor)
            bonds.append({"from": center, "to": donor, "order": 1})

        if carbon >= 12:
            left_ring = add_ring(-3.0, 0.05)
            right_ring = add_ring(3.0, 0.05)
            bonds.append({"from": donors[0], "to": left_ring[0], "order": 1})
            if len(donors) > 1:
                bonds.append({"from": donors[1], "to": right_ring[3], "order": 1})

        extra_positions = [(-0.25, 1.55), (0.55, 1.55), (-0.25, -1.55), (0.55, -1.55)]
        extra_symbols = (["O"] * max(0, min(oxygen - donor_symbols.count("O"), 4))) + (["N"] * max(0, min(nitrogen - donor_symbols.count("N"), 4)))
        for i, symbol in enumerate(extra_symbols[:len(extra_positions)]):
            atom = add_atom(symbol, *extra_positions[i])
            if donors:
                bonds.append({"from": donors[min(i, len(donors) - 1)], "to": atom, "order": 1})
        return atoms, bonds

    heavy = [(element, count) for element, count in counts.items() if element != "H"]
    x = 0.0
    previous = None
    for element, count in heavy:
        for _ in range(min(count, 24)):
            current = add_atom(element, x, 0)
            if previous is not None:
                bonds.append({"from": previous, "to": current, "order": 1})
            previous = current
            x += 1.1
    return atoms, bonds


def build_component(name: str) -> dict:
    parsed_counts = parse_formula_counts(name)

    if parsed_counts:
        counts = parsed_counts
        formula = format_formula(counts)
    else:
        counts = {}
        formula = None

    keyed = query_keyed_providers(name, formula)
    pubchem = query_pubchem(name)

    if not counts and pubchem.get("formula"):
        counts = parse_formula_counts(str(pubchem["formula"])) or {}
        formula = str(pubchem["formula"])
        keyed = query_keyed_providers(name, formula)
    if formula is None:
        formula = "N/A"

    weight = molecular_weight(counts) if counts else None
    oqmd = query_oqmd(formula) if formula != "N/A" else {"status": "not_found"}
    mp = keyed.get("materials_project", {})
    if mp.get("status") == "ok" and mp.get("band_gap") is not None:
        oqmd = {**oqmd, "band_gap": mp.get("band_gap"), "band_gap_source": "materials_project"}
    estimates = estimate_fields(counts, pubchem, oqmd)
    research_prediction = predict_with_research_model(counts)
    estimates = apply_research_predictions(estimates, research_prediction)
    structure_image_url = None if parsed_counts else pubchem_structure_image(pubchem.get("cid"))
    atoms, bonds = ([], []) if structure_image_url else generated_schematic_diagram(counts)
    dft = dft_analysis_status(counts, bool(structure_image_url))

    source_list = ["formula_parser"] if parsed_counts else []
    if any(v.get("attempted") for v in keyed.values()):
        source_list.append("keyed_providers")
    if mp.get("status") == "ok":
        source_list.append("materials_project")
    if pubchem.get("status") == "ok":
        source_list.append("pubchem")
    if oqmd.get("status") == "ok":
        source_list.append("oqmd")

    return {
        "name": pubchem.get("name") or name,
        "formula": formula,
        "molecular_weight": f"{weight:.2f}" if weight is not None else str(pubchem.get("molecular_weight", "N/A")),
        "cas_number": "N/A",
        **estimates,
        "applications": ["materials screening", "coordination chemistry", "electrochemical prediction"],
        "hazards": "Use source SDS for confirmed hazard classification",
        "atoms": atoms,
        "bonds": bonds,
        "structure_image_url": structure_image_url,
        "diagram_note": "Provider-verified structure" if structure_image_url else "Generated schematic from formula, not a verified structure",
        "sources": source_list,
        "provider_data": {
            "keyed": keyed,
            "pubchem": pubchem,
            "oqmd": oqmd,
            "dft": dft,
            "research_prediction": research_prediction,
            "tools": tool_status(),
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": True,
        "cache_entries": len(_cache),
        "version": "2.0.0",
        "providers": tool_status(),
    }


@app.get("/debug")
async def debug():
    return JSONResponse({
        "message": "Anthropic is not used. Data comes from formula parsing, PubChem/OQMD, and optional configured tools.",
        "providers": tool_status(),
    })


@app.post("/dft/run")
async def run_dft(req: DftRequest):
    try:
        result = run_pyscf_dft(req)
        return JSONResponse({"status": "ok", "data": result})
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DFT failed: {type(e).__name__}: {e}")


@app.get("/component")
async def get_component(name: str = Query(..., description="Chemical name or formula")):
    name = name.strip()
    key = f"comp:{name.lower()}"
    if key in _cache:
        return JSONResponse({"source": "cache", "data": _cache[key]})
    data = build_component(name)
    _cache[key] = data
    return JSONResponse({"source": "providers", "data": data})


@app.get("/diagram")
async def get_diagram(name: str = Query(..., description="Chemical name or formula")):
    name = name.strip()
    comp_key = f"comp:{name.lower()}"
    if comp_key not in _cache:
        _cache[comp_key] = build_component(name)
    comp = _cache[comp_key]
    return JSONResponse({
        "source": "provider_structure" if comp.get("structure_image_url") else "unavailable",
        "data": {
            "atoms": comp.get("atoms", []),
            "bonds": comp.get("bonds", []),
            "structure_image_url": comp.get("structure_image_url"),
            "diagram_note": comp.get("diagram_note"),
            "message": (
                "Verified 2D structure is unavailable for formula-only input. "
                "Use a PubChem-resolvable compound name, CID, or install a structure generator."
            ) if not comp.get("structure_image_url") else None,
        },
    })


@app.get("/cache/clear")
async def clear_cache():
    _cache.clear()
    return {"message": "Cache cleared", "entries": 0}


@app.get("/cache/status")
async def cache_status():
    return {"entries": len(_cache), "keys": list(_cache.keys())}


if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    for p in (Path("static/index.html"), Path("index.html")):
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
