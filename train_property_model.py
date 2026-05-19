"""
Train a simple reproducible materials-property model.

Input:
  data/material_properties.csv

Output:
  models/property_model.json

This uses transparent formula descriptors and ridge regression so the model is
auditable. Replace or extend with Matminer/DeepChem descriptors when you have a
larger curated dataset.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from main import parse_formula_counts

DATA_PATH = Path("data/material_properties.csv")
MODEL_PATH = Path("models/property_model.json")

TARGETS = [
    "specific_capacitance_f_g",
    "energy_density_wh_kg",
    "band_gap_ev",
    "cycling_stability_pct",
    "dipole_moment_debye",
    "logp",
    "polar_surface_area",
    "redox_potential_v",
    "solvation_energy_kcal_mol",
    "mulliken_charge",
]

FEATURES = [
    "bias",
    "heavy_atoms",
    "metal_atoms",
    "carbon",
    "nitrogen",
    "oxygen",
    "sulfur",
    "hetero_ratio",
    "metal_ratio",
    "molecular_size",
]

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv",
}


def formula_features(formula: str) -> list[float]:
    counts = parse_formula_counts(formula)
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")
    heavy = sum(v for k, v in counts.items() if k != "H") or 1
    metals = sum(counts.get(k, 0) for k in METALS)
    hetero = sum(counts.get(k, 0) for k in ("N", "O", "S", "P", "F", "Cl", "Br", "I"))
    return [
        1.0,
        float(heavy),
        float(metals),
        float(counts.get("C", 0)),
        float(counts.get("N", 0)),
        float(counts.get("O", 0)),
        float(counts.get("S", 0)),
        hetero / heavy,
        metals / heavy,
        math.log1p(heavy),
    ]


def solve_ridge(xs: list[list[float]], ys: list[float], alpha: float = 1.0) -> list[float]:
    n = len(xs[0])
    a = [[0.0 for _ in range(n)] for _ in range(n)]
    b = [0.0 for _ in range(n)]

    for x, y in zip(xs, ys):
        for i in range(n):
            b[i] += x[i] * y
            for j in range(n):
                a[i][j] += x[i] * x[j]
    for i in range(n):
        a[i][i] += alpha

    return gaussian_solve(a, b)


def gaussian_solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        div = a[i][i]
        if abs(div) < 1e-12:
            raise ValueError("Training matrix is singular; add more diverse data.")
        for j in range(i, n):
            a[i][j] /= div
        b[i] /= div
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            for j in range(i, n):
                a[r][j] -= factor * a[i][j]
            b[r] -= factor * b[i]
    return b


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Create it using data/README.md.")

    rows = list(csv.DictReader(DATA_PATH.open(newline="", encoding="utf-8")))
    if len(rows) < 20:
        raise SystemExit("Need at least 20 curated rows before training a research model.")

    xs = [formula_features(row["formula"]) for row in rows]
    coefficients = {}
    metrics = {}
    for target in TARGETS:
        ys = [float(row[target]) for row in rows if row.get(target)]
        if len(ys) != len(xs):
            raise SystemExit(f"Target {target} is missing values in some rows.")
        coef = solve_ridge([x[:] for x in xs], ys)
        coefficients[target] = coef
        preds = [sum(c * v for c, v in zip(coef, x)) for x in xs]
        mae = sum(abs(p - y) for p, y in zip(preds, ys)) / len(ys)
        metrics[target] = {"training_mae": mae}

    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "model_type": "ridge_regression_formula_descriptors",
        "features": FEATURES,
        "targets": TARGETS,
        "n_training_rows": len(rows),
        "coefficients": coefficients,
        "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
