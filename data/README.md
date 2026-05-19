# Research-Grade ML Dataset

Put curated training data in:

```text
data/material_properties.csv
```

Required columns for full dashboard prediction:

```csv
formula,specific_capacitance_f_g,energy_density_wh_kg,band_gap_ev,cycling_stability_pct,dipole_moment_debye,logp,polar_surface_area,redox_potential_v,solvation_energy_kcal_mol,mulliken_charge
```

Optional columns:

```csv
source,doi,method,notes
```

Use experimentally measured or validated DFT values only. Do not mix unverified AI-generated values into this dataset.
