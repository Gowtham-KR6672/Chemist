# Real DFT Setup

Formula-only input cannot run DFT. Real DFT requires 3D atomic coordinates.

Supported app endpoint:

```text
POST /dft/run
```

Input JSON:

```json
{
  "xyz": "3\nwater\nO 0 0 0\nH 0 0.757 0.586\nH 0 -0.757 0.586",
  "charge": 0,
  "spin": 0,
  "basis": "def2-svp",
  "xc": "pbe0"
}
```

Output includes:

```text
total_energy_hartree
HOMO-LUMO gap in eV
dipole moment in Debye
SCF convergence status
```

## Engine

The app uses PySCF for molecular DFT when installed.

On this Windows machine, direct `pip install pyscf` failed because no C/C++ build toolchain was available. Practical options:

1. Install PySCF from a prebuilt compatible environment, usually conda-forge:

```powershell
conda install -c conda-forge pyscf
```

2. Use WSL/Linux, then:

```bash
python -m pip install pyscf
```

3. Install Visual Studio Build Tools and retry:

```powershell
py -m pip install pyscf
```

For periodic solid-state DFT, install Quantum ESPRESSO separately and make sure `pw.x` is on PATH.

