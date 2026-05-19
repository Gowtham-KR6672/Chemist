# Deployment Guide

This project is a FastAPI backend that serves `index.html` directly.

## 1. Production Files

Required files:

```text
main.py
index.html
requirements.txt
```

Optional but recommended:

```text
models/property_model.json
.env
data/material_properties.csv
```

Do not commit real API keys.

## 2. Environment Variables

Optional provider keys:

```text
MP_API_KEY=
CHEMSPIDER_API_KEY=
CITRINE_API_KEY=
```

The app still works without these keys using formula parsing, PubChem, OQMD, and local estimates.

## 3. Local Production Run

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run:

```powershell
py -m uvicorn main:app --host 0.0.0.0 --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

## 4. Render Deployment

Create a new **Web Service**.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Add environment variables in Render dashboard:

```text
MP_API_KEY
CHEMSPIDER_API_KEY
CITRINE_API_KEY
```

## 5. Railway Deployment

Create a new project from GitHub.

Railway usually detects Python automatically.

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Add provider keys in Railway Variables.

## 6. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"]
```

Build:

```bash
docker build -t quantum-chem .
```

Run:

```bash
docker run -p 8001:8001 --env-file .env quantum-chem
```

Open:

```text
http://localhost:8001/
```

## 7. ML Model Deployment

If you have a trained model, include:

```text
models/property_model.json
```

The app automatically detects it and uses trained predictions.

If the file is missing, the app falls back to formula/provider estimates.

## 8. DFT Deployment

Real DFT is optional and not recommended on basic web hosting because it can be CPU-heavy.

For real DFT, deploy on a VM or workstation with:

```text
PySCF or Quantum ESPRESSO
real 3D structures
enough CPU/RAM
```

Formula-only input cannot run real DFT.

## 9. Health Check

Use:

```text
/health
```

Example:

```text
https://your-domain.com/health
```

## 10. Notes

- Generated molecular diagrams are schematic when no verified structure is available.
- PubChem and OQMD do not need keys.
- Materials Project, ChemSpider, and Citrine are optional keyed providers.
- For research-grade ML, deploy with a validated `models/property_model.json`.

