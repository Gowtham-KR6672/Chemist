# Vercel Deployment Guide

This setup deploys both:

```text
Frontend: index.html
Backend/API: FastAPI main.py
```

Vercel routes all requests to the FastAPI app through:

```text
api/index.py
```

## Files Added

```text
vercel.json
api/index.py
```

## 1. Install Vercel CLI

```powershell
npm install -g vercel
```

Login:

```powershell
vercel login
```

## 2. Deploy

From the project folder:

```powershell
cd "d:\Day Shift\Thaya\Code\Siva sir\files"
vercel
```

For production:

```powershell
vercel --prod
```

## 3. Vercel Project Settings

Framework preset:

```text
Other
```

Build command:

```text
None
```

Install command:

```text
pip install -r requirements.txt
```

Output directory:

```text
None
```

## 4. Environment Variables

In Vercel dashboard, add optional keys:

```text
MP_API_KEY
CHEMSPIDER_API_KEY
CITRINE_API_KEY
```

The app still works without these using:

```text
formula parser
PubChem
OQMD
local estimates
generated diagrams
```

## 5. URLs

After deployment:

```text
https://your-project.vercel.app/
https://your-project.vercel.app/health
https://your-project.vercel.app/docs
```

## 6. Important Vercel Limits

Vercel serverless functions are not suitable for heavy real DFT jobs.

These are OK on Vercel:

```text
formula parsing
PubChem/OQMD API calls
Materials Project API calls
trained model JSON predictions
schematic diagrams
dashboard rendering
```

These are not recommended on Vercel:

```text
PySCF DFT calculations
Quantum ESPRESSO calculations
large model training
long CPU-heavy jobs
```

For real DFT, use a separate VM/workstation API and call it from this Vercel app.

