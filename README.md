# JD Skill Extractor — LLM-only (Groq) & Text Sections

This build removes JSON parsing and strict validation entirely.
The LLM returns **human-readable sections** that the UI renders directly.
No schema -> no parse failures. The app never crashes.

## Quick Start
```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure env
cp .env.example .env
# edit .env and set GROQ_API_KEY
# (Optional) set LLM_MODEL (default: llama-3.3-70b-versatile)

# 4) Run
streamlit run app.py
```

## Design
- **LLM 100% text sections**; no JSON, no schema, no pydantic.
- 7 fixed sections (see below). If LLM fails, the app shows a safe fallback text block.
- Optional **lightweight badges** (non-blocking): missing evidence, possible technical "hybrid", missing years number.
  They never block rendering.

## Required Output Sections (in order)
1. Summary
2. Top Skills
3. Work Mode
4. Employment Type
5. Seniority & Years
6. Locations
7. Evidence & Warnings

Keep headers exactly, no code fences. See in-app prompt for exact guidance.
