# JD Skill Extractor — LLM-only (Groq) with Confidence (no badges)
# URL field removed; added Job Title & Work location inputs.
# Run: streamlit run app.py

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_client():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in .env")
    return Groq(api_key=key)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# -------- Prompt (with Confidence rubric; NO badges) --------
SYSTEM_PROMPT = """
You are a recruiter-facing assistant. 
Return the following seven sections in this exact order. 
No extra prose and no code fences. 
Keep it concise and skimmable. 
Use short bullets where applicable. 

If something is not clearly stated as a policy (e.g., work location type or employment type), set it to Unspecified, give a short rationale, and note ambiguity under Warnings.

AUTHORITATIVE USER CONTEXT (OVERRIDES JD ON CONFLICT):
- If Job Title or Work location is provided by the user, TREAT THESE AS SOURCE OF TRUTH and prefer them over the JD when there is any conflict.
- Use the JD to supplement or fill gaps only when the user context is missing or silent.
- When you override a JD value with user-provided context, add a short note under “Advisories” like:
  “User-provided override applied — Job Title/Work location.”
- If the JD clearly contradicts the user context, still keep the user context as final, but mention the JD alternative in “Advisories”.

Allowed seniority labels (use exactly one): Entry, Mid, Senior, Lead, Manager, Director, Executive, Unspecified.
If the JD mentions Staff or Principal and those labels are not in the allowed set, choose the nearest fit (typically Lead for IC leadership, Director for organizational leadership) and note the mapping under “Advisories”.

Confidence rubric (use exactly High / Medium / Low): 
- High: explicit tokens or clear, repeated policy (e.g., “Full-time”, “Hybrid: 2–3 days/week”, city/state listed, “5+ years”). 
- Medium: reasonable inference but not explicitly stated (e.g., benefits imply full-time; nationwide clues; years implied by seniority). 
- Low: ambiguous, conflicting, or only indirect hints (e.g., “hybrid” appears only in technical context; no numeric years; “lead” used as a verb).

1. Summary (Each item on its own line)
Summary MUST MIRROR the final values decided in sections 2 to 7 (do NOT introduce new judgments here).
Compose sections 2–7 first (internally), then WRITE OR REWRITE the Summary so it copies those final values exactly.
If any Summary value differs from its source section (2 to 7), REWRITE the Summary to match exactly.
If a source section is Unspecified, the Summary must also show Unspecified for that field.

- Job Title : <User-provided Job Title or Unspecified>
- Seniority : <value from section 2> 
- Years required : <value from section 2, normalize to forms like “5+” when possible> 
- Employment type : <first line from section 3> 
- Salary : <first line from section 4> 
- Work Location(s) : <primary item from section 5> 
- Work Location Type : <first line from section 6> 
- Top required skills : <first 3 skills from section 7, comma-separated> 

2. Seniority & Years (Each item on its own line) 
- Seniority: one of Entry / Mid / Senior / Lead / Manager / Director / Executive / Unspecified.
- One short sentence explaining why on seniority.
- Years required: <number or Unspecified, normalize to compact forms like “5+” when possible>.
- One short sentence explaining why on years required.
- Confidence: <High|Medium|Low> 

3. Employment Type (Each item on its own line) 
- One of: Full-time / Contract / Part-time / Internship / Unspecified. 
- One short sentence explaining why. 
- Confidence: <High|Medium|Low> 

4. Salary (Each item on its own line) 
- If stated, give a number or range (e.g., "$80,000–$120,000" or "$25/hr"). If not stated, say "Unspecified". 
- Confidence: <High|Medium|Low> 

5. Work Locations (Each item on its own line) 
- If nationwide or multi-location, use "United States (nationwide)" or similar. 
- Confidence: <High|Medium|Low> 

6. Work Location Type (Each item on its own line) 
- One of: Remote / Hybrid / Onsite / Unspecified. 
- One short sentence explaining why. 
- Confidence: <High|Medium|Low> 

7. Top Skills (Each item on its own line) 
- 3–6 bullets. No "High/Medium/Low" qualifiers. 

Advisories (Each item on its own line) 
- Brief bullets for any ambiguities or disambiguations (e.g., 'lead' used as a verb; 'hybrid' used in technical context rather than a work policy; conflicting signals). 

Formatting rules: 
- Keep exactly these section headers and order. 
- Use plain text (no markdown fences). 
- For each of sections 2–6, include a final line "Confidence: High|Medium|Low" exactly.
- Self-check for precedence: If a user-provided Work location exists, ensure the Summary “Work Location(s)” equals that value, and list the JD’s different location (if any) only under “Advisories”. If a user-provided Job Title exists, infer seniority from it when ambiguous, and treat it as definitive over the JD.

Consistency checklist (must pass before sending):
- Summary.Seniority == Section 2 Seniority
- Summary.Years required == Section 2 Years required
- Summary.Employment type == first line of Section 3
- Summary.Salary == first line of Section 4
- Summary.Work Location(s) == primary item in Section 5
- Summary.Work Location Type == first line of Section 6
- Summary.Top required skills == first 3 items in Section 7 (comma-separated)
If any check fails, fix the Summary and re-verify.
"""

# This user instruction now includes user-provided context (job title & work location) when present.
USER_INSTR_PREFIX = """
Use the JD text below. Prefer direct quotes for evidence. Keep everything crisp.

AUTHORITATIVE USER CONTEXT (OVERRIDES JD ON CONFLICT):
- If Job Title or Work location is provided by the user, TREAT THESE AS SOURCE OF TRUTH and prefer them over the JD when there is any conflict.
- Use the JD to supplement or fill gaps only when the user context is missing or silent.
- When you override a JD value with the user-provided context, add a short note under “Advisories” like:
  “User-provided override applied — Job Title/Work location.”
- If the JD clearly contradicts the user context, still keep the user context as final, but mention the JD alternative in “Advisories”.

USER-PROVIDED CONTEXT:
- Job Title: <enter Job title or Unspecified>
- Work location: <enter Work location or Unspecified>

JD TEXT (verbatim):
<paste the full JD here>
"""

USER_INSTR_JD_HEADER = """
JD TEXT (verbatim):
"""

def call_llm(full_user_message: str, model: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
    client = get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_message},
        ],
    )
    return completion.choices[0].message.content.strip()

def ping_groq(model: str = None):
    """Minimal health check using a tiny prompt (doesn't depend on the big system prompt)."""
    model = model or DEFAULT_MODEL
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_completion_tokens=10,
        messages=[
            {"role": "system", "content": "Reply with exactly: PONG"},
            {"role": "user", "content": "ping"},
        ],
    )
    return resp.choices[0].message.content.strip()

def safe_fallback_block() -> str:
    # Always-renderable fallback with confidence lines present
    return """Summary
Seniority: Unspecified · Years required: Unspecified · Employment type: Unspecified · Salary: Unspecified · Location(s): Unspecified · Work location type: Unspecified · Top required skills: (add top 3)

Seniority & Years
Seniority: Unspecified / Years required: Unspecified
Confidence: Low

Employment Type
Unspecified
Rationale: Not explicitly stated.
Confidence: Low

Salary
Unspecified
Confidence: Low

Locations
- Unspecified
Confidence: Low

Work Location Type
Unspecified
Rationale: Not explicitly stated.
Confidence: Low

Top Skills
- (add bullets)

Advisories
- Could not obtain a reliable model output at this time.
"""

# ---------------- UI ----------------

st.set_page_config(page_title="JD Summary & Skills Extractor", page_icon="🧩", layout="wide")
st.title("JD Summary & Skills Extractor 🧩")
st.caption("Powered by Groq LLMs")

# Ping Groq ABOVE the inputs
top_l, top_r = st.columns([1, 3])
with top_l:
    ping = st.button("Ping Groq", use_container_width=True)

if ping:
    try:
        pong = ping_groq(DEFAULT_MODEL)
        if pong.strip() == "PONG":
            st.success("Groq OK — minimal completion succeeded.")
        else:
            st.warning(f"Groq responded, but not 'PONG': {pong[:80]!r}")
    except Exception as e:
        st.error(f"Groq FAILED — {e}")

# Inputs (Job Title & Work location, JD text)
c1, c2 = st.columns([1, 1])
with c1:
    job_title = st.text_input("Job Title")
with c2:
    work_location = st.text_input("Work location")

jd_text = st.text_area("Paste JD text", height=320, placeholder="Paste job description here…")

with st.expander("Model & Settings"):
    model = st.text_input("Groq model", value=DEFAULT_MODEL, help="e.g., llama-3.3-70b-versatile or llama-3.1-8b-instant")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max completion tokens", 256, 4096, 1200, 64)

run = st.button("Generate", type="primary")

if run:
    if not jd_text.strip():
        st.warning("Please paste a Job Description.")
        st.stop()

    # Build user message with user-provided context (if any) + JD text
    context_lines = []
    context_lines.append(f"- Job Title: {job_title.strip() or 'Unspecified'}")
    context_lines.append(f"- Work location: {work_location.strip() or 'Unspecified'}")
    context_block = "\n".join(context_lines)

    full_user_message = USER_INSTR_PREFIX + context_block + USER_INSTR_JD_HEADER + jd_text

    try:
        out = call_llm(full_user_message, model=model, temperature=temperature, max_tokens=max_tokens)
        if not out or len(out.strip()) < 20:
            raise RuntimeError("Empty or too short model output")

        st.markdown("### Result")
        st.markdown(out)

        st.markdown("### Quick Edit")
        edited = st.text_area("Tweak the summary text (optional)", value=out, height=260)
        st.download_button("Download as .txt", data=edited.encode("utf-8"), file_name="jd_summary.txt")

    except Exception as e:
        st.error(f"LLM call failed: {e}")
        fb = safe_fallback_block()
        st.markdown("### Fallback")
        st.markdown(fb)
        st.download_button("Download fallback .txt", data=fb.encode("utf-8"), file_name="jd_summary_fallback.txt")

