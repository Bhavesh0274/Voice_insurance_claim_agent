"""
ClaimSaathi — Voice Insurance Claim Agent (Streamlit)
=====================================================
A deployable web app: customer speaks → ASR (Sarvam) → LLM intake (Groq)
→ TTS (Sarvam) → spoken reply, while a structured claim record fills up
and risky claims get flagged for human review.
 
Run locally:   streamlit run app.py
Deploy:        push to GitHub → share.streamlit.io  (see README)
"""
 
import os
import io
import json
import time
import base64
import requests
import streamlit as st
 
# ---------------------------------------------------------------------------
# Config & secrets
# ---------------------------------------------------------------------------
# On Streamlit Cloud, set these in the app's Secrets (see README).
# Locally, you can use a .streamlit/secrets.toml file or environment variables.
def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")
 
GROQ_API_KEY = get_secret("GROQ_API_KEY")
SARVAM_API_KEY = get_secret("SARVAM_API_KEY")
 
GROQ_MODEL = "llama-3.3-70b-versatile"
SARVAM_TTS_MODEL = "bulbul:v2"
SARVAM_VOICE = "anushka"
 
# ---------------------------------------------------------------------------
# Claim schema, mock policy DB, escalation rules
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "policy_number":     "the policy number",
    "claimant_name":     "the caller's full name",
    "incident_type":     "type of incident (motor accident / theft / fire / health / other)",
    "incident_date":     "when it happened",
    "incident_location": "where it happened",
    "description":       "a short description of what happened",
    "estimated_loss":    "rough estimated loss amount (if known)",
    "injuries_reported": "whether anyone was injured (yes/no)",
}
 
VALID_POLICIES = {
    "POL123456": {"holder": "Rahul Sharma",   "type": "motor",  "active": True},
    "POL789012": {"holder": "Anita Deshmukh",  "type": "health", "active": True},
    "POL000000": {"holder": "Expired User",    "type": "motor",  "active": False},
}
HIGH_LOSS_THRESHOLD = 500000  # ₹5 lakh
 
FIELD_LABELS = {
    "policy_number": "Policy Number", "claimant_name": "Claimant Name",
    "incident_type": "Incident Type", "incident_date": "Incident Date",
    "incident_location": "Location", "description": "Description",
    "estimated_loss": "Estimated Loss", "injuries_reported": "Injuries?",
}
 
# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes, filename="input.wav") -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set.")
    files_ = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data = {"model": "saaras:v3", "mode": "codemix", "language_code": "unknown"}
    r = requests.post("https://api.sarvam.ai/speech-to-text",
                      headers={"api-subscription-key": SARVAM_API_KEY},
                      data=data, files=files_, timeout=300)
    r.raise_for_status()
    return r.json().get("transcript", "")
 
 
INTAKE_SYSTEM = """You are ClaimSaathi, a calm, empathetic insurance claims intake agent for Indian customers reporting an incident (First Notice of Loss).
 
The caller may speak in Hindi, English, or a mix (Hinglish). Reply in the SAME language they use, kept short and spoken-friendly (1-3 sentences) since your reply is read aloud.
 
Your job: collect these claim fields, ONE question at a time, conversationally:
{fields}
 
Rules:
- Ask only for fields still missing. Do not re-ask fields already filled.
- Be warm and reassuring; the caller may be stressed.
- Never invent claim details. Only record what the caller actually says.
- If the caller asks what is left, clearly tell them which details are still needed.
- When ALL required fields are collected, set "complete": true and give a brief spoken confirmation summary.
 
You MUST respond with ONLY a valid JSON object (no markdown) in this exact shape:
{{"spoken_reply": "<what to say next, in the caller's language>",
  "claim_data": {{<only fields you could fill or update this turn>}},
  "complete": <true or false>}}"""
 
 
def llm_intake(user_text: str, claim_state: dict, history: list) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    fields_desc = "\n".join(f"- {k}: {v}" for k, v in REQUIRED_FIELDS.items())
    system = INTAKE_SYSTEM.format(fields=fields_desc)
    state_note = f"Fields collected so far: {json.dumps(claim_state, ensure_ascii=False)}"
    msgs = [{"role": "system", "content": system},
            {"role": "system", "content": state_note}]
    msgs += history
    msgs.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=msgs, temperature=0.3, max_tokens=400,
        response_format={"type": "json_object"})
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
 
 
def check_escalation(claim: dict) -> dict:
    reasons = []
    pol = claim.get("policy_number")
    if pol:
        rec = VALID_POLICIES.get(str(pol).upper().replace(" ", ""))
        if rec is None:
            reasons.append("policy number not found in records")
        elif not rec["active"]:
            reasons.append("policy is inactive/expired")
    if str(claim.get("injuries_reported", "")).strip().lower() in ("yes", "haan"):
        reasons.append("injuries reported")
    loss = claim.get("estimated_loss")
    try:
        digits = "".join(c for c in str(loss) if c.isdigit())
        if digits and int(digits) >= HIGH_LOSS_THRESHOLD:
            reasons.append(f"high estimated loss (>= ₹{HIGH_LOSS_THRESHOLD:,})")
    except Exception:
        pass
    return {"escalate": bool(reasons), "reasons": reasons}
 
 
def synthesize(text: str, language_code="hi-IN") -> bytes:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set.")
    from pydub import AudioSegment
    chunks = [text[i:i+450] for i in range(0, len(text), 450)] or [""]
    combined = None
    for ch in chunks:
        r = requests.post("https://api.sarvam.ai/text-to-speech",
                          headers={"api-subscription-key": SARVAM_API_KEY,
                                   "Content-Type": "application/json"},
                          json={"text": ch, "target_language_code": language_code,
                                "speaker": SARVAM_VOICE, "model": SARVAM_TTS_MODEL},
                          timeout=300)
        r.raise_for_status()
        seg = AudioSegment.from_file(
            io.BytesIO(base64.b64decode(r.json()["audios"][0])), format="wav")
        combined = seg if combined is None else combined + seg
    buf = io.BytesIO()
    combined.export(buf, format="wav")
    return buf.getvalue()
 
 
# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ClaimSaathi — Voice Claim Agent",
                   page_icon="🛡️", layout="wide")
 
st.markdown("""
<style>
  .main { background: #0f1117; }
  .claim-title { font-size: 2.1rem; font-weight: 800; color: #f5a623; margin-bottom: 0; }
  .claim-sub { color: #8b8f9c; font-size: 0.95rem; margin-top: 0; }
  .pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem;
          font-weight:600; margin-right:6px; }
  .pill-done { background:#143d2b; color:#5fd38d; }
  .pill-wait { background:#3a2e14; color:#f5a623; }
  .stChatMessage { background: transparent; }
</style>
""", unsafe_allow_html=True)
 
st.markdown('<p class="claim-title">🛡️ ClaimSaathi</p>', unsafe_allow_html=True)
st.markdown('<p class="claim-sub">Voice-based insurance claim intake (FNOL) · '
            'speaks Hinglish & Indic languages · ASR → LLM → TTS</p>',
            unsafe_allow_html=True)
 
# Session state
if "claim_state" not in st.session_state:
    st.session_state.claim_state = {k: None for k in REQUIRED_FIELDS}
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.completed = False
 
# Sidebar: status + controls
with st.sidebar:
    st.subheader("⚙️ Setup")
    ok_groq = "✅" if GROQ_API_KEY else "❌"
    ok_sarvam = "✅" if SARVAM_API_KEY else "❌"
    st.write(f"{ok_groq} Groq key  ·  {ok_sarvam} Sarvam key")
    if not (GROQ_API_KEY and SARVAM_API_KEY):
        st.warning("Add GROQ_API_KEY and SARVAM_API_KEY in app Secrets to enable.")
 
    reply_language = st.selectbox(
        "Reply / voice language",
        ["hi-IN", "en-IN", "mr-IN", "bn-IN", "ta-IN", "te-IN",
         "kn-IN", "gu-IN", "ml-IN", "pa-IN"], index=0)
 
    st.divider()
    st.subheader("📊 Claim progress")
    cs = st.session_state.claim_state
    filled = sum(1 for v in cs.values() if v not in (None, "", "null"))
    st.progress(filled / len(cs), text=f"{filled}/{len(cs)} details collected")
    for k in REQUIRED_FIELDS:
        v = cs[k]
        if v not in (None, "", "null"):
            st.markdown(f'<span class="pill pill-done">✓ {FIELD_LABELS[k]}</span> {v}',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-wait">… {FIELD_LABELS[k]}</span>',
                        unsafe_allow_html=True)
 
    esc = check_escalation(cs)
    if esc["escalate"]:
        st.error("🚩 Flagged for human adjuster:\n\n- " + "\n- ".join(esc["reasons"]))
 
    st.divider()
    if st.button("🔄 Start new claim"):
        for key in ["claim_state", "history", "messages", "completed", "last_sig"]:
            st.session_state.pop(key, None)
        # Advance the recorder key so a fresh, empty recorder is shown.
        st.session_state.turn = st.session_state.get("turn", 0) + 1
        st.rerun()
 
# Greeting (first load)
if not st.session_state.messages:
    greeting = ("Namaste! Main ClaimSaathi hoon. Main aapka claim file karne mein "
                "madad karungi. Kya aap mujhe apna policy number bata sakte hain?")
    st.session_state.messages.append({"role": "assistant", "text": greeting})
    st.session_state.history.append({"role": "assistant", "content": greeting})
 
# Render conversation
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["text"])
        if m.get("audio"):
            st.audio(m["audio"], format="audio/wav", autoplay=m.get("autoplay", False))
 
# Audio input (records in the browser; no extra libs)
# A turn counter; changing the widget key forces a fresh, empty recorder each turn.
if "turn" not in st.session_state:
    st.session_state.turn = 0
 
st.write("🎤 **Tap to record your answer, then stop:**")
audio_value = st.audio_input("Record", label_visibility="collapsed",
                             key=f"rec_{st.session_state.turn}",
                             disabled=not (GROQ_API_KEY and SARVAM_API_KEY))
 
# Process a new recording
if audio_value is not None and not st.session_state.completed:
    audio_bytes = audio_value.read()
    # Avoid reprocessing the same clip on reruns
    sig = hash(audio_bytes)
    if st.session_state.get("last_sig") != sig:
        st.session_state.last_sig = sig
        try:
            with st.spinner("Transcribing…"):
                user_text = transcribe(audio_bytes)
            st.session_state.messages.append({"role": "user", "text": user_text})
            st.session_state.history.append({"role": "user", "content": user_text})
 
            with st.spinner("Thinking…"):
                result = llm_intake(user_text, st.session_state.claim_state,
                                    st.session_state.history)
            for k, v in (result.get("claim_data") or {}).items():
                if k in st.session_state.claim_state and v not in (None, "", "null"):
                    st.session_state.claim_state[k] = v
 
            reply = result.get("spoken_reply", "")
            missing = [k for k, v in st.session_state.claim_state.items()
                       if v in (None, "", "null")]
            complete = bool(result.get("complete")) and not missing  # safety net
 
            with st.spinner("Generating voice…"):
                reply_audio = synthesize(reply, reply_language)
 
            st.session_state.history.append({"role": "assistant", "content": reply})
            st.session_state.messages.append({"role": "assistant", "text": reply,
                                              "audio": reply_audio, "autoplay": True})
            if complete:
                st.session_state.completed = True
            st.session_state.turn += 1      # fresh recorder next turn
            st.rerun()
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
 
# Completion: show + download final claim
if st.session_state.completed:
    st.success("✅ All details collected. Claim ready for processing.")
    cs = st.session_state.claim_state
    esc = check_escalation(cs)
    claim_id = "CLM" + str(int(time.time()))[-6:]
    final_claim = {
        "claim_id": claim_id,
        "status": "needs_human_review" if esc["escalate"] else "auto_intake_complete",
        "escalation": esc,
        "fields": cs,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.json(final_claim)
    st.download_button("⬇️ Download claim (JSON)",
                       data=json.dumps(final_claim, indent=2, ensure_ascii=False),
                       file_name=f"{claim_id}.json", mime="application/json")
