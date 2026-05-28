# 🛡️ ClaimSaathi — Voice Insurance Claim Agent

A deployable web app that takes an insurance **First Notice of Loss (FNOL)** entirely by **voice**, in Hinglish / Indic languages.

**Pipeline:** 🎤 customer speaks → **ASR** (Sarvam `saaras:v3`) → **LLM intake** (Groq Llama 3.3 70B) → **TTS** (Sarvam `bulbul:v2`) → 🔊 spoken reply, while a structured JSON claim record fills up and risky claims are flagged for a human adjuster.

---

## What it does
- Records the customer's voice in the browser (no install for the user).
- Transcribes code-switched Hindi-English speech.
- Runs a **slot-filling conversation** — asks for one missing claim detail at a time.
- Shows a **live claim record** and **progress bar** in the sidebar.
- **Flags for human review** on unknown/expired policy, reported injuries, or high loss.
- Exports the finished claim as a downloadable **JSON** file.

---

## Run locally

```bash
pip install -r requirements.txt
# install ffmpeg (pydub needs it):
#   Ubuntu/Debian:  sudo apt install ffmpeg
#   macOS:          brew install ffmpeg
#   Windows:        choco install ffmpeg

# add your keys to .streamlit/secrets.toml  (template provided)
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501).

---

## Deploy free on Streamlit Community Cloud

1. **Push this folder to a GitHub repo** (public or private).
   Make sure `app.py`, `requirements.txt`, and `packages.txt` are in the repo root.
   > `packages.txt` installs **ffmpeg** on the server — required for audio.

2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app**.

3. Pick your repo, branch, and `app.py` as the main file.

4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   SARVAM_API_KEY = "your_sarvam_key"
   ```

5. Click **Deploy**. First build takes a few minutes (it installs ffmpeg + Python deps).
   You'll get a public URL like `https://your-app.streamlit.app` — put this on your CV/GitHub.

> ⚠️ **Never commit real keys.** `.gitignore` already excludes `secrets.toml`.
> Keys live only in Streamlit Cloud's Secrets panel.

---

## Get API keys
- **Groq** (LLM) → https://console.groq.com  (generous free tier)
- **Sarvam** (ASR + TTS, Indic) → https://dashboard.sarvam.ai

---

## Demo script (for showing it off)
1. *"My policy number is POL123456"* → fills policy, validates against records.
2. Walk through name, incident type, date, location, description, loss.
3. To show **escalation**: say someone was injured, or give an unknown policy
   number, or a loss over ₹5 lakh — the sidebar flags it red.
4. Finish → download the JSON claim record.

---

## Notes & production roadmap
- **Voices:** uses `bulbul:v2` / `anushka`. Sarvam's newer `bulbul:v3` has 30+ voices
  and a 2500-char limit (less chunking) — change `SARVAM_TTS_MODEL`/`SARVAM_VOICE` in `app.py`.
- **LLM:** `llama-3.3-70b-versatile`; switch to `llama-3.1-8b-instant` for lower latency.
- **Mock policy DB:** `VALID_POLICIES` in `app.py` is a stand-in for a real
  policy-admin API — swap it for a live lookup in production.
- **Next steps:** streaming ASR/TTS over WebSockets for real-time response,
  telephony (Twilio/Exotel) for real phone calls, and a downstream fraud-scoring model.
