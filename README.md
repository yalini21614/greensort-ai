# GreenSort AI — Smart Waste Management

A beginner-friendly Flask + SQLite waste-sorting starter app with user accounts, text-based waste estimates, image upload preview, QR scanning, disposal guidance, profile settings, community follows, activity history, and idempotent eco-points.

## Important capability notes
- **Image recognition:** Uploaded JPG, PNG, and WEBP images are analyzed with the local pretrained `openai/clip-vit-base-patch32` zero-shot vision model. The model downloads on first use and may take time. Predictions can be wrong; check the item and follow local municipal guidance. If the model cannot load, a typed description can be used as a clearly labelled fallback.
- **QR scanning:** Uses the browser camera and the `html5-qrcode` library loaded from a CDN. It recognizes demo values `greensort:organic`, `greensort:plastic`, `greensort:ewaste`, and `greensort:battery`; other QR codes get an explicit unsupported message. Camera access usually requires `localhost` or HTTPS and user permission.
- **Languages:** The selector saves English, Tamil, Hindi, Telugu, Malayalam, or Kannada preferences. Core interface labels are translated for English/Tamil/Hindi; the rest fall back to English. The disposal guidance is English in this starter build.
- **Social features:** Search, follow/unfollow, counts, and leaderboard use actual public user/database records. There is no fake sample ranking.
- **Eco-points:** +5 for each saved identification and +10 once per learning lesson. Unique database event keys prevent duplicate awards for the same record/lesson. User-reported actions are not independently verified.
- **Local rules:** Disposal rules vary. Follow local municipal guidelines and authorized collection instructions, especially for hazardous waste.

## Requirements
- Python 3.10+
- A modern browser
- Internet connection for the QR scanner JavaScript CDN (other core pages work locally without it)

## Setup (Windows PowerShell)
```powershell
cd greensort_ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```
Copy the generated secret into `.env` as the value of `SECRET_KEY`. Keep `.env` private.

Then run:
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser and register a new account. SQLite tables are initialized automatically in `instance/greensort.db`.

## Setup (macOS / Linux)
```bash
cd greensort_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
# Put the generated value in .env as SECRET_KEY
python app.py
```

## Features
- Registration/login/logout with Werkzeug password hashing and session-based access control.
- CSRF protection on forms through Flask-WTF.
- SQLite with parameterized queries and foreign-key constraints.
- User-specific history and history deletion.
- Upload allow-list, MIME check, and 5 MB request limit.
- Local pretrained CLIP image analysis with disposal guidance and a clearly labelled text fallback.
- Text keyword estimates and clear confidence/source labels.
- Camera QR scan for supported GreenSort demo QR payloads.
- Profile editing, public/private visibility, search, follows, points, lessons, rewards and activity history.
- Responsive UI, keyboard-friendly form controls, accessible labels, and empty states.

## Security / production notes
This is a local learning project, not a production deployment. Before deploying publicly: use a strong secret; set `COOKIE_SECURE=true` behind HTTPS; disable debug; add rate limiting and account verification; use a production WSGI server; add image malware scanning and randomized server-generated filenames; configure upload cleanup/storage; add security headers and backups; review privacy/legal requirements. For production uploads, avoid trusting the client filename and MIME header alone. The demo saves images in `static/uploads` and does not yet implement deletion of those files.


## Recent additions

- **Local AI image classification:** upload a JPG, PNG, or WEBP image to classify the likely waste type and show the existing disposal guidance. The first image analysis downloads the CLIP model and may take a while; subsequent analyses reuse it. No paid API key is required.
- **Profile logout:** the Profile page includes a POST-based logout button.

### Image-classifier setup
Install all packages with `python -m pip install -r requirements.txt`. PyTorch and Transformers are used for local image analysis; model files download from Hugging Face on first use, so an internet connection is needed for that first run. If the model cannot load, the app falls back to text-description classification when a description is supplied. Image predictions are estimates and should be checked against the real item and local disposal rules.
