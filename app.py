import os, re, sqlite3, secrets, uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, abort
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
INSTANCE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, instance_path=str(INSTANCE_DIR))
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    UPLOAD_FOLDER=str(UPLOAD_DIR),
)
csrf = CSRFProtect(app)
DB_PATH = INSTANCE_DIR / "greensort.db"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Local zero-shot image classifier. The model downloads once on first image use.
# No paid API key is required.
IMAGE_MODEL_ID = os.getenv("GREENSORT_IMAGE_MODEL", "openai/clip-vit-base-patch32")
_image_classifier = None
_image_classifier_error = None
IMAGE_LABELS = {
    "organic": "a photo of wet organic food waste, fruit peels, vegetables, or food scraps",
    "plastic": "a photo of plastic waste such as a plastic bottle, bag, wrapper, or container",
    "paper": "a photo of paper or cardboard waste such as newspaper, paper, or a cardboard box",
    "glass": "a photo of glass waste such as a glass bottle or jar",
    "metal": "a photo of metal waste such as an aluminium can or steel tin",
    "ewaste": "a photo of electronic waste such as a phone, charger, keyboard, or computer",
    "battery": "a photo of a loose household, lithium, or rechargeable battery waste",
    "chemical": "a photo of hazardous chemical or medicine waste such as paint, pesticide, or expired medicine",
    "sanitary": "a photo of sanitary waste such as a diaper or sanitary pad",
    "general": "a photo of mixed general non-recyclable household waste",
}

def classify_image(image_path):
    """Return (waste_key, confidence) from image pixels using a local CLIP model."""
    global _image_classifier, _image_classifier_error
    if _image_classifier_error:
        raise RuntimeError(_image_classifier_error)
    if _image_classifier is None:
        try:
            from transformers import pipeline
            _image_classifier = pipeline("zero-shot-image-classification", model=IMAGE_MODEL_ID)
        except Exception as exc:
            _image_classifier_error = str(exc)
            raise RuntimeError(_image_classifier_error) from exc
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        predictions = _image_classifier(img, candidate_labels=list(IMAGE_LABELS.values()))
    if not predictions:
        raise RuntimeError("The image model returned no predictions.")
    best = predictions[0]
    reverse_labels = {label: key for key, label in IMAGE_LABELS.items()}
    return reverse_labels[best["label"]], float(best["score"])

# Practical starter guidance; local municipal rules always take priority.
WASTE = {
    "organic": {"name":"Wet / organic waste","category":"Wet or organic waste","bin":"Green / wet-waste bin","recycle":"Usually compostable","prep":"Keep food scraps separate from dry recyclables. Drain excess liquid; compost where accepted.","safety":"Do not mix batteries, chemicals, or e-waste into compost.","impact":"Composting can reduce landfill methane and return nutrients to soil.","reuse":"Compost fruit and vegetable scraps if your local program accepts them.","examples":"Fruit peels, vegetable scraps, tea leaves, food leftovers","tips":"Avoid plastic bags in home compost unless specifically certified and accepted."},
    "plastic": {"name":"Plastic item","category":"Plastic waste","bin":"Dry-waste / recyclable bin only if locally accepted","recycle":"Depends on resin type and local collection","prep":"Empty and rinse; keep dry. Check local recycling symbols and collection rules.","safety":"Never burn plastic; fumes can be harmful.","impact":"Correct sorting can keep recoverable plastic out of drains and landfills.","reuse":"Reuse clean containers when safe and suitable.","examples":"Plastic bottles, tubs, packaging","tips":"Avoid assuming every plastic wrapper is recyclable."},
    "paper": {"name":"Paper or cardboard","category":"Paper and cardboard","bin":"Dry recyclable bin","recycle":"Often recyclable if clean and dry","prep":"Keep dry; flatten boxes; remove food-contaminated sections.","safety":"Keep away from flames and wet waste.","impact":"Recycling clean paper can reduce demand for virgin materials.","reuse":"Reuse boxes, paper bags, or one-sided sheets.","examples":"Cardboard boxes, newspapers, office paper","tips":"Greasy or heavily soiled paper may not be accepted."},
    "glass": {"name":"Glass item","category":"Glass","bin":"Glass collection point or dry-waste stream as locally directed","recycle":"Often recyclable","prep":"Empty and rinse; keep intact glass separate where required.","safety":"Wrap broken glass securely, label it, and follow local instructions; never place loose shards in a bag.","impact":"Glass can often be recycled repeatedly where collection infrastructure exists.","reuse":"Reuse intact jars after cleaning.","examples":"Glass jars, bottles","tips":"Do not mix ceramics, mirrors, bulbs, or window glass with container glass unless accepted."},
    "metal": {"name":"Metal item","category":"Metal","bin":"Metal recycling / dry-waste collection","recycle":"Often recyclable","prep":"Empty containers and rinse when appropriate.","safety":"Cover sharp edges; do not handle unknown pressurized or chemical containers.","impact":"Metal recovery can save raw materials and energy.","reuse":"Reuse clean tins for storage only when safe.","examples":"Aluminium cans, steel tins","tips":"Aerosol cans and contaminated containers may need special handling."},
    "ewaste": {"name":"Electronic waste","category":"Electronic waste","bin":"Authorized e-waste collection or take-back point","recycle":"Requires specialist collection","prep":"Back up and erase personal data from devices; remove batteries only if safely designed to do so.","safety":"Do not dismantle, crush, burn, or place electronics in household bins.","impact":"Proper recovery helps prevent toxic materials from contaminating soil and water.","reuse":"Repair, donate, or trade in working devices through trusted channels.","examples":"Phones, chargers, keyboards, laptops","tips":"Use authorized collection channels and retain proof of handover if available."},
    "battery": {"name":"Battery","category":"Hazardous waste","bin":"Battery take-back point or authorized hazardous-waste collection","recycle":"Special collection required","prep":"Keep terminals from touching metal; store cool and dry; follow collection-point instructions.","safety":"Never burn, puncture, crush, or put batteries in ordinary household bins. If swollen or leaking, avoid handling and seek local hazardous-waste advice.","impact":"Safe collection reduces fire and chemical contamination risks.","reuse":"Do not reuse damaged batteries; use rechargeable batteries where suitable.","examples":"AA cells, button cells, laptop or phone batteries","tips":"Lithium batteries can cause fires if damaged or short-circuited."},
    "chemical": {"name":"Chemical / medicine","category":"Hazardous waste","bin":"Authorized hazardous-waste or medicine take-back service","recycle":"Do not put in ordinary recycling","prep":"Keep in original labelled container where safe; follow local collection advice.","safety":"Do not pour chemicals or medicines into drains or mix chemicals. Keep away from children and pets.","impact":"Correct collection helps protect waterways and sanitation workers.","reuse":"Do not reuse chemical containers for food or drinking water.","examples":"Paint, solvents, pesticides, expired medicines","tips":"Pharmacies or municipal programs may accept some medicines; rules vary."},
    "sanitary": {"name":"Sanitary waste","category":"General / sanitary waste","bin":"Wrapped sanitary-waste or residual-waste stream as locally directed","recycle":"Usually not recyclable in household recycling","prep":"Wrap securely and follow local segregation rules.","safety":"Do not flush sanitary products. Use gloves if handling waste from another person.","impact":"Proper containment protects sanitation workers and public spaces.","reuse":"Choose reusable alternatives if appropriate and safely maintain them.","examples":"Sanitary pads, diapers, tissues","tips":"Never place sanitary waste loose in dry recyclables."},
    "general": {"name":"General waste","category":"General or non-recyclable waste","bin":"Residual / general-waste bin","recycle":"Usually not recyclable through standard household streams","prep":"Separate any recyclable or hazardous components first.","safety":"If the item is sharp, leaking, chemical, or unknown, seek local advice before disposal.","impact":"Reducing residual waste lowers pressure on landfill and collection systems.","reuse":"Consider repair or reuse before discarding.","examples":"Mixed-material wrappers, contaminated items","tips":"Local rules determine the correct bin and collection schedule."},
}
KEYWORDS = {
    "organic": ["banana peel","fruit peel","vegetable","food waste","food scrap","tea leaves","leftover","organic","eggshell"],
    "plastic": ["plastic bottle","plastic bag","plastic wrapper","plastic container","polythene","plastic"],
    "paper": ["paper","cardboard","newspaper","carton","book","paper box"],
    "glass": ["glass jar","glass bottle","broken glass","glass"],
    "metal": ["aluminium can","aluminum can","metal can","tin can","metal","steel"],
    "ewaste": ["phone","mobile","charger","laptop","keyboard","computer","electronic","earphone","cable","e-waste","ewaste"],
    "battery": ["battery","cell battery","lithium battery","power bank"],
    "chemical": ["chemical","paint","pesticide","medicine","tablet","syringe","solvent","cleaner","expired drug"],
    "sanitary": ["diaper","sanitary pad","tissue","nappy","sanitary waste"],
}
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL,
          email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
          language TEXT NOT NULL DEFAULT 'en', bio TEXT NOT NULL DEFAULT '',
          is_public INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS history (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
          description TEXT NOT NULL, category TEXT NOT NULL, waste_key TEXT NOT NULL,
          disposal TEXT NOT NULL, confidence REAL, source TEXT NOT NULL,
          created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS follows (
          follower_id INTEGER NOT NULL, followed_id INTEGER NOT NULL,
          created_at TEXT NOT NULL, PRIMARY KEY(follower_id, followed_id),
          CHECK(follower_id != followed_id),
          FOREIGN KEY(follower_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY(followed_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS point_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
          event_key TEXT NOT NULL, points INTEGER NOT NULL, note TEXT NOT NULL,
          created_at TEXT NOT NULL, UNIQUE(user_id, event_key),
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
init_db()

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # A session can outlive a deleted/reset database user. Validate it before
        # protected routes access the user's database row.
        if not session.get("user_id") or current_user() is None:
            session.clear()
            flash("Your session expired. Please log in again.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped

def current_user():
    if not session.get("user_id"): return None
    with db() as c:
        return c.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()

def classify_text(text):
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized: return None, 0.0
    for key, terms in KEYWORDS.items():
        if any(term in normalized for term in terms):
            return key, 0.78
    return "general", 0.35

def award_once(user_id, event_key, points, note):
    with db() as c:
        c.execute("INSERT OR IGNORE INTO point_events(user_id,event_key,points,note,created_at) VALUES(?,?,?,?,?)",
                  (user_id, event_key, points, note, datetime.now().isoformat(timespec="seconds")))
        return c.execute("SELECT changes()").fetchone()[0] == 1

def points_total(user_id):
    with db() as c:
        return c.execute("SELECT COALESCE(SUM(points),0) FROM point_events WHERE user_id=?", (user_id,)).fetchone()[0]

@app.context_processor
def common():
    u = current_user()
    return {"current_user": u, "point_total": points_total(u["id"]) if u else 0,
            "lang": (u["language"] if u else session.get("language","en"))}

@app.route("/")
def home():
    u = current_user()
    # Recover safely if the browser still has a session for a user that no
    # longer exists in the current SQLite database.
    if u is None:
        session.clear()
        flash("Please log in again to continue.", "warning")
        return redirect(url_for("login"))
    with db() as c:
        count = c.execute("SELECT COUNT(*) FROM history WHERE user_id=?", (u["id"],)).fetchone()[0]
        recent = c.execute("SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 4", (u["id"],)).fetchall()
        followers = c.execute("SELECT COUNT(*) FROM follows WHERE followed_id=?", (u["id"],)).fetchone()[0]
        following = c.execute("SELECT COUNT(*) FROM follows WHERE follower_id=?", (u["id"],)).fetchone()[0]
    return render_template("index.html", count=count, recent=recent, followers=followers, following=following, guide=WASTE)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("full_name","").strip()
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        confirm = request.form.get("confirm_password","")
        if len(name) < 2 or len(name) > 80 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid name and email address.", "error")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            try:
                with db() as c:
                    c.execute("INSERT INTO users(full_name,email,password_hash,created_at) VALUES(?,?,?,?)",
                              (name,email,generate_password_hash(password),datetime.now().isoformat(timespec="seconds")))
                    uid = c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()["id"]
                session.clear(); session["user_id"] = uid
                flash("Welcome to GreenSort AI! Your account is ready.", "success")
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email=request.form.get("email","").strip().lower()
        password=request.form.get("password","")
        with db() as c:
            u=c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if u and check_password_hash(u["password_hash"], password):
            session.clear(); session["user_id"]=u["id"]
            flash("You're signed in.", "success")
            return redirect(url_for("home"))
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", mode="login")

@app.post("/logout")
@login_required
def logout():
    session.clear(); flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.post("/identify")
@login_required
def identify():
    description=request.form.get("description","").strip()
    image=request.files.get("image")
    source="text description estimate"
    image_error=None
    if image and image.filename:
        original_name=secure_filename(image.filename)
        ext=original_name.rsplit(".",1)[-1].lower() if "." in original_name else ""
        if ext not in ALLOWED_EXTENSIONS or image.mimetype not in {"image/jpeg","image/png","image/webp"}:
            flash("Please upload a JPG, PNG, or WEBP image.", "error")
            return redirect(url_for("home"))
        filename=f"{uuid.uuid4().hex}.{ext}"
        image_path=os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(image_path)
        try:
            # The category comes from image pixels, not from the filename or typed description.
            key, confidence=classify_image(image_path)
            source="local AI image analysis (CLIP)"
            if not description:
                description=original_name.rsplit(".",1)[0].replace("_"," ").replace("-"," ") or WASTE[key]["name"]
        except Exception as exc:
            image_error=str(exc)
            key, confidence=classify_text(description)
            if key:
                source="image uploaded; text-description fallback (AI model unavailable)"
            else:
                flash("The image classifier could not start. Install the requirements and try again, or enter a short description. Details: " + image_error[:180], "error")
                return redirect(url_for("home"))
    else:
        key, confidence=classify_text(description)
    if not key:
        flash("Please describe the item or upload an image to analyze.", "error")
        return redirect(url_for("home"))
    info=WASTE[key]
    with db() as c:
        c.execute("""INSERT INTO history(user_id,description,category,waste_key,disposal,confidence,source,created_at)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (session["user_id"],description[:250],info["category"],key,info["bin"],confidence,source,datetime.now().isoformat(timespec="seconds")))
        record_id=c.execute("SELECT last_insert_rowid()").fetchone()[0]
    awarded=award_once(session["user_id"],f"identify:{record_id}",5,"Waste identification")
    if awarded: flash("Waste estimate saved. +5 eco-points earned!", "success")
    return redirect(url_for("result", record_id=record_id))

@app.route("/result/<int:record_id>")
@login_required
def result(record_id):
    with db() as c:
        record=c.execute("SELECT * FROM history WHERE id=? AND user_id=?", (record_id,session["user_id"])).fetchone()
    if not record: abort(404)
    return render_template("result.html", record=record, info=WASTE.get(record["waste_key"],WASTE["general"]))

@app.route("/guide")
@login_required
def disposal_guide():
    return render_template("disposal_guide.html", guide=WASTE)

@app.route("/history")
@login_required
def history():
    with db() as c:
        records=c.execute("SELECT * FROM history WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", records=records)

@app.post("/history/<int:record_id>/delete")
@login_required
def delete_history(record_id):
    with db() as c:
        c.execute("DELETE FROM history WHERE id=? AND user_id=?", (record_id,session["user_id"]))
    flash("History record deleted. Previously earned points are retained for a transparent audit trail.", "success")
    return redirect(url_for("history"))

@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    if request.method=="POST":
        name=request.form.get("full_name","").strip()
        bio=request.form.get("bio","").strip()[:240]
        public=1 if request.form.get("is_public")=="on" else 0
        if len(name)<2 or len(name)>80:
            flash("Name must be between 2 and 80 characters.", "error")
        else:
            with db() as c:
                c.execute("UPDATE users SET full_name=?,bio=?,is_public=? WHERE id=?",
                          (name,bio,public,session["user_id"]))
            flash("Profile updated.", "success")
            return redirect(url_for("profile"))
    u=current_user()
    with db() as c:
        records=c.execute("SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 5",(u["id"],)).fetchall()
        followers=c.execute("SELECT COUNT(*) FROM follows WHERE followed_id=?",(u["id"],)).fetchone()[0]
        following=c.execute("SELECT COUNT(*) FROM follows WHERE follower_id=?",(u["id"],)).fetchone()[0]
        total=c.execute("SELECT COUNT(*) FROM history WHERE user_id=?",(u["id"],)).fetchone()[0]
    return render_template("profile.html", u=u, records=records, followers=followers, following=following, total=total)

@app.route("/settings", methods=["GET","POST"])
@login_required
def settings():
    if request.method=="POST":
        language=request.form.get("language","en")
        if language not in {"en","ta","hi","te","ml","kn"}: language="en"
        with db() as c: c.execute("UPDATE users SET language=? WHERE id=?",(language,session["user_id"]))
        session["language"]=language
        flash("Language preference saved. Core labels are translated; disposal guidance is maintained in English in this starter build.", "success")
        return redirect(url_for("settings"))
    return render_template("settings.html", u=current_user())

@app.post("/change-password")
@login_required
def change_password():
    old=request.form.get("old_password",""); new=request.form.get("new_password",""); confirm=request.form.get("confirm_password","")
    u=current_user()
    if not check_password_hash(u["password_hash"],old):
        flash("Current password is incorrect.", "error")
    elif len(new)<8:
        flash("New password must be at least 8 characters.", "error")
    elif new!=confirm:
        flash("New passwords do not match.", "error")
    else:
        with db() as c: c.execute("UPDATE users SET password_hash=? WHERE id=?",(generate_password_hash(new),u["id"]))
        flash("Password changed successfully.", "success")
    return redirect(url_for("settings"))

@app.route("/community")
@login_required
def community():
    q=request.args.get("q","").strip()[:80]
    with db() as c:
        users=c.execute("""SELECT u.id,u.full_name,u.bio,
          (SELECT COUNT(*) FROM follows f WHERE f.followed_id=u.id) followers,
          (SELECT COALESCE(SUM(points),0) FROM point_events p WHERE p.user_id=u.id) points,
          EXISTS(SELECT 1 FROM follows f WHERE f.follower_id=? AND f.followed_id=u.id) followed
          FROM users u WHERE u.id!=? AND u.is_public=1 AND (u.full_name LIKE ? OR u.email LIKE ?)
          ORDER BY points DESC,u.full_name LIMIT 50""",
          (session["user_id"],session["user_id"],f"%{q}%",f"%{q}%")).fetchall()
    return render_template("leaderboard.html", users=users, q=q)

@app.post("/follow/<int:user_id>")
@login_required
def follow(user_id):
    if user_id==session["user_id"]: flash("You cannot follow yourself.", "error")
    else:
        with db() as c:
            target=c.execute("SELECT id FROM users WHERE id=? AND is_public=1",(user_id,)).fetchone()
            if target:
                c.execute("INSERT OR IGNORE INTO follows(follower_id,followed_id,created_at) VALUES(?,?,?)",
                          (session["user_id"],user_id,datetime.now().isoformat(timespec="seconds")))
                flash("You're now following this community member.", "success")
            else: flash("That profile is unavailable or private.", "error")
    return redirect(request.referrer or url_for("community"))

@app.post("/unfollow/<int:user_id>")
@login_required
def unfollow(user_id):
    with db() as c: c.execute("DELETE FROM follows WHERE follower_id=? AND followed_id=?",(session["user_id"],user_id))
    flash("You unfollowed this member.", "success")
    return redirect(request.referrer or url_for("community"))

@app.route("/rewards")
@login_required
def rewards():
    with db() as c:
        events=c.execute("SELECT * FROM point_events WHERE user_id=? ORDER BY id DESC",(session["user_id"],)).fetchall()
    total=points_total(session["user_id"])
    return render_template("rewards.html", events=events, total=total)

@app.post("/learn/<lesson>")
@login_required
def learn(lesson):
    allowed={"sort-basics":"Sorting basics","battery-safety":"Battery safety","reuse-first":"Reuse before recycling"}
    if lesson not in allowed: abort(404)
    awarded=award_once(session["user_id"],f"lesson:{lesson}",10,f"Completed learning: {allowed[lesson]}")
    flash("Lesson completed! +10 eco-points." if awarded else "You have already earned points for this lesson.", "success")
    return redirect(url_for("rewards"))

@app.route("/api/qr-info")
@login_required
def qr_info():
    # QR scan displays a safe informational result for known demo codes, not arbitrary URLs.
    value=request.args.get("value","").strip().lower()
    known={"greensort:organic":WASTE["organic"],"greensort:ewaste":WASTE["ewaste"],"greensort:battery":WASTE["battery"],"greensort:plastic":WASTE["plastic"]}
    if value in known:
        i=known[value]
        return jsonify({"supported":True,"name":i["name"],"category":i["category"],"bin":i["bin"],"tips":i["tips"]})
    return jsonify({"supported":False,"message":"QR code scanned, but it is not a supported GreenSort demo code. Try greensort:organic, greensort:plastic, greensort:ewaste, or greensort:battery."})

@app.errorhandler(413)
def too_large(_): 
    flash("Image is too large. Maximum upload size is 5 MB.", "error")
    return redirect(url_for("home"))
@app.errorhandler(404)
def not_found(_): return render_template("404.html"),404

if __name__ == "__main__":
    if app.config["SECRET_KEY"]=="dev-only-change-me":
        print("WARNING: Set SECRET_KEY in your .env before using beyond local development.")
    app.run(debug=os.getenv("FLASK_DEBUG","false").lower()=="true")
