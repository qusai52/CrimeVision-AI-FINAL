import os, re, hashlib, hmac, threading
from flask import Flask, request, jsonify, render_template, send_file, abort, Response
from werkzeug.utils import secure_filename
import store, fusion

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 ** 3
LOCK = threading.Lock()  # one analysis at a time (models are heavy)
EXT = {"image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
       "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
       "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac"}}

@app.before_request
def gate():
    # Optional shared password for public demos: set APP_PASSWORD (any username works).
    pw = os.getenv("APP_PASSWORD")
    if pw:
        auth = request.authorization
        if not (auth and hmac.compare_digest(auth.password or "", pw)):
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="CrimeVision"'})

def kind(name):
    e = os.path.splitext(name)[1].lower()
    return next((k for k, v in EXT.items() if e in v), None)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def case_or_404(cid):
    if not re.fullmatch(r"[a-f0-9]{10}", cid) or not (store.ROOT / cid).exists():
        abort(404)
    return store.load(cid)

def process(cid):
    with LOCK:
        c = store.load(cid)
        try:
            import vision, audio, llm  # heavy imports, loaded on first use
            events, dets, trs = [], [], []
            for ev in c["evidence"]:
                c["progress"] = f"Analyzing {ev['name']}"
                store.save(cid, c)
                p = str(store.path(cid) / "evidence" / ev["name"])
                if ev["kind"] == "image":
                    e, d = vision.analyze_image(p, ev["name"]); events += e; dets += d
                elif ev["kind"] == "video":
                    e, d = vision.analyze_video(p, ev["name"]); events += e; dets += d
                else:
                    e, tr = audio.transcribe(p, ev["name"]); events += e; trs.append(tr)
            c["timeline"], c["detections"], c["transcripts"] = fusion.build(events), dets, trs
            c["progress"] = "Reasoning over fused evidence"
            store.save(cid, c)
            c["summary"] = llm.analyze(fusion.context(c))
            c["status"] = "done"
        except Exception as e:
            c["status"], c["error"] = "error", str(e)
        store.save(cid, c)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/cases")
def create():
    files = request.files.getlist("files")
    cid = store.new_case()
    c = store.load(cid)
    for f in files:
        name = secure_filename(f.filename or "")
        k = kind(name)
        if not k:
            continue
        if (store.path(cid) / "evidence" / name).exists():
            name = f"{len(c['evidence'])}_{name}"
        dest = store.path(cid) / "evidence" / name
        f.save(dest)
        c["evidence"].append({"name": name, "kind": k, "sha256": sha256(dest)})
    if not c["evidence"]:
        return jsonify(error="No supported image, video or audio files."), 400
    c["status"] = "processing"
    store.save(cid, c)
    threading.Thread(target=process, args=(cid,), daemon=True).start()
    return jsonify(id=cid), 202

@app.get("/api/cases/<cid>")
def get_case(cid):
    return jsonify(case_or_404(cid))

@app.get("/api/cases/<cid>/timeline")
def timeline(cid):
    return jsonify(case_or_404(cid)["timeline"])

@app.post("/api/cases/<cid>/chat")
def chat(cid):
    import llm
    c = case_or_404(cid)
    q = (request.get_json(silent=True) or {}).get("q", "").strip()
    if c["status"] != "done" or not q:
        return jsonify(error="Case not ready or empty question."), 400
    try:
        a = llm.chat(fusion.context(c), c["chat"], q)
    except Exception as e:
        return jsonify(error=f"LLM error: {e}"), 502
    c["chat"] += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    store.save(cid, c)
    return jsonify(answer=a)

@app.get("/api/cases/<cid>/report")
def report_pdf(cid):
    import report
    c = case_or_404(cid)
    if c["status"] != "done":
        abort(409)
    return send_file(report.build(c), mimetype="application/pdf", download_name=f"forensic_report_{cid}.pdf")

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 5000)))  # local by default
