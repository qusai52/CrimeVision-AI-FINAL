import json, uuid, pathlib

ROOT = pathlib.Path("data/cases")
ROOT.mkdir(parents=True, exist_ok=True)

def new_case():
    cid = uuid.uuid4().hex[:10]
    (ROOT / cid / "evidence").mkdir(parents=True)
    save(cid, {"id": cid, "status": "new", "evidence": [], "timeline": [],
               "detections": [], "transcripts": [], "summary": {}, "chat": []})
    return cid

def path(cid):
    return ROOT / cid

def load(cid):
    return json.loads((ROOT / cid / "case.json").read_text())

def save(cid, c):
    (ROOT / cid / "case.json").write_text(json.dumps(c, indent=2))
