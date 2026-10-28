"""RT-DETR object detection. Default checkpoint is COCO (knife, bat, scissors only).
For firearms etc., point DETECTOR_MODEL at a weapon-trained RT-DETR checkpoint
in Hugging Face format and set WEAPON_LABELS to its weapon class names."""
import os, torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

MODEL = os.getenv("DETECTOR_MODEL", "PekingU/rtdetr_r50vd")
WEAPONS = {w.strip().lower() for w in os.getenv(
    "WEAPON_LABELS", "knife,baseball bat,scissors,gun,pistol,rifle,firearm").split(",")}
_proc = _model = None

def _load():
    global _proc, _model
    if _model is None:
        _proc = RTDetrImageProcessor.from_pretrained(MODEL)
        _model = RTDetrForObjectDetection.from_pretrained(MODEL).eval()
    return _proc, _model

def detect(img: Image.Image, thr=0.5):
    proc, model = _load()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_object_detection(
        out, target_sizes=torch.tensor([img.size[::-1]]), threshold=thr)[0]
    dets = []
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        name = model.config.id2label[l.item()]
        dets.append({"label": name, "score": round(s.item(), 3),
                     "box": [round(x, 1) for x in b.tolist()],
                     "weapon": name.lower() in WEAPONS})
    return dets
