"""CLIP zero-shot scene classification + BLIP captions."""
import os, torch
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration

CLIP_ID = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
BLIP_ID = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
SCENES = ["a robbery", "a physical assault", "a traffic accident", "a shooting",
          "a burglary", "vandalism", "a fire", "a normal everyday scene"]
_clip = _blip = None

def classify(img):
    global _clip
    if _clip is None:
        _clip = (CLIPProcessor.from_pretrained(CLIP_ID), CLIPModel.from_pretrained(CLIP_ID).eval())
    proc, model = _clip
    inp = proc(text=[f"a photo of {s}" for s in SCENES], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        p = model(**inp).logits_per_image.softmax(-1)[0]
    return sorted(zip(SCENES, p.tolist()), key=lambda x: -x[1])

def caption(img):
    global _blip
    if _blip is None:
        _blip = (BlipProcessor.from_pretrained(BLIP_ID),
                 BlipForConditionalGeneration.from_pretrained(BLIP_ID).eval())
    proc, model = _blip
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40)
    return proc.decode(out[0], skip_special_tokens=True)
