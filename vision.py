"""Image and frame-by-frame video analysis -> timeline events."""
import os
from collections import Counter
import cv2, numpy as np
from PIL import Image
import detector, scene

MOTION_T = 12.0  # mean abs gray diff that counts as rapid movement

def _summ(dets):
    return ", ".join(f"{k} x{v}" for k, v in Counter(d["label"] for d in dets).items())

def _ev(t, src, typ, text, **kw):
    return {"t": t, "source": src, "type": typ, "text": text, **kw}

def analyze_image(path, name):
    img = Image.open(path).convert("RGB")
    dets = detector.detect(img)
    cls = scene.classify(img)
    ev = [_ev(None, name, "scene_caption", scene.caption(img)),
          _ev(None, name, "scene_class",
              f"Scene most resembles {cls[0][0]} ({cls[0][1]:.0%}), then {cls[1][0]} ({cls[1][1]:.0%})",
              score=round(cls[0][1], 3))]
    if dets:
        ev.append(_ev(None, name, "object_appeared", "Detected: " + _summ(dets),
                      score=max(d["score"] for d in dets)))
    for d in dets:
        if d["weapon"]:
            ev.append(_ev(None, name, "weapon_detected",
                          f"Possible {d['label']} (confidence {d['score']:.2f})",
                          score=d["score"], box=d["box"]))
    return ev, [{"source": name, "t": None, **d} for d in dets]

def analyze_video(path, name, every=1.0, max_samples=int(os.getenv("MAX_SAMPLES", 150)), caption_every=5):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps * every)))
    events, out_dets = [], []
    seen, prev_w, prev_people, prev_gray, prev_fast = set(), set(), 0, None, False
    i = n = 0
    t = 0.0
    while n < max_samples:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = round(i / fps, 2)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = detector.detect(img)

            for lbl in sorted({d["label"] for d in dets} - seen):
                seen.add(lbl)
                events.append(_ev(t, name, "object_appeared", f"First appearance of: {lbl}"))
            people = sum(d["label"] == "person" for d in dets)
            if people != prev_people:
                events.append(_ev(t, name, "people_change", f"People in frame: {prev_people} to {people}"))
                prev_people = people
            w = {d["label"] for d in dets if d["weapon"]}
            for lbl in sorted(w - prev_w):
                best = max(d["score"] for d in dets if d["label"] == lbl)
                events.append(_ev(t, name, "weapon_detected",
                                  f"Possible {lbl} (confidence {best:.2f})", score=best))
            prev_w = w
            out_dets += [{"source": name, "t": t, **d} for d in dets if d["weapon"]]

            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev_gray is not None:
                m = float(np.mean(cv2.absdiff(g, prev_gray)))
                fast = m > MOTION_T
                if fast and not prev_fast:
                    events.append(_ev(t, name, "high_motion",
                                      f"Rapid movement or large scene change (motion score {m:.1f})",
                                      score=min(1.0, m / 40)))
                prev_fast = fast
            prev_gray = g

            if n % caption_every == 0:
                events.append(_ev(t, name, "scene_caption", scene.caption(img)))
            n += 1
        i += 1
    cap.release()
    if n >= max_samples:
        events.append(_ev(t, name, "note", f"Analysis stopped after {max_samples} sampled frames (t={t}s); later footage not analyzed."))
    return events, out_dets
