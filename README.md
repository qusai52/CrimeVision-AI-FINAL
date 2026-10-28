---
title: CrimeVision AI
emoji: 🔍
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# CrimeVision AI

Multimodal forensic analysis: RT-DETR detection, CLIP + BLIP scene understanding, Whisper transcription,
frame-by-frame video analysis, fused timeline, Groq LLM reasoning and evidence chat, Flask dashboard
with 3D timeline, PDF reports.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs ffmpeg installed for some audio formats
export GROQ_API_KEY=...                  # https://console.groq.com
python app.py                            # http://127.0.0.1:5000
```
First run downloads models (~2 GB). CPU works; a GPU makes video much faster.

## Config (env vars)
| Var | Default | Purpose |
|---|---|---|
| DETECTOR_MODEL | PekingU/rtdetr_r50vd | RT-DETR checkpoint (HF format) |
| WEAPON_LABELS | knife,baseball bat,scissors,gun,... | class names flagged as weapons |
| BLIP_MODEL | Salesforce/blip-image-captioning-base | caption model |
| WHISPER_SIZE | small | tiny/base/small/medium/large-v3 |
| GROQ_MODEL | llama-3.3-70b-versatile | LLM |

## API
POST `/api/cases` (multipart `files`) · GET `/api/cases/<id>` · GET `/api/cases/<id>/timeline` ·
POST `/api/cases/<id>/chat` `{"q": "..."}` · GET `/api/cases/<id>/report`

## Known limits
- The default COCO RT-DETR cannot see firearms. Fine-tune or swap in a weapon-trained checkpoint.
- "Action detection" is heuristic (people-count changes, motion spikes, BLIP captions), not a trained action model.
- Prototype: localhost only, no auth, JSON file storage. Add authentication, audit logging and access
  control before any real casework. All output is decision support and needs human verification.
