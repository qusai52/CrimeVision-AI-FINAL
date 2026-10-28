"""Whisper (faster-whisper) multilingual transcription with timestamps."""
import os
from faster_whisper import WhisperModel

_model = None

def transcribe(path, name):
    global _model
    if _model is None:
        _model = WhisperModel(os.getenv("WHISPER_SIZE", "small"), device="auto", compute_type="int8")
    segs, info = _model.transcribe(path, vad_filter=True)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
    events = [{"t": s["start"], "source": name, "type": "speech", "text": f"\u201c{s['text']}\u201d",
               "end": s["end"]} for s in segs if s["text"]]
    return events, {"source": name, "language": info.language, "segments": segs}
