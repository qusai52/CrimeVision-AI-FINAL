"""Merge per-modality events into one timeline + LLM context."""

def mmss(t):
    return "--:--" if t is None else f"{int(t // 60):02d}:{int(t % 60):02d}"

def build(events):
    # untimed (still-image) events first, then chronological
    return sorted(events, key=lambda e: (e["t"] is not None, e["t"] or 0))

def context(c, limit=14000):
    files = "\n".join(f"- {e['name']} ({e['kind']}, sha256 {e['sha256'][:12]})" for e in c["evidence"])
    lines = "\n".join(f"[{mmss(e['t'])}] ({e['source']}/{e['type']}) {e['text']}" for e in c["timeline"])
    return f"EVIDENCE FILES:\n{files}\n\nTIMELINE (automated output, may contain errors):\n{lines}"[:limit]
