"""Forensic PDF report (reportlab)."""
from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import fusion

def build(c):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Forensic Report {c['id']}",
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
    P = lambda t, s="BodyText": Paragraph(escape(str(t)), ss[s])
    cell = lambda t: Paragraph(escape(str(t)), small)

    def table(rows, widths):
        t = Table([[cell(x) for x in r] for r in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("GRID", (0, 0), (-1, -1), .25, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    s = c.get("summary", {})
    out = [P("CrimeVision AI \u2013 Forensic Analysis Report", "Title"),
           P(f"Case {c['id']} \u00b7 generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"),
           Spacer(1, 8),
           P("Evidence received", "Heading2"),
           table([["File", "Type", "SHA-256"]] + [[e["name"], e["kind"], e["sha256"]] for e in c["evidence"]],
                 [140, 50, 345]),
           P("AI assessment", "Heading2"),
           P(f"Classification: {s.get('crime_type', 'n/a')} (model confidence {s.get('confidence', 0)})"),
           P(s.get("reasoning", "")),
           P("Possible scenario", "Heading2"), P(s.get("scenario", ""))]
    for title, key in (("Key findings", "key_findings"), ("Suggested human verification", "follow_up"),
                       ("Limitations", "limitations")):
        out.append(P(title, "Heading2"))
        out += [P("\u2022 " + str(x)) for x in s.get(key, [])] or [P("None listed.")]
    out += [P("Unified timeline", "Heading2"),
            table([["Time", "Source", "Type", "Description"]] +
                  [[fusion.mmss(e["t"]), e["source"], e["type"], e["text"]] for e in c["timeline"]],
                  [40, 100, 85, 310])]
    for tr in c.get("transcripts", []):
        out += [P(f"Transcript: {tr['source']} (detected language: {tr['language']})", "Heading2"),
                table([["Start", "End", "Text"]] + [[x["start"], x["end"], x["text"]] for x in tr["segments"]],
                      [40, 40, 455])]
    out += [Spacer(1, 12), P("This report is automated decision-support output. Detections, captions, "
                             "transcripts and conclusions can be wrong and must be verified by a qualified "
                             "human examiner against the original evidence before any use.", "Italic")]
    doc.build(out)
    buf.seek(0)
    return buf
