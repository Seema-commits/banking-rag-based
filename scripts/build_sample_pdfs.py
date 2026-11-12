"""Build the Thistlewood Bank sample PDFs from the Markdown sources.

Usage (from the project root):
    pip install reportlab
    python scripts/build_sample_pdfs.py

Supports the small Markdown subset used in sample-docs/: '# ' and '## ' headings,
'- ' bullets, '1. ' numbered steps, paragraphs, '---pagebreak---' and '---' (rule).
"""

import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "sample-docs" / "thistlewood"
FOOTER = "Thistlewood Bank is fictional. Portfolio demo, not affiliated with any bank or the RBI."

styles = getSampleStyleSheet()


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(20 * mm, 12 * mm, FOOTER)
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(md_path: Path) -> Path:
    story, items, numbered = [], [], False

    def flush_list():
        nonlocal items
        if items:
            kw = {"bulletType": "1"} if numbered else {"bulletType": "bullet", "start": "•"}
            story.append(ListFlowable([ListItem(Paragraph(_inline(i), styles["BodyText"])) for i in items], **kw))
            items = []

    for line in md_path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        bullet = re.match(r"^- (.*)", line)
        step = re.match(r"^\d+\. (.*)", line)
        if bullet or step:
            if items and numbered != bool(step):
                flush_list()
            numbered = bool(step)
            items.append((bullet or step).group(1))
            continue
        flush_list()
        if not line:
            continue
        if line == "---pagebreak---":
            story.append(PageBreak())
        elif line == "---":
            story.append(HRFlowable(width="100%"))
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["Heading2"]))
        else:
            story.append(Paragraph(_inline(line), styles["BodyText"]))
            story.append(Spacer(1, 2))
    flush_list()

    out = md_path.with_suffix(".pdf")
    title = md_path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    SimpleDocTemplate(str(out), pagesize=A4, title=title, author="Thistlewood Bank (fictional)",
                      bottomMargin=22 * mm).build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out


if __name__ == "__main__":
    for md in sorted(SRC_DIR.glob("*.md")):
        print("built", build(md).relative_to(ROOT))
