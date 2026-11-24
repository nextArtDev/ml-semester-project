"""Build an editable DOCX from the validated Phase 5 thesis source.

The script reads only docs/thesis/THESIS.md and the copied validated
figures. It uses the repository's existing docx_kit styling helpers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT / "docs" / "tools"
sys.path.insert(0, str(TOOLS))

from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from docx_kit import (  # noqa: E402
    add_page_number_footer,
    add_table_of_contents,
    code,
    h1,
    h2,
    h3,
    new_document,
    p,
    table as add_table,
)

SOURCE = HERE / "THESIS.md"
OUTPUT = HERE / "THESIS.docx"


def strip_markdown(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("$", "")
    return text.strip()


def is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def parse_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = new_document(body_font="Times New Roman", size=12)
    add_page_number_footer(doc, "Thesis Draft")

    # Cover page.
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Exact versus Approximate Nearest Neighbor Search\nfor KNN Classification in PostgreSQL with pgvector")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("MSc Digital Electronics — Semester Project")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("Validated thesis draft — generated from docs/thesis/THESIS.md")
    doc.add_page_break()

    toc_title = doc.add_paragraph()
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    toc_title.add_run("Table of Contents").bold = True
    add_table_of_contents(doc, "1-3")
    doc.add_page_break()

    i = 0
    in_code = False
    code_lines: list[str] = []
    skipped_title = False
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if line.startswith("```"):
            if in_code:
                code(doc, "\n".join(code_lines))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not line:
            i += 1
            continue

        if line == "---":
            i += 1
            continue

        if line.startswith("# "):
            if not skipped_title:
                skipped_title = True
            else:
                h1(doc, strip_markdown(line[2:]))
            i += 1
            continue
        if line.startswith("## "):
            # Markdown level 2 is a thesis chapter; level 3 is a section.
            heading_text = strip_markdown(line[3:])
            if heading_text == "Table of Contents":
                i += 1
                continue
            h1(doc, heading_text)
            i += 1
            continue
        if line.startswith("### "):
            h2(doc, strip_markdown(line[4:]))
            i += 1
            continue
        if line.startswith("#### "):
            h3(doc, strip_markdown(line[5:]))
            i += 1
            continue

        # Markdown image with caption carried in alt text.
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            caption, rel_path = image_match.groups()
            cap = doc.add_paragraph(strip_markdown(caption), style="Caption")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            image_path = HERE / rel_path
            if image_path.is_file():
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.add_run().add_picture(str(image_path), width=Cm(15.5))
            else:
                note = doc.add_paragraph(f"[Missing image: {rel_path}]")
                note.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # Markdown table.
        if line.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            headers = parse_table_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_row(lines[i]))
                i += 1
            caption = ""
            if rows:
                add_table(doc, headers, rows, caption=None)
            continue

        # Bold standalone caption line, e.g. **Table 1. ...**
        if line.startswith("**") and line.endswith("**") and line.count("**") == 2:
            cap = doc.add_paragraph(strip_markdown(line), style="Caption")
            cap.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        if line.startswith("- "):
            doc.add_paragraph(strip_markdown(line[2:]), style="List Bullet")
            i += 1
            continue

        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.add_run(strip_markdown(line))
        i += 1

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
