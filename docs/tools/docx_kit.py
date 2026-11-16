"""Shared helpers for generating the project's Word documents.

The two documents produced by this package are:

* ``build_thesis.py``  — the academic thesis.
* ``build_guide.py``   — the function-by-function project guide.

Both share the styling helpers defined here so that they look like parts of
the same series.
"""

from __future__ import annotations

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# --- Palette ---------------------------------------------------------------
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_LIGHT = "D9E2F3"
CODE_SHADE = "F4F5F7"
NOTE_SHADE = "FFF6E5"
HEADER_FILL = "1F4E79"


# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------
def _set_cell_background(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _add_field(paragraph, instruction: str, placeholder: str = "") -> None:
    """Insert a Word field code (used for the TOC and page numbers)."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    run._r.append(text)
    run._r.append(end)


# ---------------------------------------------------------------------------
# Document scaffolding
# ---------------------------------------------------------------------------
def new_document(body_font: str = "Times New Roman", size: int = 12) -> Document:
    """Create a document with the shared page setup and base styles."""
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    styles = doc.styles

    normal = styles["Normal"]
    normal.font.name = body_font
    normal.font.size = Pt(size)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size_pt, color, before, after in (
        ("Heading 1", 18, ACCENT, 18, 10),
        ("Heading 2", 15, ACCENT, 14, 8),
        ("Heading 3", 13, ACCENT, 12, 6),
        ("Heading 4", 12, ACCENT, 10, 4),
    ):
        style = styles[name]
        style.font.name = body_font
        style.font.size = Pt(size_pt)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.keep_with_next = True

    title = styles["Title"]
    title.font.name = body_font
    title.font.size = Pt(26)
    title.font.bold = True
    title.font.color.rgb = ACCENT

    caption = styles["Caption"]
    caption.font.name = body_font
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    return doc


def add_page_number_footer(doc: Document, label: str = "") -> None:
    """Add a footer with a page number (and an optional document label)."""
    for section in doc.sections:
        footer = section.footer
        paragraph = footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0

        run = paragraph.add_run(f"{label}  ·  Page " if label else "Page ")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        _add_field(paragraph, "PAGE", "1")

        if label:
            paragraph.add_run(" of ").font.size = Pt(9)
            _add_field(paragraph, "NUMPAGES", "1")


def add_table_of_contents(doc: Document, levels: str = "1-3") -> None:
    """Insert a table-of-contents field (refresh with F9 in Word)."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.0
    _add_field(paragraph, f'TOC \\o "{levels}" \\h \\z \\u', "Right-click and choose Update Field")


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def h1(doc: Document, text: str) -> None:
    doc.add_heading(text, level=1)


def h2(doc: Document, text: str) -> None:
    doc.add_heading(text, level=2)


def h3(doc: Document, text: str) -> None:
    doc.add_heading(text, level=3)


def h4(doc: Document, text: str) -> None:
    doc.add_heading(text, level=4)


def p(doc: Document, text: str = "", bold_lead: str | None = None) -> None:
    """Add a body paragraph, optionally with a bold leading phrase."""
    paragraph = doc.add_paragraph()
    if bold_lead:
        run = paragraph.add_run(bold_lead)
        run.bold = True
    paragraph.add_run(text)
    return paragraph


def bullet(doc: Document, text: str, level: int = 0) -> None:
    paragraph = doc.add_paragraph(text, style="List Bullet")
    paragraph.paragraph_format.left_indent = Cm(0.75 + 0.6 * level)
    paragraph.paragraph_format.line_spacing = 1.3
    paragraph.paragraph_format.space_after = Pt(2)
    return paragraph


def numbered(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(text, style="List Number")
    paragraph.paragraph_format.left_indent = Cm(0.75)
    paragraph.paragraph_format.line_spacing = 1.3
    paragraph.paragraph_format.space_after = Pt(2)
    return paragraph


def code(doc: Document, snippet: str, caption: str | None = None) -> None:
    """Add a shaded monospace code block."""
    if caption:
        cap = doc.add_paragraph(caption, style="Caption")
        cap.paragraph_format.space_after = Pt(2)

    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.left_indent = Cm(0.5)
    fmt.space_before = Pt(4)
    fmt.space_after = Pt(8)
    fmt.line_spacing = 1.05
    fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT

    run = paragraph.add_run(snippet.rstrip("\n"))
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")

    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), CODE_SHADE)
    p_pr.append(shd)

    borders = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:space"), "4")
        edge.set(qn("w:color"), "D0D3D8")
        borders.append(edge)
    p_pr.append(borders)

    return paragraph


def note(doc: Document, text: str, label: str = "Note") -> None:
    """Add a shaded call-out box."""
    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.left_indent = Cm(0.4)
    fmt.right_indent = Cm(0.4)
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(10)
    fmt.line_spacing = 1.2

    lead = paragraph.add_run(f"{label}: ")
    lead.bold = True
    lead.font.size = Pt(10)
    run = paragraph.add_run(text)
    run.font.size = Pt(10)

    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), NOTE_SHADE)
    p_pr.append(shd)
    return paragraph


def table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    caption: str | None = None,
    widths: list[float] | None = None,
    font_size: int = 9,
) -> None:
    """Add a formatted table with a repeating header row."""
    if caption:
        cap = doc.add_paragraph(caption, style="Caption")
        cap.paragraph_format.space_after = Pt(4)
        cap.alignment = WD_ALIGN_PARAGRAPH.LEFT

    table_obj = doc.add_table(rows=1, cols=len(headers))
    table_obj.style = "Table Grid"
    table_obj.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_obj.autofit = True

    header_cells = table_obj.rows[0].cells
    for cell, text in zip(header_cells, headers):
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(text)
        run.bold = True
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_background(cell, HEADER_FILL)
    _set_repeat_header(table_obj.rows[0])

    for row_values in rows:
        cells = table_obj.add_row().cells
        for index, (cell, text) in enumerate(zip(cells, row_values)):
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(1)
            run = paragraph.add_run(str(text))
            run.font.size = Pt(font_size)
            if index > 0 and str(text).replace(".", "").replace("-", "").isdigit():
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    if widths:
        for row in table_obj.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Cm(width)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(text, style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(10)


def page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def bullet_list(doc: Document, items: list[str], level: int = 0) -> None:
    for item in items:
        bullet(doc, item, level)


def key_value_table(doc: Document, pairs: list[tuple[str, str]], widths=(5.0, 10.5)) -> None:
    table(doc, ["Item", "Description"], [[k, v] for k, v in pairs], widths=list(widths))
