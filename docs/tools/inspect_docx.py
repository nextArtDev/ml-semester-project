"""Inspect a generated .docx: headings, paragraph and table counts."""

import sys
from pathlib import Path

from docx import Document


def main() -> int:
    path = Path(sys.argv[1])
    doc = Document(path)

    headings = [
        (p.style.name, p.text)
        for p in doc.paragraphs
        if p.style.name.startswith(("Heading", "Title"))
    ]
    tables = doc.tables
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    table_words = sum(
        len(c.text.split()) for t in tables for r in t.rows for c in r.cells
    )

    print(f"file      : {path.name}")
    print(f"size      : {path.stat().st_size / 1024:.0f} KB")
    print(f"paragraphs: {len(doc.paragraphs)}")
    print(f"tables    : {len(tables)}")
    print(f"words     : {words + table_words:,} (body {words:,} + tables {table_words:,})")
    print()
    print("--- outline ---")
    for style, text in headings:
        level = 0 if style == "Title" else int(style.split()[-1])
        print(f"{'  ' * level}{text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
