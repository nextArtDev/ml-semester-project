"""Verify that the code blocks in the generated documents match the source.

A snippet is considered verbatim when its non-empty, whitespace-normalised
lines appear, in order and contiguously, in the concatenated source of the
repository. Snippets that are deliberately illustrative are reported so the
guide can be checked by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ["backend/app", "backend/scripts", "frontend"]


def collect_source() -> str:
    """Concatenate every source file, normalised for whitespace comparison."""
    chunks: list[str] = []
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".ts", ".tsx", ".sql", ".json", ".css", ".mjs"}:
                continue
            if "node_modules" in path.parts or ".next" in path.parts:
                continue
            if "components/ui" in path.as_posix():
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
    return "\n".join(chunks)


def normalise(text: str) -> list[str]:
    """Normalise a snippet into comparable lines."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"\s+", " ", stripped)
        lines.append(stripped)
    return lines


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def longest_prefix(haystack: list[str], needle: list[str]) -> int:
    """Length of the longest prefix of ``needle`` found contiguously."""
    for n in range(len(needle), 0, -1):
        if contains_sequence(haystack, needle[:n]):
            return n
    return 0


def classify(text: str) -> str:
    """Classify a snippet so that only real source is held to the verbatim bar.

    Shell command blocks, SQL listings, equations and ASCII diagrams are
    illustrations of what the reader types or sees, not excerpts of the
    source tree, so they are reported separately.
    """
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    first = lines[0] if lines else ""

    if "argmax" in stripped or first.startswith(("ŷ", "ŷ(")):
        return "equation"
    if first.startswith("ml-semester-project/") or first.startswith("CSV /"):
        return "diagram"
    if first.startswith(
        (
            "cd ",
            "git clone",
            "$PGVECTOR",
            "icacls",
            "psql ",
            "python -m",
            "nmake",
            "-- ",
            "# ",
            "curl",
        )
    ):
        return "command"
    if first.startswith(("CREATE ", "ALTER ", "SELECT ", "DROP ", "SET ")):
        return "sql"
    return "code"


def main() -> int:
    doc_path = Path(sys.argv[1])
    doc = Document(doc_path)
    source = normalise(collect_source())

    blocks: list[tuple[str, str]] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        runs = paragraph.runs
        if not runs:
            continue
        # Code blocks are the paragraphs set in Consolas.
        if runs[0].font.name != "Consolas":
            continue
        blocks.append((text, ""))

    print(f"file      : {doc_path.name}")
    print(f"code blocks: {len(blocks)}")
    print()

    verbatim = 0
    other: dict[str, list[int]] = {
        "command": [],
        "sql": [],
        "diagram": [],
        "equation": [],
    }
    failures: list[int] = []

    for index, (text, _) in enumerate(blocks, start=1):
        needle = normalise(text)
        kind = classify(text)

        if kind != "code":
            other[kind].append(index)
            continue

        if contains_sequence(source, needle):
            verbatim += 1
            continue

        prefix = longest_prefix(source, needle)
        failures.append(index)
        print(f"[{index:>2}] CODE BLOCK DIVERGES after {prefix}/{len(needle)} lines")
        if prefix < len(needle):
            print(f"     first unmatched: {needle[prefix][:88]}")
        print()

    code_blocks = verbatim + len(failures)
    print(f"code blocks verified verbatim : {verbatim}/{code_blocks}")
    for kind, indices in other.items():
        print(f"{kind:<26} : {len(indices)} (illustrative, not source)")
    if failures:
        print(f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
