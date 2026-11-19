"""Extract Chapter II from the RBI Annual Report 2025-26."""
import json
import sys
from pathlib import Path

from pypdf import PdfReader


def extract(pdf):
    reader = PdfReader(pdf)
    if len(reader.pages) == 46:
        chapter = reader.pages
    elif len(reader.pages) == 246:
        chapter = reader.pages[38:84]
    else:
        raise ValueError("Use the included chapter PDF or the full 2025-26 report.")
    pages = {}
    for page, pdf_page in enumerate(chapter, start=13):
        lines = pdf_page.extract_text().splitlines()
        header = "ECONOMIC REVIEW" if page % 2 else "ANNUAL REPORT 2025-26"
        if lines[0].strip() != header or lines[1].strip() != str(page):
            raise ValueError(f"Unexpected header on PDF page {page + 26}. Check the edition.")
        pages[page] = " ".join(" ".join(lines[2:]).split())
    return pages


if __name__ == "__main__":
    output = Path(__file__).with_name("pages.json")
    output.write_text(json.dumps(extract(sys.argv[1]), ensure_ascii=False, indent=2), encoding="utf-8")
