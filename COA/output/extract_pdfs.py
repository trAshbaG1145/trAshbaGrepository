#!/usr/bin/env python3
"""Extract text from all COA PDFs using PyMuPDF (fitz)."""
import fitz
import os
import glob
import re

PDF_DIR = r"e:\Coding\trAshbaGrepository\COA\PPT"
OUT_DIR = r"e:\Coding\trAshbaGrepository\COA\output\extracted_text"

pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

for pdf_path in pdf_files:
    basename = os.path.basename(pdf_path)
    txt_name = os.path.splitext(basename)[0] + ".txt"
    out_path = os.path.join(OUT_DIR, txt_name)

    print(f"Processing: {basename} ...")
    doc = fitz.open(pdf_path)
    lines = []
    lines.append(f"# {os.path.splitext(basename)[0]}")
    lines.append(f"# Pages: {doc.page_count}")
    lines.append("=" * 70)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            lines.append(f"\n## Page {page_num + 1}")
            lines.append(text.strip())

    doc.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    file_size_kb = os.path.getsize(out_path) / 1024
    print(f"  -> Saved: {txt_name} ({file_size_kb:.1f} KB)")

print("\n✅ All PDFs extracted!")
