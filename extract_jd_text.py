import sys
from docx import Document


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_jd_text.py <input.docx> <output.txt>")
        sys.exit(1)

    doc = Document(sys.argv[1])
    text = "\n".join(p.text for p in doc.paragraphs)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extracted {len(text)} characters to {sys.argv[2]}")


if __name__ == "__main__":
    main()
