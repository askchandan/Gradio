from pdf2image import convert_from_path
import pytesseract


def extract_text_from_pdf(pdf_path, output_txt_path):
    """Extract text from a printed/typed PDF using Tesseract OCR."""
    print(f"Extracting text from {pdf_path}...")
    pages = convert_from_path(pdf_path)

    full_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        full_text += f"--- Page {i + 1} ---\n{text}\n\n"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  Saved → {output_txt_path}")
    return full_text


if __name__ == "__main__":
    extract_text_from_pdf("questions.pdf", "question.txt")
    extract_text_from_pdf("solutions.pdf", "solutions.txt")
    print("\nDone! Run htr_pipeline.py next.")
