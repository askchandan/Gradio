import os
import json
import re
from pathlib import Path
from google import genai
from google.genai import types

# ── LOAD .env FILE ────────────────────────────────────────────────────────────
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.5-flash-lite"   
HANDWRITTEN_PDF = "handwritten-ans.pdf"
OUTPUT_TXT      = "handwritten-ans.txt"
OUTPUT_JSON     = "handwritten-ans.json"


# ── CLIENT ───────────────────────────────────────────────────────────────────
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "\n❌ Gemini API key not set!\n"
            "Open the .env file in your project folder and replace:\n"
            "   GEMINI_API_KEY=your_gemini_api_key_here\n"
            "with your actual key from https://aistudio.google.com/apikey\n"
        )
    return genai.Client(api_key=api_key)


# ── UPLOAD PDF ────────────────────────────────────────────────────────────────
def upload_pdf(client, pdf_path):
    print(f"  Uploading PDF to Gemini File API...")
    uploaded = client.files.upload(file=pdf_path)
    print(f"  Uploaded: {uploaded.name}")
    return uploaded


# ── EXTRACT ALL PAGES IN ONE REQUEST ─────────────────────────────────────────
def extract_all_pages(client, pdf_file):
    print(f"  Sending entire PDF in ONE request (all pages at once)...")

    prompt = """You are an expert handwriting recognition system for exam answer sheets.

Process ALL pages of this PDF and transcribe ALL handwritten content.

Rules:
- Transcribe EXACTLY what is written — preserve the student's words verbatim
- For MCQ answers: detect patterns like "1. B", "2. C", "3. A" etc.
- For written answers: transcribe the full handwritten text faithfully
- If handwriting is unclear, note [unclear]
- Group content by question number
- Do NOT correct grammar or spelling — just transcribe
- Ignore printed text — only extract handwritten content
- Process ALL pages, not just the first one

Return ALL handwritten answers as JSON:
{
  "answers": [
    {"question_number": "1", "answer_text": "B", "answer_type": "mcq"},
    {"question_number": "2", "answer_text": "The auditor should verify...", "answer_type": "written"}
  ],
  "total_questions_found": 10,
  "pages_processed": 9,
  "extraction_notes": "Any notes about handwriting quality"
}

Return ONLY valid JSON. No markdown, no explanation."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, prompt]
    )

    raw = response.text.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "answers": [{"question_number": "?", "answer_text": raw, "answer_type": "raw"}],
            "total_questions_found": 0,
            "pages_processed": "unknown",
            "extraction_notes": "JSON parse failed — raw text included"
        }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def extract_handwritten_text(pdf_path=HANDWRITTEN_PDF,
                              output_txt=OUTPUT_TXT,
                              output_json=OUTPUT_JSON):
    print("=" * 58)
    print("  Paper Correction AI — Gemini Vision HTR (FREE)")
    print("  Entire PDF processed in a SINGLE API call")
    print("=" * 58)

    client = get_client()

    print(f"\n[1/3] Uploading PDF: {pdf_path}")
    pdf_file = upload_pdf(client, pdf_path)

    print(f"\n[2/3] Extracting handwriting...")
    result = extract_all_pages(client, pdf_file)

    # Cleanup uploaded file from Gemini servers
    try:
        client.files.delete(name=pdf_file.name)
    except Exception:
        pass

    found = result.get("total_questions_found", len(result.get("answers", [])))
    pages = result.get("pages_processed", "?")
    print(f"  → Found {found} answer(s) across {pages} page(s)")
    if result.get("extraction_notes"):
        print(f"  ⚠ Notes: {result['extraction_notes']}")

    result["source_pdf"] = pdf_path
    result["api_calls_used"] = 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] Saved JSON → {output_json}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"# Handwritten Answers from: {pdf_path}\n")
        f.write(f"# Questions found: {found}\n")
        f.write(f"# Notes: {result.get('extraction_notes', 'Clean extraction')}\n\n")
        for ans in result.get("answers", []):
            f.write(f"Q{ans['question_number']}. [{ans.get('answer_type','written').upper()}]\n")
            f.write(f"{ans['answer_text']}\n\n")
    print(f"         Saved text  → {output_txt}")

    print(f"\n✓ Done! Used only 1 API call for {pages} pages.")
    return result


if __name__ == "__main__":
    extract_handwritten_text()