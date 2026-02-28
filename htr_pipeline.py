import os
import json
import re
from pathlib import Path
from google import genai
from htr import extract_handwritten_text, get_client, load_env

load_env()

GEMINI_MODEL    = "gemini-2.5-flash-lite"
HANDWRITTEN_PDF = "handwritten-ans.pdf"
OUTPUT_JSON     = "final_output.json"
OUTPUT_REPORT   = "grading_report.txt"


def _call_gemini(client, prompt):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text.strip()


def _parse_json(raw):
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse response as JSON:\n{raw[:300]}")


# ── PARSE SOLUTIONS ───────────────────────────────────────────────────────────
def extract_solutions(client, solutions_txt_path="solutions.txt"):
    print("\n[Step 2] Parsing solution key...")
    with open(solutions_txt_path, "r", encoding="utf-8") as f:
        solutions_text = f.read()

    prompt = f"""Parse this exam solutions document and extract all correct answers.

SOLUTIONS TEXT:
{solutions_text}

Return JSON in this exact format:
{{
  "solutions": [
    {{
      "question_number": "1",
      "correct_answer": "B",
      "answer_type": "mcq",
      "marks": 2,
      "explanation": "Brief explanation"
    }}
  ],
  "total_marks": 50
}}

Return ONLY valid JSON."""

    raw = _call_gemini(client, prompt)
    result = _parse_json(raw)
    print(f"      Found {len(result.get('solutions', []))} solution(s)")
    print(f"      Total marks: {result.get('total_marks', 'N/A')}")
    return result


# ── GRADE ANSWERS ─────────────────────────────────────────────────────────────
def grade_answers(client, extracted_answers, solutions, questions_text=""):
    print("\n[Step 3] Grading answers...")

    prompt = f"""You are a strict but fair exam grader for CA (Chartered Accountancy) exams.

Grading rules:
- MCQ: Full marks for correct answer only. Zero for wrong.
- Written: Award marks proportionally based on key points covered.
- Be specific in feedback.

STUDENT ANSWERS:
{json.dumps(extracted_answers.get("answers", []), indent=2)}

SOLUTION KEY:
{json.dumps(solutions.get("solutions", []), indent=2)}

{f"QUESTIONS CONTEXT:{chr(10)}{questions_text[:3000]}" if questions_text else ""}

Return grading report as JSON:
{{
  "results": [
    {{
      "question_number": "1",
      "student_answer": "B",
      "correct_answer": "B",
      "is_correct": true,
      "marks_awarded": 2,
      "marks_available": 2,
      "feedback": "Correct!"
    }}
  ],
  "total_marks_awarded": 20,
  "total_marks_available": 50,
  "percentage": 40.0,
  "grade": "Pass",
  "overall_feedback": "Summary of performance."
}}

Return ONLY valid JSON."""

    raw = _call_gemini(client, prompt)
    result = _parse_json(raw)
    awarded = result.get("total_marks_awarded", 0)
    available = result.get("total_marks_available", 0)
    print(f"      Score: {awarded}/{available} ({result.get('percentage', 0):.1f}%)")
    print(f"      Grade: {result.get('grade', 'N/A')}")
    return result


# ── GENERATE REPORT ───────────────────────────────────────────────────────────
def generate_report(extracted, solutions, grading, output_path=OUTPUT_REPORT):
    lines = []
    lines.append("=" * 60)
    lines.append("         PAPER CORRECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"\nSource: {extracted.get('source_pdf', 'N/A')}")
    lines.append(f"Questions extracted: {extracted.get('total_questions_found', 0)}")
    lines.append("\n── SCORE SUMMARY " + "─" * 43)
    awarded = grading.get("total_marks_awarded", 0)
    available = grading.get("total_marks_available", 0)
    lines.append(f"  Total Marks : {awarded} / {available}")
    lines.append(f"  Percentage  : {grading.get('percentage', 0):.1f}%")
    lines.append(f"  Grade       : {grading.get('grade', 'N/A')}")
    lines.append("\n── OVERALL FEEDBACK " + "─" * 40)
    lines.append(f"  {grading.get('overall_feedback', '')}")
    lines.append("\n── QUESTION-WISE RESULTS " + "─" * 35)
    for r in grading.get("results", []):
        status = "✓" if r.get("is_correct") else ("~" if r.get("marks_awarded", 0) > 0 else "✗")
        lines.append(f"\n  Q{r['question_number']} [{status}] {r.get('marks_awarded',0)}/{r.get('marks_available',0)} marks")
        lines.append(f"  Student : {str(r.get('student_answer',''))[:100]}")
        lines.append(f"  Correct : {str(r.get('correct_answer',''))[:100]}")
        lines.append(f"  Feedback: {r.get('feedback','')}")
    lines.append("\n" + "=" * 60)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Step 4] Report saved → {output_path}")
    return report_text


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("   PAPER CORRECTION AI  —  Full Pipeline (FREE)")
    print("   Powered by Google Gemini 2.5 Flash-Lite (1000 RPD free)")
    print("=" * 60)

    client = get_client()

    print("\n[Step 1] Extracting handwritten answers...")
    extracted = extract_handwritten_text(
        pdf_path=HANDWRITTEN_PDF,
        output_txt="handwritten-ans.txt",
        output_json="handwritten-ans.json",
    )

    if os.path.exists("solutions.txt"):
        solutions = extract_solutions(client, "solutions.txt")
    else:
        print("\n[Step 2] solutions.txt not found — run extractor.py first")
        solutions = {"solutions": [], "total_marks": 0}

    questions_text = ""
    if os.path.exists("question.txt"):
        with open("question.txt", "r", encoding="utf-8") as f:
            questions_text = f.read()

    grading = grade_answers(client, extracted, solutions, questions_text)

    final_output = {"extraction": extracted, "solutions": solutions, "grading": grading}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n      Full output saved → {OUTPUT_JSON}")

    report = generate_report(extracted, solutions, grading)
    print("\n" + "─" * 60)
    print(report)
    return final_output


if __name__ == "__main__":
    run_pipeline()