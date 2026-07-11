"""Builds the downloadable PDF claim report for ClaimPilot."""
from fpdf import FPDF


def build_report(result: dict, decision_record: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ClaimPilot - AI Claim Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {decision_record['timestamp']}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Damage Assessment (CNN)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Predicted severity: {result['severity']} (confidence {result['confidence']:.0%})", ln=True)
    pdf.ln(2)
    pdf.image(result["overlay_img"], w=90)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Risk Assessment (ANN)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Estimated fraud/review risk: {result['risk_score']:.0%}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Gemini Reasoning Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    gemini = result["gemini"]
    for line in (
        f"Summary: {gemini['summary']}",
        f"Consistency check: {gemini['consistency_flag']}",
        f"Recommendation: {gemini['recommendation']}",
    ):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, line)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Human Decision", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Decision: {decision_record['decision']}", ln=True)
    pdf.cell(0, 7, f"Adjuster comment: {decision_record.get('comment') or '(none)'}", ln=True)
    pdf.cell(0, 7, f"Decision timestamp: {decision_record['timestamp']}", ln=True)

    return bytes(pdf.output())
