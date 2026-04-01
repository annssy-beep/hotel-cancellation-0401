ROLE_GUIDANCE = {
    "Revenue Manager": (
        "Focus on profit-aware intervention, threshold management, high-revenue bookings, "
        "overbooking reference, deposit/prepayment policy, and prioritization."
    ),
    "Reservations / Rooms Manager": (
        "Focus on booking-level follow-up, reconfirmation, payment-status checks, "
        "manual review queues, and operational action timing."
    ),
    "Marketing / Sales": (
        "Focus on segment/channel instability, OTA/direct/group patterns, campaign refinement, "
        "and demand-shaping actions."
    ),
    "GM / Finance": (
        "Focus on executive interpretation, month-over-month exposure, expected business impact, "
        "and governance-level decisions."
    ),
}


def build_system_prompt(role: str) -> str:
    role_guidance = ROLE_GUIDANCE.get(role, "Provide stakeholder-specific revenue-management guidance.")
    return (
        "You are a hotel revenue-management decision support copilot. "
        "You do not predict outcomes yourself. "
        "You interpret model outputs, SHAP evidence, and retrieved textbook excerpts. "
        "Separate observed model signals from your RM interpretation. "
        "Do not invent evidence. "
        "Keep the answer compact, executive-friendly, and action-oriented. "
        "Prefer short sentences. Avoid long paragraphs. "
        "Do not restate every metric unless necessary. "
        "Prioritize intervention guidance over explanation detail. "
        f"Stakeholder emphasis: {role_guidance}"
    )


def build_user_prompt(summary_payload: dict, retrieved_chunks: list[dict]) -> str:
    evidence_lines = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        chapter = chunk.get("chapter", "Unknown")
        heading = chunk.get("heading", "Unknown")
        text = chunk.get("text", "").strip()[:320]
        evidence_lines.append(f"[Source {idx}] chapter={chapter} heading={heading}\n{text}")

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No retrieved textbook context."

    return f"""
Observed dashboard summary:
{summary_payload}

Retrieved textbook evidence:
{evidence_block}

Write a very compact dashboard brief using exactly these sections:
Recommended interventions:
- 3 bullets maximum.
- Put the most important action first.
- Each bullet must be concrete and immediately usable.

Summary:
- 2 sentences maximum.

Why it matters:
- 1 sentence only.

Source grounding:
- 1 short line.
- Mention only chapter/page style grounding, not long excerpts.

Do not dump the retrieved text.
Do not include more than 140 words total.
""".strip()
