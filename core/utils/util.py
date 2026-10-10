# -*- coding: utf-8 -*-
"""Query helpers shared by the agent workflow."""


def concat_feature_descriptions(description, raw_text=""):
    """
    Concat feature descriptions for retrieval query.
    Extracts key procurement points in Thai and combines with original query text for optimal BM25/Dense matching.
    """
    if isinstance(description, str):
        return description.strip()

    parts = []
    # Primary procurement issues
    criminal_acts = description.get("criminal_acts", [])
    if criminal_acts:
        parts.append("ประเด็น: " + ", ".join(str(x) for x in criminal_acts))

    # Procurement object, items, TOR, or budget
    victim_property = description.get("victim_property_details", [])
    if victim_property:
        parts.append("พัสดุ/ขอบเขต: " + ", ".join(str(x) for x in victim_property))

    # Intent, conditions, or exceptions
    intent = description.get("intent_remorse", [])
    if intent:
        parts.append("เงื่อนไข: " + ", ".join(str(x) for x in intent))

    # Defendant / entity info
    defendant = description.get("defendant_info", [])
    if defendant:
        parts.append("หน่วยงาน/ผู้เกี่ยวข้อง: " + ", ".join(str(x) for x in defendant))

    feature_summary = " ".join(parts).strip()
    # If original inquiry text is available, prioritize it directly as query
    # to avoid BM25 term frequency dilution from repetitive metadata labels
    if raw_text and raw_text.strip():
        return raw_text.strip()
    return feature_summary if feature_summary else str(raw_text)
