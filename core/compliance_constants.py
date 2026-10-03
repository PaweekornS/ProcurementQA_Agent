# -*- coding: utf-8 -*-
"""Statutory monetary thresholds shared by /verify, the guardrail and the thresholds resource.

Keep every ceiling here so /qa and /verify cannot drift apart. Values are overridable via env
so a legal change does not need a code change.
"""

import os

# General ceiling for วิธีเฉพาะเจาะจง (มาตรา 56 (2) (ข)). Special cases (e.g. MoE schools under
# กฎกระทรวงพัสดุที่รัฐต้องการส่งเสริมหรือสนับสนุน พ.ศ. 2563 ข้อ ๑๐ (๒): ไม่เกินหนึ่งล้านบาท) are exceptions, not the rule.
SPECIFIC_METHOD_CEILING_THB = int(os.getenv("SPECIFIC_METHOD_CEILING_THB", "500000"))

# Words that show an answer is describing a special-case ceiling rather than the general rule
SPECIAL_CASE_QUALIFIERS = ("สถานศึกษา", "กระทรวงศึกษาธิการ", "ส่งเสริมหรือสนับสนุน", "ส่งเสริมการเรียนการสอน")


def fmt_thb(amount: float) -> str:
    return f"{amount:,.0f}"
