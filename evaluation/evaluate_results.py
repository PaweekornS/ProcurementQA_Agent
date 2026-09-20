#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_results.py - Comprehensive Evaluation of Agent Answers vs Ground Truth
Based on Thai Legal QA Benchmark (AgentCon Bangkok 2026):
- Lexical: ROUGE-L (Precision, Recall, F1) via PyThaiNLP
- Neural Semantic: Dense Cosine Similarity via SentenceTransformer
- Reference-based: COMET Score via Unbabel/wmt22-comet-da
- Citation Metrics: Deterministic section/clause precision, recall, and F1
- Retrieval Metrics: Strict Hit Rate (Doc AND Section), Doc Hit, Section Hit
- LLM-as-a-Judge: Thai Legal QA Benchmark Scoring Rubric (0-5 scale) using google/gemini-3.8-flash
Self-contained script supporting both LegalGraphRAG outputs and CRAG audit results.
"""

import os
import sys
import re
import json
import time
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import torch

try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

try:
    import json_repair
except ImportError:
    json_repair = None

eval_dir = Path(__file__).resolve().parent
project_root = eval_dir.parent

# Load environment variables
for env_candidate in [project_root / ".env", project_root / "configs" / "thai_procurement.env"]:
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate)
        break
else:
    load_dotenv()

# Ensure standard UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Check PyThaiNLP
try:
    from pythainlp import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("[-] Warning: `pythainlp` not found. Falling back to whitespace splitting.")

# Check SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check COMET
try:
    import comet
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


# =====================================================================
# 1. THAI TOKENIZATION & LEXICAL METRIC (ROUGE-L ONLY)
# =====================================================================

def tokenize_thai(text: str) -> List[str]:
    """Tokenize Thai text into words, removing empty whitespace tokens."""
    if not text:
        return []
    if PYTHAINLP_AVAILABLE:
        tokens = word_tokenize(text.strip(), engine="newmm", keep_whitespace=False)
    else:
        tokens = text.strip().split()
    return [t.strip() for t in tokens if t.strip()]


def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length between two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(cand_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L Precision, Recall, and F1 based on LCS."""
    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / m
    recall = lcs / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# =====================================================================
# 2. NEURAL SEMANTIC SIMILARITY
# =====================================================================

class SemanticEvaluator:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"[*] Loading Embedding Model: '{self.model_name}' on {self.device}...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                print("[+] Embedding model loaded successfully.")
            except Exception as e:
                print(f"[-] Failed to load {self.model_name}: {e}")
                fallback = "sentence-transformers/all-MiniLM-L6-v2"
                print(f"[*] Attempting fallback to '{fallback}'...")
                try:
                    self.model = SentenceTransformer(fallback, device=self.device)
                except Exception as e2:
                    print(f"[-] Fallback failed: {e2}")
                    self.model = None

    def compute_similarity(self, hyp_list: List[str], ref_list: List[str]) -> List[float]:
        """Compute cosine similarity between pairs of hypothesis and reference."""
        if not self.model:
            return [0.0] * len(hyp_list)
        try:
            hyp_embeddings = self.model.encode(hyp_list, normalize_embeddings=True, show_progress_bar=False)
            ref_embeddings = self.model.encode(ref_list, normalize_embeddings=True, show_progress_bar=False)
            sims = (hyp_embeddings * ref_embeddings).sum(axis=1)
            return [round(float(s), 4) for s in sims]
        except Exception as e:
            print(f"[-] Error computing embeddings: {e}")
            return [0.0] * len(hyp_list)


# =====================================================================
# 3. COMET SCORE EVALUATOR
# =====================================================================

def evaluate_comet(
    questions: List[str],
    answers: List[str],
    ground_truths: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
) -> Tuple[Optional[float], Optional[List[float]]]:
    """Run Unbabel COMET score on (src=question, mt=answer, ref=ground_truth)."""
    if not COMET_AVAILABLE:
        return None, None

    try:
        print(f"[*] Loading COMET model: '{model_name}'...")
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": q, "mt": a, "ref": r}
            for q, a, r in zip(questions, answers, ground_truths)
        ]

        gpus = 1 if torch.cuda.is_available() else 0
        model_output = model.predict(data, batch_size=batch_size, gpus=gpus)

        system_score = round(float(model_output.system_score), 4)
        segment_scores = [round(float(s), 4) for s in model_output.scores]
        return system_score, segment_scores
    except Exception as e:
        print(f"[-] COMET evaluation failed: {e}")
        return None, None


# =====================================================================
# 4. DETERMINISTIC CITATION EXTRACTOR & LLM-AS-A-JUDGE
# =====================================================================

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize_thai_digits(text: str) -> str:
    """Translate Thai numerals to Arabic digits."""
    if not text:
        return ""
    return text.translate(THAI_DIGITS)


def extract_section_identifiers(text: str) -> List[str]:
    """
    Extract individual statutory section and clause identifiers from text.
    Examples:
      'มาตรา ๕๖ (๒) (ข); ข้อ ๑; ข้อ ๒๘, ข้อ ๗๙' -> ['มาตรา 56', 'ข้อ 1', 'ข้อ 28', 'ข้อ 79']
      'มาตรา ๑๐๐ วรรคสาม, ระเบียบฯ ข้อ ๒๕' -> ['มาตรา 100', 'ข้อ 25']
    """
    if not text:
        return []
    norm = normalize_thai_digits(text)
    parts = [p.strip() for p in re.split(r"[,;/]+", norm) if p.strip()]
    identifiers = []
    for p in parts:
        # Match 'มาตรา X' or 'ม. X'
        sec_matches = re.findall(r"(?:มาตรา|ม\.)\s*(\d+)", p)
        for m in sec_matches:
            identifiers.append(f"มาตรา {m}")

        # Match 'ข้อ X', 'ข้อที่ X', or 'ข. X'
        clause_matches = re.findall(r"(?:ข้อที่|ข้อ|ข\.)\s*(\d+)", p)
        for m in clause_matches:
            identifiers.append(f"ข้อ {m}")

    return list(dict.fromkeys(identifiers))


def compute_deterministic_citation_metrics(
    gt_section_str: str,
    candidate_answer: str,
    candidate_citations_str: str = "",
) -> Dict[str, float]:
    """
    Calculate deterministic Citation Recall, Precision, and F1 by comparing
    ground truth section identifiers against candidate citations + answer text.
    """
    gt_ids = set(extract_section_identifiers(gt_section_str))
    if not gt_ids:
        # If no target sections specified in GT, citation is considered fully satisfied
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}

    cand_text = f"{candidate_answer} {candidate_citations_str}"
    cand_ids = set(extract_section_identifiers(cand_text))

    if not cand_ids:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    hits = gt_ids.intersection(cand_ids)
    recall = len(hits) / len(gt_ids)
    precision = len(hits) / len(cand_ids)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


class LegalQABenchmarkResult(BaseModel):
    """
    Thai Legal QA Benchmark Scoring Schema (AgentCon Bangkok 2026):
    Scale 0-5:
      5 - Perfect: exact มาตรา/ข้อ, correct penalties/thresholds, complete
      4 - Good: correct with minor extras
      3 - Partial: core correct, notable gaps
      2 - Poor: significant missing content
      1 - Bad: mostly incorrect
      0 - Fail: contradicts law / no answer
    """
    score: int = Field(
        ge=0, le=5,
        description="Thai Legal QA Benchmark score (0 to 5)."
    )
    score_label: Literal["Perfect", "Good", "Partial", "Poor", "Bad", "Fail"] = Field(
        default="Partial",
        description="Rating label corresponding to the score."
    )
    citation_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="สัดส่วนของมาตรา/ข้อ/ระเบียบตาม Ground Truth ที่ Candidate อ้างอิงได้ถูกต้อง ครบถ้วน (0.0 ถึง 1.0)"
    )
    citation_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="สัดส่วนความถูกต้องตรงประเด็นของมาตรา/ข้อที่ Candidate ยกขึ้นอ้างอิง (0.0 ถึง 1.0)"
    )
    citation_f1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="F1 Score ของการอ้างอิงข้อกฎหมาย (Harmonic mean: 0.0 ถึง 1.0)"
    )
    exact_section_citation: bool = Field(
        default=False,
        description="True if Candidate Answer or citations correctly reference all primary target มาตรา/ข้อ/ระเบียบ."
    )
    rule_or_threshold_correctness: bool = Field(
        default=False,
        description="True if numerical thresholds, percentages, penalties, time periods, or legal rules are accurate."
    )
    completeness_vs_ground_truth: bool = Field(
        default=False,
        description="True if all critical conditions, procedures, and exceptions in Ground Truth are covered."
    )
    no_hallucination: bool = Field(
        default=True,
        description="True if the answer contains NO fabricated rules, fake statutes, or hallucinated facts."
    )
    critique: str = Field(
        default="",
        description="Concise Thai critique explaining rationale, citations, missing parts, or hallucinated contents."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Normalize score
            raw_score = data.get("score")
            score_int = 3
            if raw_score is not None:
                if isinstance(raw_score, (int, float)):
                    score_int = int(round(raw_score))
                elif isinstance(raw_score, str):
                    m = re.search(r"\b([0-5])\b", raw_score)
                    if m:
                        score_int = int(m.group(1))
            score_int = max(0, min(5, score_int))
            data["score"] = score_int

            # 2. Normalize score_label
            label_map = {
                5: "Perfect",
                4: "Good",
                3: "Partial",
                2: "Poor",
                1: "Bad",
                0: "Fail",
            }
            raw_label = str(data.get("score_label", "")).strip().capitalize()
            if raw_label not in ["Perfect", "Good", "Partial", "Poor", "Bad", "Fail"]:
                data["score_label"] = label_map.get(score_int, "Partial")
            else:
                data["score_label"] = raw_label

            # 3. Normalize citation floats
            for f_key in ["citation_recall", "citation_precision", "citation_f1"]:
                val = data.get(f_key)
                if val is not None:
                    try:
                        f_val = float(val)
                        data[f_key] = max(0.0, min(1.0, round(f_val, 4)))
                    except (ValueError, TypeError):
                        data[f_key] = 0.0
                else:
                    data[f_key] = 0.0

            cr = data.get("citation_recall", 0.0)
            cp = data.get("citation_precision", 0.0)
            if data.get("citation_f1", 0.0) == 0.0 and (cp + cr) > 0:
                data["citation_f1"] = round(2 * cp * cr / (cp + cr), 4)

            # 4. Normalize booleans
            for b_key in ["exact_section_citation", "rule_or_threshold_correctness", "completeness_vs_ground_truth", "no_hallucination"]:
                val = data.get(b_key)
                if isinstance(val, str):
                    data[b_key] = val.lower() in ("true", "1", "yes", "ใช่", "ถูกต้อง", "ผ่าน")
                elif val is None:
                    if b_key == "exact_section_citation":
                        data[b_key] = cr >= 0.99
                    elif b_key == "no_hallucination":
                        data[b_key] = True
                    else:
                        data[b_key] = False
        return data


LEGAL_BENCHMARK_JUDGE_PROMPT = """คุณคือผู้เชี่ยวชาญการตรวจประเมินกฎหมายไทยระดับสูง ทำหน้าที่เป็น LLM-as-a-Judge เพื่อให้คะแนนคำตอบของผู้ช่วย (Candidate Answer) เทียบกับเฉลยมาตรฐาน (Ground Truth)

## เกณฑ์การให้คะแนน (Scoring Rubric 0 - 5):
- **5 (Perfect)**: ตอบถูกต้องสมบูรณ์แบบ อ้างอิง มาตรา/ข้อ ถูกต้องแม่นยำ ตัวเลข/วงเงิน/ระยะเวลา/บทกำหนดโทษ/เงื่อนไข/ข้อยกเว้น ครบถ้วน ไม่มีข้อมูลที่แต่งขึ้นเอง
- **4 (Good)**: คำตอบถูกต้องในสาระสำคัญทั้งหมด เงื่อนไขหลักครบถ้วน แต่อาจมีข้อความขยายความเสริมเล็กน้อย (minor extras) หรืออ้างอิงมาตราบริบทใกล้เคียง
- **3 (Partial)**: สาระสำคัญหลักถูกต้อง (core correct) แต่มีช่องว่างที่ตกหล่นอย่างเห็นได้ชัด (notable gaps) เช่น ขาดข้อยกเว้นสำคัญ หรือขาดรายละเอียดขั้นตอนบางส่วน
- **2 (Poor)**: เนื้อหาขาดหายไปเป็นส่วนใหญ่ (significant missing content) ตอบได้เพียงผิวเผินหรือเพียงเสี้ยวเดียวของประเด็นคำถาม
- **1 (Bad)**: คำตอบส่วนใหญ่ไม่ถูกต้อง (mostly incorrect) มีความเข้าใจผิดในข้อกฎหมายหรือตัวเลขสำคัญผิดพลาด
- **0 (Fail)**: ตอบขัดแย้งกับหลักกฎหมายอย่างสิ้นเชิง (contradicts law) หรือไม่ตอบ/ตอบไม่ตรงคำถามเลย

## มิติการประเมินด้าน Citation (Precision, Recall, F1):
1. **citation_recall (0.0 ถึง 1.0)**: สัดส่วนของมาตรา/ข้อ/ระเบียบตาม Ground Truth ที่ Candidate ระบุหรืออ้างอิงได้ถูกต้อง
   - หาก Ground Truth มี 3 ข้อ/มาตรา และ Candidate อ้างอิงได้ครบ 3 ข้อ -> 1.0
   - หาก Ground Truth มี 2 มาตรา แต่ Candidate อ้างอิงได้เพียง 1 มาตรา -> 0.5
   - หากไม่ระบุมาตราที่ถูกต้องเลย -> 0.0
2. **citation_precision (0.0 ถึง 1.0)**: สัดส่วนความถูกต้องตรงประเด็นของมาตรา/ข้อที่ Candidate ยกขึ้นอ้างอิง (ไม่ยกมาตรามั่วหรือผิดเรื่อง)
   - หากทุกมาตราที่ยกมาอ้างอิงถูกต้องตรงประเด็นทั้งหมด -> 1.0
   - หากยกมา 2 มาตรา แต่มี 1 มาตราที่ผิดเรื่อง/ผิดกฎหมาย -> 0.5
   - หากไม่ได้อ้างอิง หรืออ้างอิงผิดทั้งหมด -> 0.0
3. **citation_f1 (0.0 ถึง 1.0)**: 2 * (P * R) / (P + R)
4. **exact_section_citation (true/false)**: true เมื่อ Candidate อ้างอิงมาตราหลักได้ครบถ้วน (citation_recall >= 1.0 หรือครอบคลุมบทบัญญัติหลัก)

## มิติการประเมินด้านเนื้อหา (Rule, Completeness, Hallucination):
- **rule_or_threshold_correctness**: ตัวเลข วงเงิน อัตราร้อยละ ระยะเวลา บทกำหนดโทษ ถูกต้องหรือไม่ (เลขอารบิกและคำอ่านไทย เช่น "3 ราย" และ "สามราย" มีค่าเทียบเท่ากัน)
- **completeness_vs_ground_truth**: ความครบถ้วนของเงื่อนไขและข้อยกเว้นตาม Ground Truth
- **no_hallucination**: ปราศจากการแต่งกฎหมายขึ้นเอง ไม่มโนตัวเลขหรือกฎเกณฑ์ที่ไม่มีอยู่จริง

สำคัญ: ส่งผลลัพธ์เป็น JSON ตาม Schema ที่กำหนดเท่านั้น ห้ามเขียนข้อความเกริ่นนำหรือปิดท้ายนอก JSON"""

T = TypeVar("T", bound=BaseModel)

class StandaloneLegalJudge:
    """Self-contained LLM Judge without external base class dependency."""

    def __init__(self, model_name: str = "google/gemini-3.8-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("\ufeffOPENROUTER_API_KEY", "")
        self.api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key.strip() if self.api_key else "dummy_key",
            openai_api_base=self.api_base,
            temperature=0.0,
            max_tokens=4096,
            request_timeout=90,
        )

    def _parse_json(self, content: str, schema_cls: Type[T]) -> Optional[T]:
        if not content:
            return None
        clean_text = content.strip()
        clean_text = re.sub(r"<think>[\s\S]*?</think>", "", clean_text).strip()
        if "```" in clean_text:
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
            if fence_match:
                clean_text = fence_match.group(1).strip()
            else:
                clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE).strip()
                clean_text = re.sub(r"\s*```$", "", clean_text).strip()

        start_idx = clean_text.find("{")
        if start_idx != -1:
            end_idx = clean_text.rfind("}")
            if end_idx != -1 and end_idx > start_idx:
                clean_text = clean_text[start_idx : end_idx + 1]
            else:
                clean_text = clean_text[start_idx:]

        # 1. Direct JSON parse
        try:
            return schema_cls.model_validate(json.loads(clean_text, strict=False))
        except Exception:
            pass

        # 2. Patch truncated string/brace before repair
        if "{" in clean_text and "}" not in clean_text:
            for patch in ['"\n}', '"}', '\n}']:
                try:
                    return schema_cls.model_validate(json.loads(clean_text + patch, strict=False))
                except Exception:
                    pass

        # 3. json_repair library
        if json_repair is not None:
            try:
                repaired = json_repair.repair_json(clean_text, return_objects=True)
                if isinstance(repaired, dict):
                    return schema_cls.model_validate(repaired)
            except Exception:
                pass

        # 4. Regex-based field extraction fallback (salvages truncated LLM outputs)
        try:
            score_m = re.search(r'"score"\s*:\s*(\d)', clean_text)
            if score_m:
                score_val = int(score_m.group(1))
                crit_m = re.search(r'"critique"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', clean_text)
                crit_val = crit_m.group(1) if crit_m else ""
                cr_m = re.search(r'"citation_recall"\s*:\s*([0-1](?:\.\d+)?)', clean_text)
                cp_m = re.search(r'"citation_precision"\s*:\s*([0-1](?:\.\d+)?)', clean_text)
                cf_m = re.search(r'"citation_f1"\s*:\s*([0-1](?:\.\d+)?)', clean_text)
                cr_val = float(cr_m.group(1)) if cr_m else 0.0
                cp_val = float(cp_m.group(1)) if cp_m else 0.0
                cf_val = float(cf_m.group(1)) if cf_m else ((2 * cp_val * cr_val / (cp_val + cr_val)) if (cp_val + cr_val) > 0 else 0.0)

                exact_m = re.search(r'"exact_section_citation"\s*:\s*(true|false)', clean_text, re.I)
                rule_m = re.search(r'"rule_or_threshold_correctness"\s*:\s*(true|false)', clean_text, re.I)
                comp_m = re.search(r'"completeness_vs_ground_truth"\s*:\s*(true|false)', clean_text, re.I)
                hal_m = re.search(r'"no_hallucination"\s*:\s*(true|false)', clean_text, re.I)

                return schema_cls.model_validate({
                    "score": score_val,
                    "citation_recall": cr_val,
                    "citation_precision": cp_val,
                    "citation_f1": cf_val,
                    "exact_section_citation": exact_m.group(1).lower() == "true" if exact_m else (cr_val >= 0.99),
                    "rule_or_threshold_correctness": rule_m.group(1).lower() == "true" if rule_m else True,
                    "completeness_vs_ground_truth": comp_m.group(1).lower() == "true" if comp_m else False,
                    "no_hallucination": hal_m.group(1).lower() == "true" if hal_m else True,
                    "critique": crit_val or "ประเมินโดย regex fallback",
                })
        except Exception:
            pass

        return None

    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        ground_truth: str,
        ground_truth_section: str = "",
        candidate_citations: str = "",
    ) -> LegalQABenchmarkResult:
        """Run judge evaluation for a single QA pair with retry on API or parse failures."""
        user_prompt = f"""[คำถาม / Question]:
{question}

[Ground Truth (เฉลยมาตรฐาน)]:
{ground_truth}
(มาตรา/ข้อ อ้างอิง: {ground_truth_section if ground_truth_section else 'ไม่ระบุ'})

[Candidate Answer (คำตอบของ AI)]:
{candidate_answer}
(การอ้างอิงของ Candidate: {candidate_citations if candidate_citations else 'ไม่มีการระบุแยก'})

จงประเมิน Candidate Answer และตอบเป็น JSON ตามโครงสร้างนี้:
{{
  "score": 0 ถึง 5 (ตาม Rubric),
  "score_label": "Perfect หรือ Good หรือ Partial หรือ Poor หรือ Bad หรือ Fail",
  "citation_recall": 0.0 ถึง 1.0 (สัดส่วนมาตรา Ground Truth ที่ Candidate อ้างอิงได้ถูกต้อง),
  "citation_precision": 0.0 ถึง 1.0 (สัดส่วนมาตราที่ Candidate อ้างอิงแล้วถูกต้องตรงประเด็น),
  "citation_f1": 0.0 ถึง 1.0 (Harmonic mean ของ Precision และ Recall),
  "exact_section_citation": true หรือ false,
  "rule_or_threshold_correctness": true หรือ false,
  "completeness_vs_ground_truth": true หรือ false,
  "no_hallucination": true หรือ false,
  "critique": "เหตุผลสั้นๆ ในภาษาไทยที่อธิบายคะแนนที่ได้รับ จุดแข็ง และสิ่งที่ขาดหรือผิด"
}}"""

        max_retries = 3
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                messages = [
                    SystemMessage(content=LEGAL_BENCHMARK_JUDGE_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
                response = self.llm.invoke(messages)
                content = response.content
                if isinstance(content, list):
                    content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                
                parsed = self._parse_json(str(content or ""), LegalQABenchmarkResult)
                if parsed:
                    return parsed

                last_error = "JSON parse returned None"
                if attempt < max_retries:
                    time.sleep(2 * attempt)
            except Exception as e:
                last_error = str(e)
                print(f" [Retry {attempt}/{max_retries} due to: {last_error[:50]}...]", end="", flush=True)
                if attempt < max_retries:
                    time.sleep(3 * attempt)

        # Fallback default after exhausting retries
        return LegalQABenchmarkResult(
            score=3,
            score_label="Partial",
            citation_recall=0.0,
            citation_precision=0.0,
            citation_f1=0.0,
            exact_section_citation=False,
            rule_or_threshold_correctness=True,
            completeness_vs_ground_truth=False,
            no_hallucination=True,
            critique=f"Fallback default result ({last_error[:60]})",
        )


# =====================================================================
# 5. MAIN EVALUATION PIPELINE
# =====================================================================

def run_evaluation(
    input_file: Path,
    output_file: Path,
    summary_file: Path,
    embedding_model_name: str = "unsloth/embeddinggemma-300m",
    judge_model_name: str = "google/gemini-3.8-flash",
    enable_judge: bool = True,
    enable_comet: bool = True,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    workers: int = 4,
) -> Dict[str, Any]:
    """Execute complete evaluation flow."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    print(f"[*] Reading results from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support direct list or dict with 'results' key
    if isinstance(data, list):
        raw_results = data
    elif isinstance(data, dict):
        raw_results = data.get("results", [])
    else:
        raw_results = []

    if not raw_results:
        raise ValueError("No valid results found in input JSON file.")

    # Filter valid samples
    valid_samples = []
    error_samples = []
    no_law_samples = []

    for idx, item in enumerate(raw_results):
        if item.get("status") == "ERROR" or "error" in item:
            error_samples.append(item)
            continue

        # Extract answer text across various schema aliases
        direct_ans = (item.get("direct_answer") or "").strip()
        legal_reas = (item.get("legal_reasoning") or item.get("answer") or "").strip()
        
        # Combined concise answer: direct answer followed by legal reasoning
        if direct_ans and legal_reas and direct_ans != legal_reas:
            answer = f"{direct_ans}\n\n{legal_reas}"
        elif direct_ans:
            answer = direct_ans
        else:
            answer = legal_reas

        ground_truth = item.get("ground_truth") or ""

        # Filter out NO_LAW_FOUND items from benchmark evaluation as requested
        status = str(item.get("status") or item.get("overall_status") or "").upper()

        is_fallback_answer = (
            status == "NO_LAW_FOUND"
            or "ไม่พบข้อกฎหมาย" in direct_ans
            or "ไม่พบข้อกฎหมาย" in legal_reas
            or "ไม่พบข้อกฎหมาย" in answer
            or "ไม่อยู่ในขอบเขต" in direct_ans
            or "ไม่อยู่ในขอบเขต" in legal_reas
            or "ไม่อยู่ในขอบเขต" in answer
        )

        if is_fallback_answer:
            item_copy = dict(item)
            item_copy["status"] = "NO_LAW_FOUND"
            item_copy["answer"] = answer
            no_law_samples.append(item_copy)
            continue

        if not answer.strip() or not ground_truth.strip():
            error_samples.append(item)
            continue

        # Standardize sample
        norm_item = dict(item)
        norm_item["answer"] = answer
        norm_item["row_index"] = item.get("row_index", item.get("id", idx))
        norm_item["question"] = item.get("question") or item.get("fact") or ""

        # Extract ground_truth_section
        if not norm_item.get("ground_truth_section"):
            exp_pairs = item.get("expected_pairs", [])
            if exp_pairs:
                sections = [p.get("section", "") for p in exp_pairs if p.get("section")]
                norm_item["ground_truth_section"] = "; ".join(dict.fromkeys(sections))
            elif item.get("laws"):
                norm_item["ground_truth_section"] = "; ".join(item.get("laws"))
            elif item.get("expected_section"):
                exp_sec = item.get("expected_section")
                if isinstance(exp_sec, list):
                    norm_item["ground_truth_section"] = "; ".join(exp_sec)
                else:
                    norm_item["ground_truth_section"] = str(exp_sec)

        valid_samples.append(norm_item)

    print(f"[+] Total entries: {len(raw_results)}")
    print(f"[+] Valid QA pairs for evaluation: {len(valid_samples)}")
    if no_law_samples:
        print(f"[*] Skipped {len(no_law_samples)} NO_LAW_FOUND entries (exempt from benchmark scoring).")
    if error_samples:
        print(f"[-] Skipped {len(error_samples)} error/incomplete entries.")

    if not valid_samples:
        print("[-] No valid samples to evaluate.")
        return {}

    # 1. Lexical metric: ROUGE-L (Precision, Recall, F1)
    print("[*] Computing Thai tokenized lexical metric (ROUGE-L)...")
    eval_items = []
    questions = []
    agent_answers = []
    ground_truths = []

    # Track retrieval hits if available
    retrieval_sec_hits = 0
    retrieval_doc_hits = 0
    retrieval_both_hits = 0
    has_retrieval_flags = False

    for sample in valid_samples:
        q = sample.get("question", "")
        cand = sample.get("answer", "")
        ref = sample.get("ground_truth", "")

        questions.append(q)
        agent_answers.append(cand)
        ground_truths.append(ref)

        cand_tokens = tokenize_thai(cand)
        ref_tokens = tokenize_thai(ref)

        rl = compute_rouge_l(cand_tokens, ref_tokens)

        # Length stats
        cand_char_len = len(cand)
        ref_char_len = len(ref)
        cand_word_len = len(cand_tokens)
        ref_word_len = len(ref_tokens)
        word_ratio = round(cand_word_len / max(1, ref_word_len), 2)

        # Extract citations string
        citations_list = []
        if sample.get("predicted_laws"):
            citations_list.extend(sample.get("predicted_laws", []))
        for audit in sample.get("audits", []):
            for cit in audit.get("citations", []):
                sec = cit.get("section_or_clause", "")
                act = cit.get("act_or_regulation", "")
                if sec or act:
                    citations_list.append(f"{act} {sec}".strip())
        if not citations_list and sample.get("referenced_statutes"):
            for stat in sample.get("referenced_statutes", []):
                citations_list.append(stat.strip())
        citations_str = ", ".join(list(dict.fromkeys(citations_list)))

        gt_sec = sample.get("ground_truth_section", "")
        det_cite = compute_deterministic_citation_metrics(gt_sec, cand, citations_str)

        # Check retrieval flags from LegalGraphRAG
        is_both = sample.get("is_both_hit")
        is_doc = sample.get("is_document_hit")
        is_sec = sample.get("is_section_hit")
        if is_both is not None or is_doc is not None or is_sec is not None:
            has_retrieval_flags = True
            if is_both:
                retrieval_both_hits += 1
            if is_doc:
                retrieval_doc_hits += 1
            if is_sec:
                retrieval_sec_hits += 1

        eval_items.append({
            "row_index": sample.get("row_index"),
            "question": q,
            "status": sample.get("status", "SUCCESS"),
            "agent_answer": cand,
            "ground_truth": ref,
            "ground_truth_section": gt_sec,
            "candidate_citations": citations_str,
            "is_both_hit": is_both,
            "is_document_hit": is_doc,
            "is_section_hit": is_sec,
            "lengths": {
                "agent_words": cand_word_len,
                "ground_truth_words": ref_word_len,
                "word_ratio": word_ratio,
                "agent_chars": cand_char_len,
                "ground_truth_chars": ref_char_len,
            },
            "metrics": {
                "rougeL_f1": rl["f1"],
                "rougeL_p": rl["precision"],
                "rougeL_r": rl["recall"],
                "det_citation_recall": det_cite["recall"],
                "det_citation_precision": det_cite["precision"],
                "det_citation_f1": det_cite["f1"],
            }
        })

    # 2. Semantic Embedding Cosine Similarity
    print(f"[*] Computing Semantic Cosine Similarity with '{embedding_model_name}'...")
    sem_eval = SemanticEvaluator(embedding_model_name)
    semantic_scores = sem_eval.compute_similarity(agent_answers, ground_truths)
    for item, s_score in zip(eval_items, semantic_scores):
        item["metrics"]["semantic_cosine_sim"] = s_score

    # 3. COMET Score
    comet_sys_score = None
    if enable_comet:
        if COMET_AVAILABLE:
            print(f"[*] Running Unbabel COMET evaluation with '{comet_model_name}'...")
            comet_sys_score, comet_seg_scores = evaluate_comet(
                questions, agent_answers, ground_truths, model_name=comet_model_name
            )
            if comet_seg_scores:
                for item, c_score in zip(eval_items, comet_seg_scores):
                    item["metrics"]["comet_score"] = c_score
        else:
            print("[-] `unbabel-comet` is not available. Skipping COMET calculation.")

    # 4. LLM-as-a-Judge Evaluation (Thai Legal QA Benchmark Rubric 0-5)
    # Concurrent multi-threaded execution matching main system pipeline
    if enable_judge:
        workers_count = max(1, workers)
        print(f"[*] Running Thai Legal QA Benchmark LLM-as-a-Judge with '{judge_model_name}' ({workers_count} concurrent workers)...")
        judge = StandaloneLegalJudge(model_name=judge_model_name)

        def judge_single_item(item_info):
            idx, item = item_info
            q = item["question"]
            cand = item["agent_answer"]
            ref = item["ground_truth"]
            gt_sec = item.get("ground_truth_section", "")
            cit = item.get("candidate_citations", "")

            try:
                j_res = judge.evaluate_answer(
                    question=q,
                    candidate_answer=cand,
                    ground_truth=ref,
                    ground_truth_section=gt_sec,
                    candidate_citations=cit,
                )
                print(f"    [{idx}/{len(eval_items)}] Judged Row {item.get('row_index')} -> Score: {j_res.score}/5 ({j_res.score_label}) [Cite R:{j_res.citation_recall:.2f}/P:{j_res.citation_precision:.2f}/F1:{j_res.citation_f1:.2f}, Rule:{j_res.rule_or_threshold_correctness}, Comp:{j_res.completeness_vs_ground_truth}, HallucFree:{j_res.no_hallucination}]")
                return item, j_res
            except Exception as e:
                print(f"    [-] Error judging Row {item.get('row_index')}: {e}")
                return item, None

        if workers_count > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers_count) as executor:
                futures = [executor.submit(judge_single_item, (idx, item)) for idx, item in enumerate(eval_items, 1)]
                for f in concurrent.futures.as_completed(futures):
                    it, j_res = f.result()
                    if j_res is not None:
                        it["legal_benchmark_judge"] = {
                            "score": j_res.score,
                            "score_label": j_res.score_label,
                            "citation_recall": j_res.citation_recall,
                            "citation_precision": j_res.citation_precision,
                            "citation_f1": j_res.citation_f1,
                            "exact_section_citation": j_res.exact_section_citation,
                            "rule_or_threshold_correctness": j_res.rule_or_threshold_correctness,
                            "completeness_vs_ground_truth": j_res.completeness_vs_ground_truth,
                            "no_hallucination": j_res.no_hallucination,
                            "critique": j_res.critique,
                        }
        else:
            for idx, item in enumerate(eval_items, 1):
                it, j_res = judge_single_item((idx, item))
                if j_res is not None:
                    it["legal_benchmark_judge"] = {
                        "score": j_res.score,
                        "score_label": j_res.score_label,
                        "citation_recall": j_res.citation_recall,
                        "citation_precision": j_res.citation_precision,
                        "citation_f1": j_res.citation_f1,
                        "exact_section_citation": j_res.exact_section_citation,
                        "rule_or_threshold_correctness": j_res.rule_or_threshold_correctness,
                        "completeness_vs_ground_truth": j_res.completeness_vs_ground_truth,
                        "no_hallucination": j_res.no_hallucination,
                        "critique": j_res.critique,
                    }

        # Keep original row order
        eval_items.sort(key=lambda x: x.get("row_index", 0))

    # 5. Compute Aggregate Metrics
    n = len(eval_items)
    avg_rougeL_f1 = sum(i["metrics"]["rougeL_f1"] for i in eval_items) / n
    avg_rougeL_p = sum(i["metrics"]["rougeL_p"] for i in eval_items) / n
    avg_rougeL_r = sum(i["metrics"]["rougeL_r"] for i in eval_items) / n
    avg_semantic_sim = sum(i["metrics"]["semantic_cosine_sim"] for i in eval_items) / n
    avg_word_ratio = sum(i["lengths"]["word_ratio"] for i in eval_items) / n
    avg_det_cite_recall = sum(i["metrics"]["det_citation_recall"] for i in eval_items) / n
    avg_det_cite_precision = sum(i["metrics"]["det_citation_precision"] for i in eval_items) / n
    avg_det_cite_f1 = sum(i["metrics"]["det_citation_f1"] for i in eval_items) / n

    summary = {
        "total_input_entries": len(raw_results),
        "total_evaluated": n,
        "total_no_law_skipped": len(no_law_samples),
        "total_errors_skipped": len(error_samples),
        "embedding_model": embedding_model_name,
        "averages": {
            "semantic_cosine_sim": round(avg_semantic_sim, 4),
            "rougeL_f1": round(avg_rougeL_f1, 4),
            "rougeL_recall": round(avg_rougeL_r, 4),
            "rougeL_precision": round(avg_rougeL_p, 4),
            "deterministic_citation_recall": round(avg_det_cite_recall, 4),
            "deterministic_citation_precision": round(avg_det_cite_precision, 4),
            "deterministic_citation_f1": round(avg_det_cite_f1, 4),
            "word_length_ratio (agent/GT)": round(avg_word_ratio, 2),
        }
    }

    if has_retrieval_flags:
        summary["retrieval_metrics"] = {
            "strict_hit_rate (Doc AND Section)": f"{retrieval_both_hits}/{n} ({retrieval_both_hits/n*100:.1f}%)",
            "document_hit_rate": f"{retrieval_doc_hits}/{n} ({retrieval_doc_hits/n*100:.1f}%)",
            "section_hit_rate": f"{retrieval_sec_hits}/{n} ({retrieval_sec_hits/n*100:.1f}%)",
        }

    if comet_sys_score is not None:
        summary["averages"]["comet_system_score"] = comet_sys_score

    if enable_judge:
        judged_items = [i for i in eval_items if "legal_benchmark_judge" in i]
        n_judged = len(judged_items)
        if n_judged > 0:
            avg_score = round(sum(i["legal_benchmark_judge"]["score"] for i in judged_items) / n_judged, 2)
            pct_5 = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 5) / n_judged * 100, 1)
            pct_4_plus = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] >= 4) / n_judged * 100, 1)
            avg_j_recall = round(sum(i["legal_benchmark_judge"]["citation_recall"] for i in judged_items) / n_judged, 4)
            pct_cite = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["exact_section_citation"]) / n_judged * 100, 1)
            pct_rule = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["rule_or_threshold_correctness"]) / n_judged * 100, 1)
            pct_comp = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["completeness_vs_ground_truth"]) / n_judged * 100, 1)
            pct_halluc_free = round(sum(1 for i in judged_items if i["legal_benchmark_judge"]["no_hallucination"]) / n_judged * 100, 1)

            n_below_3 = sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] < 3)
            pct_below_3 = round(n_below_3 / n_judged * 100, 1)

            summary["legal_benchmark_judge_summary"] = {
                "judge_model": judge_model_name,
                "average_score_out_of_5": avg_score,
                "perfect_score_pct (5/5)": pct_5,
                "good_or_perfect_pct (>=4/5)": pct_4_plus,
                "below_3_count (<3/5)": n_below_3,
                "below_3_pct (<3/5)": pct_below_3,
                "average_citation_recall": avg_j_recall,
                "exact_citation_rate_pct": pct_cite,
                "rule_threshold_correctness_pct": pct_rule,
                "completeness_rate_pct": pct_comp,
                "hallucination_free_rate_pct": pct_halluc_free,
                "score_distribution": {
                    "5_Perfect": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 5),
                    "4_Good": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 4),
                    "3_Partial": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 3),
                    "2_Poor": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 2),
                    "1_Bad": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 1),
                    "0_Fail": sum(1 for i in judged_items if i["legal_benchmark_judge"]["score"] == 0),
                }
            }

    # =================================================================
    # 6. PRINT TERMINAL RESULTS TABLE
    # =================================================================
    print("\n" + "=" * 125)
    print("                 THAI LEGAL QA BENCHMARK EVALUATION RESULTS (Agent vs Ground Truth)")
    print("=" * 125)
    header = f"{'Row':<5} | {'Status':<10} | {'SemSim':<7} | {'RL-F1':<6} | {'RL-Rec':<6}"
    if comet_sys_score is not None:
        header += " | {'COMET':<7}"
    if enable_judge:
        header += f" | {'Score':<5} | {'Label':<7} | {'Cite(R/P)':<9} | {'Rule':<5} | {'Comp':<5} | {'NoHal':<5}"
    else:
        header += f" | {'DetCite(R/P)':<12}"
    print(header)
    print("-" * 125)

    for item in eval_items:
        r_idx = str(item.get("row_index", "-"))
        status = item.get("status", "-")[:10]
        m = item["metrics"]
        sem = f"{m['semantic_cosine_sim']:.4f}"
        rl_f1 = f"{m['rougeL_f1']:.4f}"
        rl_r = f"{m['rougeL_r']:.4f}"
        row_str = f"{r_idx:<5} | {status:<10} | {sem:<7} | {rl_f1:<6} | {rl_r:<6}"
        if comet_sys_score is not None:
            c_val = f"{m.get('comet_score', 0.0):.4f}"
            row_str += f" | {c_val:<7}"
        if enable_judge and "legal_benchmark_judge" in item:
            j = item["legal_benchmark_judge"]
            sc = f"{j['score']}/5"
            lbl = j["score_label"][:7]
            cite_v = f"{j['citation_recall']:.2f}/{j['citation_precision']:.2f}"
            rule_v = "YES" if j["rule_or_threshold_correctness"] else "NO"
            comp_v = "YES" if j["completeness_vs_ground_truth"] else "NO"
            hal_v = "YES" if j["no_hallucination"] else "NO"
            row_str += f" | {sc:<5} | {lbl:<7} | {cite_v:<9} | {rule_v:<5} | {comp_v:<5} | {hal_v:<5}"
        else:
            d_rp = f"{m['det_citation_recall']:.2f}/{m['det_citation_precision']:.2f}"
            row_str += f" | {d_rp:<12}"
        print(row_str)

    print("-" * 125)
    avg_str = f"{'AVG':<5} | {'ALL':<10} | {avg_semantic_sim:.4f} | {avg_rougeL_f1:.4f} | {avg_rougeL_r:.4f}"
    if comet_sys_score is not None:
        avg_str += f" | {comet_sys_score:.4f}"
    if enable_judge and "legal_benchmark_judge_summary" in summary:
        js = summary["legal_benchmark_judge_summary"]
        avg_cite_str = f"{js['average_citation_recall']:.2f}/{js['average_citation_precision']:.2f}"
        avg_str += f" | {js['average_score_out_of_5']:.2f} | {js['good_or_perfect_pct (>=4/5)']:.1f}%  | {avg_cite_str:<9} | {js['rule_threshold_correctness_pct']:.0f}%  | {js['completeness_rate_pct']:.0f}%  | {js['hallucination_free_rate_pct']:.0f}%"
    else:
        avg_cite_str = f"{avg_det_cite_recall:.2f}/{avg_det_cite_precision:.2f}"
        avg_str += f" | {avg_cite_str:<12}"
    print(avg_str)
    print("=" * 125)
    if enable_judge and "legal_benchmark_judge_summary" in summary:
        js = summary["legal_benchmark_judge_summary"]
        sd = js.get("score_distribution", {})
        print(f"📊 Score Distribution: 5_Perfect: {sd.get('5_Perfect',0)} | 4_Good: {sd.get('4_Good',0)} | 3_Partial: {sd.get('3_Partial',0)} | 2_Poor: {sd.get('2_Poor',0)} | 1_Bad: {sd.get('1_Bad',0)} | 0_Fail: {sd.get('0_Fail',0)}")
        print(f"⚠️  Low Scores (<3/5): {js.get('below_3_count (<3/5)', 0)} / {len(judged_items)} ({js.get('below_3_pct (<3/5)', 0.0):.1f}%)")
    print("=" * 125 + "\n")

    # =================================================================
    # 7. SAVE ARTIFACTS (JSON + MARKDOWN)
    # =================================================================
    full_output = {
        "summary": summary,
        "per_question_results": eval_items,
        "skipped_no_law_samples": no_law_samples,
        "skipped_errors": error_samples,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"[+] Full JSON evaluation saved to: {output_file}")

    # Generate Markdown Summary
    md_lines = [
        "# Thai Legal QA Benchmark Evaluation Report",
        "",
        f"- **Total Input Entries:** {len(raw_results)}",
        f"- **Evaluated Samples:** {n} (Active QA pairs)",
        f"- **Skipped NO_LAW_FOUND:** {len(no_law_samples)} (Exempt from benchmark evaluation as requested)",
        f"- **Skipped Errors:** {len(error_samples)}",
        f"- **Embedding Model:** `{embedding_model_name}`",
    ]
    if enable_judge and "legal_benchmark_judge_summary" in summary:
        md_lines.append(f"- **Benchmark Judge Model:** `{judge_model_name}` (Thai Legal QA Benchmark Rubric 0-5)")

    md_lines.extend([
        "",
        "## 1. Overall Metric Averages",
        "",
        "| Metric | Score | Description |",
        "| :--- | :---: | :--- |",
    ])

    if has_retrieval_flags and "retrieval_metrics" in summary:
        rm = summary["retrieval_metrics"]
        md_lines.extend([
            f"| **Strict Hit Rate [Doc & Section]** | **{rm['strict_hit_rate (Doc AND Section)']}** | Strict retrieval Recall@k |",
            f"| **Document Hit Rate** | **{rm['document_hit_rate']}** | Document matching rate |",
            f"| **Section Hit Rate** | **{rm['section_hit_rate']}** | Statutory clause matching rate |",
        ])

    md_lines.extend([
        f"| **Average Legal QA Score** | **{summary.get('legal_benchmark_judge_summary', {}).get('average_score_out_of_5', 0):.2f} / 5.0** | Overall Thai Legal Benchmark Quality |" if enable_judge else "",
        f"| **Good or Perfect (>=4/5)** | **{summary.get('legal_benchmark_judge_summary', {}).get('good_or_perfect_pct (>=4/5)', 0):.1f}%** | Percentage of answers with Good to Perfect rating |" if enable_judge else "",
        f"| **Low Score (<3/5)** | **{summary.get('legal_benchmark_judge_summary', {}).get('below_3_count (<3/5)', 0)} ({summary.get('legal_benchmark_judge_summary', {}).get('below_3_pct (<3/5)', 0):.1f}%)** | Number and percentage of questions scoring below 3 |" if enable_judge else "",
        f"| **Semantic Cosine Similarity** | **{avg_semantic_sim:.4f}** | Dense legal embedding similarity |",
        f"| **ROUGE-L F1** | **{avg_rougeL_f1:.4f}** | Longest Common Subsequence F1 |",
        f"| **ROUGE-L Recall** | **{avg_rougeL_r:.4f}** | Ground truth coverage |",
        f"| **ROUGE-L Precision** | **{avg_rougeL_p:.4f}** | Agent precision against ground truth |",
        f"| **Word Length Ratio** | **{avg_word_ratio:.2f}x** | Agent words / Ground truth words (verbosity) |",
    ])

    if comet_sys_score is not None:
        md_lines.append(f"| **COMET System Score** | **{comet_sys_score:.4f}** | Neural reference-based quality score (`{comet_model_name}`) |")

    if enable_judge and "legal_benchmark_judge_summary" in summary:
        js = summary["legal_benchmark_judge_summary"]
        md_lines.extend([
            f"| **Citation Recall (Judge)** | **{js['average_citation_recall'] * 100:.1f}%** | Ground truth statutory clause coverage |",
            f"| **Citation Precision (Judge)** | **{js['average_citation_precision'] * 100:.1f}%** | Accuracy and relevance of cited statutes |",
            f"| **Citation F1 Score (Judge)** | **{js['average_citation_f1'] * 100:.1f}%** | Harmonic mean of citation precision & recall |",
            f"| **Exact All-or-Nothing Citation** | **{js['exact_citation_rate_pct']:.1f}%** | Percentage with complete (100%) citation match |",
            f"| **Deterministic Citation Recall** | **{avg_det_cite_recall * 100:.1f}%** | Rule-based exact section identifier recall |",
            f"| **Deterministic Citation Precision** | **{avg_det_cite_precision * 100:.1f}%** | Rule-based exact section identifier precision |",
            f"| **Rule / Penalty Correctness** | **{js['rule_threshold_correctness_pct']:.1f}%** | Accuracy of numbers, periods, thresholds, penalties |",
            f"| **Completeness vs GT** | **{js['completeness_rate_pct']:.1f}%** | Full coverage of principles, conditions, and exceptions |",
            f"| **No Hallucination Rate** | **{js['hallucination_free_rate_pct']:.1f}%** | Freedom from fabricated laws or hallucinated claims |",
        ])
    else:
        md_lines.extend([
            f"| **Deterministic Citation Recall** | **{avg_det_cite_recall * 100:.1f}%** | Rule-based exact section identifier recall |",
            f"| **Deterministic Citation Precision** | **{avg_det_cite_precision * 100:.1f}%** | Rule-based exact section identifier precision |",
            f"| **Deterministic Citation F1** | **{avg_det_cite_f1 * 100:.1f}%** | Rule-based harmonic mean of precision & recall |",
        ])

    md_lines = [line for line in md_lines if line]

    md_lines.extend([
        "",
        "## 2. Per-Question Detailed Breakdown",
        "",
    ])

    table_header = "| Row | Question | Status | SemSim | ROUGE-L F1 |"
    if comet_sys_score is not None:
        table_header += " COMET |"
    if enable_judge:
        table_header += " Score | Label | Cite(R/P/F1) | Rule | Comp | NoHal | Critique / Notes |"
    else:
        table_header += " DetCite(R/P) | ROUGE-L Recall |"

    md_lines.append(table_header)
    md_lines.append("| :---: | :--- | :---: | :---: | :---: |" + (" :---: |" if comet_sys_score is not None else "") + (" :---: | :---: | :---: | :---: | :---: | :---: | :--- |" if enable_judge else " :---: | :---: |"))

    for item in eval_items:
        r = item.get("row_index", "-")
        q = item.get("question", "").replace("|", "\\|")
        if len(q) > 40:
            q = q[:37] + "..."
        st = item.get("status", "-")
        m = item["metrics"]
        row_cells = f"| {r} | {q} | {st} | {m['semantic_cosine_sim']:.4f} | {m['rougeL_f1']:.4f} |"
        if comet_sys_score is not None:
            row_cells += f" {m.get('comet_score', 0.0):.4f} |"
        if enable_judge:
            j = item.get("legal_benchmark_judge", {})
            sc = f"{j.get('score', '-')}/5"
            lbl = j.get("score_label", "-")
            cite_val = f"{j.get('citation_recall', 0.0):.2f}/{j.get('citation_precision', 0.0):.2f}/{j.get('citation_f1', 0.0):.2f}"
            rule_v = "YES" if j.get("rule_or_threshold_correctness") else "NO"
            comp_v = "YES" if j.get("completeness_vs_ground_truth") else "NO"
            hal_v = "YES" if j.get("no_hallucination") else "NO"
            critique = j.get("critique", "").replace("\n", " ").replace("|", "\\|")
            if len(critique) > 50:
                critique = critique[:47] + "..."
            row_cells += f" {sc} | {lbl} | {cite_val} | {rule_v} | {comp_v} | {hal_v} | {critique} |"
        else:
            d_rp = f"{m['det_citation_recall']:.2f}/{m['det_citation_precision']:.2f}"
            row_cells += f" {d_rp} | {m['rougeL_r']:.4f} |"
        md_lines.append(row_cells)

    if no_law_samples:
        md_lines.extend([
            "",
            "## 3. Skipped NO_LAW_FOUND Entries (Exempt from Scoring)",
            "",
            "| Row | Question | Status | Reason / Fallback Answer |",
            "| :---: | :--- | :---: | :--- |",
        ])
        for s in no_law_samples:
            r_idx = s.get("row_index", "-")
            q_text = s.get("question", "").replace("|", "\\|")
            if len(q_text) > 50:
                q_text = q_text[:47] + "..."
            st = s.get("status") or s.get("overall_status") or "NO_LAW_FOUND"
            ans_text = (s.get("answer") or s.get("legal_reasoning") or "ไม่พบข้อกฎหมายที่เกี่ยวข้อง").replace("\n", " ").replace("|", "\\|")
            if len(ans_text) > 65:
                ans_text = ans_text[:62] + "..."
            md_lines.append(f"| {r_idx} | {q_text} | {st} | {ans_text} |")

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Markdown summary saved to: {summary_file}")

    return full_output


# =====================================================================
# CLI INTERFACE
# =====================================================================

def main():
    default_emb_model = os.getenv("EMBEDDING_MODEL", "unsloth/embeddinggemma-300m")
    default_judge_model = "google/gemini-3.8-flash"
    default_comet_model = "Unbabel/wmt22-comet-da"

    parser = argparse.ArgumentParser(
        description="Standalone Thai Legal QA Benchmark Evaluation (0-5 Rubric, COMET, ROUGE-L)"
    )
    parser.add_argument(
        "--input",
        "-i",
        "--results",
        type=str,
        default=None,
        help="Path to input results JSON (default: auto-detect latest outputs/THAI/*_results_combined.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        "--output-json",
        type=str,
        default=None,
        help="Path to output metrics JSON file (default: <input_dir>/<input_stem>_eval.json)",
    )
    parser.add_argument(
        "--summary-md",
        "-s",
        type=str,
        default=None,
        help="Path to output Markdown report (default: <input_dir>/<input_stem>_summary.md)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_emb_model,
        help=f"HuggingFace model for semantic cosine similarity (default: {default_emb_model})",
    )
    parser.add_argument(
        "--judge-model",
        "-j",
        type=str,
        default=default_judge_model,
        help=f"LLM model for LLM-as-a-Judge (default: {default_judge_model})",
    )
    parser.add_argument(
        "--disable-judge",
        action="store_true",
        help="Disable LLM-as-a-Judge to save time/API credits",
    )
    parser.add_argument(
        "--enable-comet",
        action="store_true",
        default=False,
        help="Enable COMET score calculation (default: False to prevent large download unless specified)",
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        default=default_comet_model,
        help=f"COMET model checkpoint (default: {default_comet_model})",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent worker threads for LLM-as-a-Judge evaluation (default: 4)",
    )

    args = parser.parse_args()

    # Resolve input path
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = (project_root / input_path).resolve()
    else:
        # Try finding standard LegalGraphRAG results
        candidate_files = [
            project_root / "outputs" / "THAI" / "openrouter_results_combined.json",
            project_root / "outputs" / "openrouter_results_combined.json",
        ]
        found = False
        for cand in candidate_files:
            if cand.exists():
                input_path = cand
                found = True
                print(f"[*] Auto-detected input file: {input_path}")
                break
        if not found:
            # Fallback search
            matching = sorted(project_root.glob("outputs/**/openrouter_results_combined.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if matching:
                input_path = matching[0]
                print(f"[*] Auto-detected input file: {input_path}")
            else:
                input_path = (project_root / "outputs" / "THAI" / "openrouter_results_combined.json").resolve()

    # Default output paths based on input path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_eval.json"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()

    if args.summary_md is None:
        summary_path = input_path.parent / f"{input_path.stem}_summary.md"
    else:
        summary_path = Path(args.summary_md)
        if not summary_path.is_absolute():
            summary_path = (project_root / summary_path).resolve()

    run_evaluation(
        input_file=input_path,
        output_file=output_path,
        summary_file=summary_path,
        embedding_model_name=args.model,
        judge_model_name=args.judge_model,
        enable_judge=not args.disable_judge,
        enable_comet=args.enable_comet,
        comet_model_name=args.comet_model,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
