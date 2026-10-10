# Retrieval Experiments — สถานะล่าสุด

บันทึกการทดลอง retrieval หลัง refactor (branch `refactor/phase1-naming`, commit ล่าสุด `1a526b4`)
ใช้ต่องานบนเครื่องอื่นได้ทันที — ทุกตัวเลขวัดที่ **@15**

## 1. ตั้งค่าเครื่องใหม่

```bash
git clone <repo> && git checkout refactor/phase1-naming
pip install -r requirements.txt
cp env.example .env          # ใส่ TOKENMIND / LLM API keys
# คัดลอกโฟลเดอร์ data_ocr/ มาวางที่ root (ไม่อยู่ใน git — อาจเป็นข้อมูล confidential)
```

- corpus (`outputs/corpus/chunks.jsonl`) สร้างอัตโนมัติจาก `data_ocr/` เมื่อไม่มีหรือเก่า
  หรือสั่งเองด้วย `python scripts/build_corpus.py`
- dense cache (`outputs/corpus/dense_<hash>.npy`) จะ embed ใหม่ครั้งแรก (ใช้ TokenMind API)
- reranker `bge-reranker-v2-m3` จะดาวน์โหลดครั้งแรก; มี GPU จะเร็วกว่ามาก
- Local mode (ค่าเริ่มต้น, ไม่ต้องใช้ Docker): `USE_TRI_STORE` ไม่ตั้ง → `InMemoryStore`

## 2. Pipeline ปัจจุบัน (สรุป)

**Chunking** (`core/chunking/`) — `classify_document` แยก 3 แบบ
- `statute` (มี มาตรา/ข้อ ≥ 2 หน่วย): 1 chunk ต่อมาตรา/ข้อ, ยาว > 2000 ตัวอักษรแตกเป็น "(ตอนที่ n)";
  ข้อความหลัง "ประกาศ/ให้ไว้ ณ วันที่" + ผู้ลงนาม = เอกสารแนบท้าย (chunk ตาม "ข้อ N"/"N)")
- `general`: ตามหัวข้อ ≤ 1500 ตัวอักษร, รวม section < 300, ตาราง 1 chunk (แตกตาม header ถ้ายาว)
- `faq`: 1 คู่ถาม-ตอบต่อ chunk (314 คู่)
- เอกสารที่ได้ chunk เดียว (ไม่ใช่ตาราง) → label `ทั้งฉบับ`
- corpus ปัจจุบัน 3,923 chunks: statute_unit 1339, section 1302, table 902, faq 314, preamble 66

**Retrieval** (`core/retrieval/retriever.py`) ต่อ 1 คำถาม
1. query variants = คำถามดิบ + sub-queries จาก decomposer + refined query
2. แต่ละ variant: dense (BGE-M3) + BM25 (newmm) → รวมทั้งหมดด้วย RRF (k=60)
3. top 40 seeds → graph expansion (cites / cited_by / adjacent, 2 ต่อ seed)
4. rerank ครั้งเดียว (cross-encoder, 1200 ตัวอักษร)
5. แบ่ง slot ต่อ variant (4) แล้วเรียง; neighbour ของ graph วางต่อจาก seed ของมัน → top 15

ค่าตั้งผ่าน env:

| env | default | ความหมาย |
|---|---|---|
| `RETRIEVE_RANK_BY` | `primary` | เรียงด้วยคะแนนเทียบคำถามผู้ใช้ (`max` = สูงสุดทุก variant) |
| `RETRIEVE_RERANK_THRESHOLD` | `-inf` | ไม่ตัดทิ้งตามคะแนน |
| `RETRIEVE_GRAPH_POSITION` | `after_seed` | หรือ `end` |
| `RETRIEVE_FAQ_MODE` | `mixed` | `separate` = FAQ แยก ไม่แข่งอันดับกับตัวบท (ต่อท้าย 2 ตัว) |

## 3. ผลการทดลอง

### Full pipeline (LLM จริง) — ก่อน fix `1a526b4`

| ชุด | recall@15 | MRR@15 | ctx | faithfulness | completeness | relevancy |
|---|---|---|---|---|---|---|
| Statute baseline (logs/, .pkl เดิม) | 0.815 | 0.884 | 25.7 | 0.868 | 0.853 | 0.921 |
| Statute (ใหม่) | 0.674 | 0.557 | 17.5 | 0.863 | 0.845 | 0.950 |
| General docs (ใหม่) | evidence 0.882 | 0.669 | — | — | — | — |

### Replay retrieval แบบ offline (ไม่ใช้ LLM, ใช้ query variants จากรันจริง)

replay ให้ผลตรงกับรันจริง (0.685/0.575 vs 0.674/0.557) จึงใช้ทดสอบสมมติฐานได้

| ชุด / config | recall@15 | MRR@15 | ctx |
|---|---|---|---|
| Statute ก่อน fix | 0.685 | 0.575 | — |
| Statute rank=primary, no threshold (corpus เดิม) | 0.761 | 0.611 | — |
| **Statute default ใหม่ + chunker fix** | **0.783** | **0.618** | 21.6 |
| Statute + FAQ separate | 0.772 | 0.707 | 24.3 |
| **General default ใหม่** | **0.875** | **0.655** | 17.4 |
| General + FAQ separate | 0.736 | 0.530 | 19.6 |

### สาเหตุที่ statute ถดถอย (แก้แล้วใน `1a526b4`)
1. rerank เอาคะแนน max ทุก variant → chunk ที่ตรง sub-query แย่งอันดับมาตราหลัก
2. threshold 0.20 ตัดมาตราที่เกี่ยวข้องทิ้ง, ctx ไม่ถึง 15
3. chunker: เอกสารแนบท้ายทั้งก้อนรวมอยู่ใน `ข้อ 5` (32 ตอน), กฎกระทรวงสั้นไม่มี label `ข้อ N`
   → คู่คำตอบที่หาไม่เจอแน่นอนลดจาก 6/91 เหลือ 1 (`ข้อ ๔` ที่ OCR หายตั้งแต่ต้นฉบับ)
4. FAQ ขึ้นอันดับ 1 ใน 7/40 ข้อ (ยังไม่แก้ — ดูข้อ 5)

### ข้อควรระวังเวลาเทียบกับ baseline
- baseline ใช้ corpus ตัวบทล้วน 2,633 chunks; ตอนนี้ 3,923 (มีเอกสารทั่วไป/ตาราง/FAQ แข่งอันดับ)
- baseline ข้ามข้อที่ตอบ NO_LAW_FOUND 2 ข้อ (ประเมิน 38/40), ctx เฉลี่ย 25.7

## 4. คำสั่ง

```bash
# Full pipeline (local)
python run.py --datasets statute
python evaluation/evaluate_rag_triad.py outputs/statute_qa/agentic_results.json \
  --datasets datasets/statute_qa.json --k 15 --corpus-chunks outputs/corpus/chunks.jsonl

python run.py --datasets general
python evaluation/evaluate_rag_triad.py outputs/general_docs_qa/agentic_results.json \
  --datasets datasets/general_docs_qa.json --k 15 --corpus-chunks outputs/corpus/chunks.jsonl

# Retrieval อย่างเดียว (systems: new, new_nograph, new_norerank)
python evaluation/retrieval_benchmark.py

# ทดลอง config ผ่าน env เช่น
RETRIEVE_FAQ_MODE=separate python run.py --datasets statute

# Tests
python -m unittest tests.test_retriever tests.test_bm25_sparse

# ต้องเปิด Docker (tri-store) — ยังไม่ได้รัน
python scripts/migrate_to_tri_store.py --reset-corpus   # ingest corpus ใหม่ + BM25 แท้ (IDF)
python scripts/rebuild_tenant_index.py                  # สร้าง tenant collection ใหม่พร้อม IDF
python scripts/rebuild_tenant_index.py --check          # ตรวจว่า IDF เปิดหรือยัง
```

หมายเหตุ tenant rebuild: Postgres ไม่ได้เก็บชื่อหัวข้อ จึง embed เป็น `[ชื่อเอกสาร]\nเนื้อหา`
embedding จะต่างจากตอน ingest เดิมเล็กน้อย

## 5. งานค้าง / ไอเดียถัดไป

1. **รัน full pipeline ใหม่** บนทั้ง 2 ชุดด้วย default ใหม่ เพื่อยืนยันผล replay
2. **จำกัด FAQ ใน top-k** (เช่น ≤ 2) แทนการแยกทั้งหมด — คาดว่าได้ MRR statute ใกล้ 0.70 โดยไม่เสีย recall general
3. **MRR statute ยังห่าง baseline** (0.62 vs 0.88) — ตรวจ per-question ว่าอะไรขึ้นอันดับ 1 แทนมาตราที่ถูก
4. reranker ตัดที่ 512 tokens — ลอง MaxP (rerank ทีละท่อนแล้วเอาคะแนนสูงสุด) สำหรับ chunk ยาว
5. รันคำสั่ง Docker ในข้อ 4 แล้ววัดผลโหมด tri-store เทียบกับ local
6. push branch `refactor/phase1-naming` (ยังไม่ได้ push)
