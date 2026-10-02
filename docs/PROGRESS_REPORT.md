# Procurement QA Agent: Progress Report

**ช่วงเวลา:** 30 ก.ย. – 2 ต.ค. 2569
**ขอบเขต:** ตั้งแต่ branch `migrate/Agentic-RAG` ถึง `fix/data-ingestion` (ปัจจุบัน)

---

## สรุปภาพรวม

1. **Workflow:** เปลี่ยน engine หลักจาก Corrective RAG เป็น **Agentic RAG** (LangGraph)
2. **Data:** แบ่ง corpus ใหม่จาก **page-level** เป็น **section-level** (หนึ่ง chunk = หนึ่งมาตรา/ข้อ) พร้อมเลขหน้าของเอกสารต้นฉบับ
3. **Database:** ย้ายจากไฟล์ `.pkl` ในหน่วยความจำ ไปเป็น **Tri-Store**: PostgreSQL + Qdrant + Neo4j
4. **Graph:** เชื่อมความสัมพันธ์ทางกฎหมาย**ข้ามเอกสาร**ได้ครบตามที่ออกแบบ (ระเบียบ/กฎกระทรวง → พ.ร.บ.)
5. **Multi-tenant:** แยกข้อมูลตามหน่วยงาน ไม่มีข้อมูลรั่ว และค้นหาเสร็จในระดับมิลลิวินาที
6. **Eval:** Hit@20 = 0.949, Faithfulness = 0.833 (RAG Triad, 40 คำถาม)

---

## 1. Updated Workflow: Corrective RAG → Agentic RAG

```
                 ┌──────────── rewrite_query ◄────────────┐
                 ▼              (ค้นไม่เจอ)                │ (ตอบไม่ได้ / NO_LAW_FOUND)
analyze_query → retrieve_and_rerank → generate_answer ────┴──► guardrail → คำตอบ
 (แยกประเด็นย่อย)  (hybrid + graph)     (สรุปพร้อมอ้างมาตรา)     (ตรวจการอ้างอิง)
```

| ขั้นตอน | หน้าที่ |
|---|---|
| **Issue Decomposer** | แตกคำถามซับซ้อนเป็นประเด็นย่อย (Q1, Q2, …) ค้นหาแยกกัน ไม่ให้ประเด็นหลักกลบประเด็นรอง |
| **Hybrid Retrieval + Rerank** | Dense (BGE-M3) + Thai BM25 รวมผลด้วย RRF แล้ว rerank ด้วย cross-encoder `bge-reranker-v2-m3` |
| **Graph Expansion** | ดึงมาตราที่เกี่ยวข้องผ่าน knowledge graph (อ้างถึง / ออกตามความใน / มาตราถัดไป) |
| **Self-Reflection** | ถ้าค้นไม่เจอหรือตอบไม่ได้ จะเขียนคำค้นใหม่แล้วลองอีกครั้ง (สูงสุด 2 รอบ) |
| **Grounding Guardrail** | ตรวจว่าทุกมาตราที่อ้างในคำตอบมาจากหลักฐานที่ค้นได้จริง ไม่ใช่ LLM แต่งขึ้น |

**ผลลัพธ์ที่ได้:**
- **`issues_breakdown`:** ระบุประเด็นที่ตอบไม่ได้ (`NO_LAW_FOUND` / `OUT_OF_LEGAL_SCOPE`) ให้ Super-Orchestrator ส่งต่อไปยัง agent อื่น
- **ช่องทางเรียกใช้:** ทั้ง REST (`POST /api/v1/qa`) และ MCP (`ask_procurement_law`) ด้วย contract เดียวกัน

---

## 2. Data: Page-level → Section-level Chunking

| | Page-level (เดิม) | Section-level (ปัจจุบัน) |
|---|---|---|
| จำนวน chunk | 3,192 | 2,633 |
| หน่วยของ chunk | หน้ากระดาษ (หนึ่ง chunk ครอบหลายมาตรา) | หนึ่งมาตรา / ข้อ |
| ตัวอย่าง label | `มาตรา ๑๕ \| มาตรา ๑๖ \| มาตรา ๑๗ \| มาตรา ๑๘` | `พ.ร.บ.จัดซื้อจัดจ้างฯ 2560 \| มาตรา ๕๖` |
| ขนาด chunk (p90) | 6,383 ตัวอักษร | 3,622 ตัวอักษร |
| การระบุแหล่งที่มา | ระบุมาตราที่แน่นอนไม่ได้ | อ้างได้ตรงมาตรา/ข้อ |

**องค์ประกอบของ corpus ปัจจุบัน:** 92 เอกสาร (พ.ร.บ., ระเบียบ, กฎกระทรวง, ประกาศ, หนังสือเวียน) และ FAQ กรมบัญชีกลาง 29 ข้อ

| ประเภท chunk | จำนวน |
|---|---|
| มาตรา (พ.ร.บ.) | 192 |
| ข้อ (ระเบียบ / กฎกระทรวง / ประกาศ) | 937 |
| ทั่วไป (ตาราง, ภาคผนวก, ราคากลาง) | 1,444 |
| คำนำ / บทนำ | 60 |

**การจัดการมาตราที่ยาว:** มาตรา/ข้อที่ยาวเกินจะแบ่งเป็น "ตอนที่ 1, 2, …" (1,570 chunk) และเชื่อมกันตามลำดับการอ่านใน graph

**เลขหน้าของเอกสารต้นฉบับ:**
- **ที่มา:** กู้คืนจาก marker หน้าในไฟล์ OCR (Typhoon OCR) ได้ **2,510 / 2,633 chunk (95%)**
- **การแสดงผล:** ทุกการอ้างอิงในคำตอบมีไฟล์และเลขหน้า เช่น `พรบ/...พ.ศ. 2560.md` หน้า `19-20/42` frontend แสดงคู่กับเอกสารจริงได้
- **การ deploy:** เลขหน้าฝังไว้ใน corpus แล้ว จึงไม่ต้องมีไฟล์ OCR บน server
- **ข้อจำกัด:** ไฟล์ OCR 7 ฉบับไม่มี marker หน้าตั้งแต่ต้นทาง จึงไม่มีเลขหน้า

---

## 3. Tri-Store Database

| Store | บทบาท | ข้อมูล |
|---|---|---|
| **PostgreSQL** | Single source of truth | ข้อความกฎหมาย, metadata, เลขหน้า, tenant, audit log |
| **Qdrant** | Vector search | BGE-M3 1,024 มิติ + Thai BM25 sparse, รวมผลด้วย RRF ฝั่ง server |
| **Neo4j** | Knowledge graph | เอกสาร, มาตรา/ข้อ, FAQ และความสัมพันธ์ระหว่างกัน |

**จำนวนข้อมูลในแต่ละ store:** 92 เอกสาร, 2,633 มาตรา/ข้อ, 29 FAQ ตรงกันทั้ง 3 store

**การ deploy:**
- **คำสั่งเดียว:** `docker compose up` เริ่ม DB → โหลดข้อมูลเข้า tri-store อัตโนมัติ (~3.5 นาที ครั้งแรก) → เปิด API
- **เปิดครั้งถัดไป:** ข้ามการโหลดเองถ้าข้อมูลครบ, เชื่อม graph ใหม่เองเมื่อกฎการเชื่อมเปลี่ยน (~20 วินาที ไม่ต้อง embed ใหม่)
- **Healthcheck:** ทุก DB มี healthcheck และ `/ready` ตอบ 503 ถ้า store ใดไม่พร้อมหรือไม่มีข้อมูล

---

## 4. Graph Relationships

| ความสัมพันธ์ | ความหมาย | จำนวน |
|---|---|---|
| `CONTAINS` | เอกสาร → มาตรา/ข้อ | 2,633 |
| `CITES_CLAUSE` | มาตรา/ข้อ อ้างถึงอีกมาตรา/ข้อ | 991 (ข้ามเอกสาร 354) |
| `EMPOWERED_BY` | ระเบียบ/กฎกระทรวง ออกตามความในมาตราของ พ.ร.บ. | 183 |
| `ADJACENT_SECTION` | ลำดับการอ่าน (มาตราถัดไป / ตอนถัดไป) | 2,429 คู่ |
| `REFERENCES_DOCUMENT` | FAQ → เอกสารที่เกี่ยวข้อง | 29 |

**ก่อน vs หลังปรับปรุง:**

| | ก่อน | หลัง |
|---|---|---|
| การอ้างข้ามเอกสาร (ระเบียบ → พ.ร.บ.) | 0 | 354 |
| `EMPOWERED_BY` | 0 | 183 |
| FAQ ที่เชื่อมกับกฎหมาย | 0 | 29 |
| เอกสารที่ถูกจัดเป็น "พ.ร.บ." | 8 (ผิด) | 1 (ถูก) |
| ลำดับมาตราที่กระโดดเลข | 11 | 0 |

**ตัวอย่าง:** มาตรา 56 (วิธีเฉพาะเจาะจง) เชื่อมไปถึงกฎหมายลูก **26 รายการ** เช่น กฎกระทรวงกำหนดกรณีการจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจง ฉบับที่ 1–4 ส่วนระเบียบฯ ข้อ 79 เชื่อมกลับไปที่มาตรา 56

**การป้องกันการเชื่อมผิด:**
- **ไม่นับหน่วยวัด:** "ม." ในตารางราคา (เมตร/มิลลิเมตร) ไม่ถูกนับเป็น "มาตรา"
- **ไม่ข้ามกฎหมายอื่น:** "มาตรา N แห่งประมวลรัษฎากร" ไม่ถูกเชื่อมกับ พ.ร.บ.จัดซื้อจัดจ้าง
- **แยกตามปี:** "ระเบียบ...พ.ศ. 2535 ข้อ 143" ไม่ถูกเชื่อมกับข้อ 143 ของระเบียบปี 2560
- **ใช้กฎชุดเดียว:** โหมด `.pkl` (PoC บนเครื่อง) และ tri-store ใช้กฎการเชื่อมชุดเดียวกัน ผลจึงตรงกัน

---

## 5. Multi-tenant Feature

**หลักการแยกข้อมูล:**
- **สิทธิ์การมองเห็น:** แต่ละหน่วยงานเห็นข้อมูลกลาง (`PUBLIC`) และข้อมูลของตัวเองเท่านั้น บังคับในทั้ง 3 store
- **การระบุหน่วยงาน:** `org_id` ใน body → header `X-Organization-Id` → `DEFAULT_ORG_ID`
- **ฝั่ง MCP:** header ชนะค่า `org_id` ที่ LLM กรอกมา ป้องกันการสั่งข้ามหน่วยงานผ่าน prompt
- **Audit log:** ทุกคำถามถูกบันทึกลง `query_audit_logs` (หน่วยงาน, คำถาม, คำตอบ, มาตราที่อ้าง, grounded, latency)

**ผลทดสอบการแยกข้อมูล:** ผ่านทั้ง 7 test (PostgreSQL / Qdrant / Neo4j / การระบุหน่วยงานผ่าน body, header, default)

### Indexing ของแต่ละ Database
ทุก record ในทั้ง 3 store มี field `org_id` (`PUBLIC` = ข้อมูลกลาง) และทุก query กรองด้วยเงื่อนไข `org_id ∈ {PUBLIC, หน่วยงานผู้ถาม}` index ของแต่ละ store ออกแบบให้การกรองนี้ทำ**ระหว่างค้นหา** ไม่ใช่กรองทีหลัง จึงไม่ช้าลงเมื่อจำนวนหน่วยงานเพิ่มขึ้น

| Database | เทคนิค Multi-tenancy | วิธีทำงาน |
|---|---|---|
| PostgreSQL | **Row-Level Security (RLS)** | Database บังคับ policy เอง แม้ query จะลืมใส่เงื่อนไขหน่วยงาน (ทำงานคู่กับ `WHERE org_id` อีกชั้น) |
| Qdrant | **Payload-based multitenancy** | แนบ `org_id` ใน payload ของทุก point แล้วกรองด้วย payload index ระหว่างค้น HNSW |
| Neo4j | **Property-based multitenancy** | แนบ `org_id` เป็น property ของทุก node แล้วกรองระหว่าง traversal |

**PostgreSQL (Single source of truth)**

| Index | คอลัมน์ | ใช้ทำอะไร |
|---|---|---|
| Primary key | `clause_id`, `doc_id`, `case_id` | ดึงข้อความเต็มของผลค้นหาจาก Qdrant ทีละชุด (hydration) |
| `idx_*_org_id` (B-tree) | `org_id` ของเอกสาร, มาตรา, FAQ | กรองข้อมูลตามหน่วยงาน |
| `idx_statute_doc_sec` / `doc_cls` (B-tree) | `(doc_id, section_num)`, `(doc_id, clause_num)` | ค้นมาตรา/ข้อแบบตรงตัว (`get_statute_section`) |
| `idx_statute_content_trgm` (GIN trigram) | `content_thai` | ค้นข้อความภาษาไทยแบบบางส่วน |
| `idx_audit_org_created` (B-tree) | `(org_id, created_at)` | ดู audit log ของแต่ละหน่วยงานเรียงตามเวลา |

**การตั้งค่า RLS:**
- **แยก role:** app ใช้ role `procurement_app` ที่ไม่ใช่ superuser และข้าม RLS ไม่ได้ (superuser และเจ้าของตารางข้าม RLS เสมอ) ส่วนการโหลดข้อมูลและงาน admin ใช้ owner role
- **ส่งหน่วยงานทีละ transaction:** ทุก request ตั้งค่า `app.org_id` แบบ transaction-local ค่าจึงไม่ค้างข้าม connection ใน pool
- **Policy ใน 4 ตาราง:** `legal_documents`, `statute_clauses`, `faq_cases`, `query_audit_logs`

| Policy | สิทธิ์ |
|---|---|
| `tenant_read` | อ่านได้: ข้อมูลกลาง (`PUBLIC`) + ข้อมูลของหน่วยงานตัวเอง |
| `tenant_insert` / `tenant_update` / `tenant_delete` | เขียนได้เฉพาะข้อมูลของหน่วยงานตัวเอง (แก้ข้อมูลกลางไม่ได้) |
| ไม่ระบุหน่วยงาน | เห็นแค่ข้อมูลกลาง (fail-closed) |

```sql
ALTER TABLE statute_clauses ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_read ON statute_clauses FOR SELECT TO procurement_app
  USING (org_id = 'PUBLIC' OR org_id = current_setting('app.org_id', true));

CREATE POLICY tenant_insert ON statute_clauses FOR INSERT TO procurement_app
  WITH CHECK (org_id = current_setting('app.org_id', true));
```

ตัวอย่าง: ดึงข้อความเต็มของผลค้นหาจาก Qdrant (hydration) ภายใต้ RLS

```sql
BEGIN;
SELECT set_config('app.org_id', 'DGA', true);   -- มีผลเฉพาะ transaction นี้

SELECT sc.*, ld.title AS doc_title, ld.source_file, ld.total_pages
FROM statute_clauses sc
JOIN legal_documents ld ON sc.doc_id = ld.doc_id
WHERE sc.clause_id IN (...)
  AND sc.org_id IN ('PUBLIC', 'DGA');            -- ชั้นที่ 2 + ใช้ index idx_statute_org_id
COMMIT;
```

**ผลทดสอบ RLS (6 tests):** query ที่ไม่ใส่ `WHERE org_id` เลยยังเห็นเฉพาะข้อมูลของหน่วยงานตัวเอง, ไม่ระบุหน่วยงานแล้วไม่เห็นข้อมูลของหน่วยงานใด, เขียนข้อมูลแทนหน่วยงานอื่นไม่ได้, แก้ข้อมูลกลางไม่ได้

**Qdrant (Vector search)**

| Index | การตั้งค่า | ใช้ทำอะไร |
|---|---|---|
| Dense vector (HNSW) | BGE-M3 1,024 มิติ, cosine, `m=16`, `ef_construct=128` | ค้นหาตามความหมาย |
| Sparse vector | Thai BM25 (ตัดคำด้วย PyThaiNLP) | ค้นหาตามคำสำคัญ / เลขมาตรา |
| Payload index (keyword) | `org_id`, `doc_id`, `entry`, `topics`, `case_id` | กรองหน่วยงาน / เอกสาร |
| Payload index (integer) | `section_num`, `clause_num`, `chapter_num` | กรองตามเลขมาตรา / ข้อ / หมวด |

- **กรองระหว่างค้น HNSW:** เงื่อนไขหน่วยงานถูกใส่ในขั้น prefetch ของทั้ง dense และ sparse ก่อนรวมผลด้วย RRF Qdrant จึงใช้ payload index กรองระหว่างค้น HNSW ได้ผลครบ top-k ของหน่วยงานนั้น ไม่เกิดกรณีที่ผลลัพธ์ส่วนใหญ่เป็นของหน่วยงานอื่นแล้วถูกตัดทิ้งจนเหลือน้อย

ตัวอย่าง: hybrid search (dense + sparse รวมด้วย RRF) ที่กรองหน่วยงานในทุก prefetch

```python
tenant = models.Filter(should=[                       # org_id = PUBLIC หรือ หน่วยงานผู้ถาม
    models.FieldCondition(key="org_id", match=models.MatchValue(value="PUBLIC")),
    models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
])

client.query_points(
    collection_name="procurement_statutes",
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense_bge_m3", limit=60, filter=tenant),
        models.Prefetch(query=models.SparseVector(indices=idx, values=val),
                        using="sparse_bm25", limit=60, filter=tenant),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),   # Reciprocal Rank Fusion
    limit=20,
)
```

**Neo4j (Knowledge graph)**

| Index | Property | ใช้ทำอะไร |
|---|---|---|
| Unique constraint | `LegalDocument.doc_id`, `StatuteClause.clause_id`, `FAQCase.case_id`, `LegalTopic.name` | จุดเริ่ม traversal จากผลค้นหา และป้องกันข้อมูลซ้ำ |
| Composite index | `StatuteClause(doc_id, section_num)`, `StatuteClause(doc_id, clause_num)` | หามาตรา/ข้อในเอกสารที่ระบุ |
| Property index | `org_id` ของ `LegalDocument`, `StatuteClause`, `FAQCase` | กรองข้อมูลตามหน่วยงาน |

- **traversal เริ่มจาก `clause_id`:** ใช้ unique index เข้าถึงโหนดได้ทันที แล้วกรองเพื่อนบ้านตาม `org_id` เฉพาะในขอบเขตที่เดินถึง ไม่ต้องสแกนทั้ง graph จึงใช้เวลาไม่ถึง 10 ms

ตัวอย่าง: หากฎหมายลูกจากเอกสารอื่นที่ออกตามความใน หรืออ้างถึงมาตรา 56 ของ พ.ร.บ.

```cypher
MATCH (act:StatuteClause {clause_id: $cid})<-[r:EMPOWERED_BY|CITES_CLAUSE]-(reg:StatuteClause)
WHERE reg.doc_id <> act.doc_id
  AND coalesce(reg.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]      // tenant filter
MATCH (reg)<-[:CONTAINS]-(doc:LegalDocument)
WHERE coalesce(doc.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
RETURN reg.entry, doc.title, type(r) AS relation
```

### การวัดความเร็ว Retrieval กรณี multi-tenant
**วิธีวัด:** จำลอง 10 หน่วยงาน (DGA, depa, ETDA, กทม., สธ., กค., จุฬาฯ, กฟผ., ปตท., ทอท.) ใส่ข้อมูลเฉพาะหน่วยงานเพิ่มบนข้อมูลกลาง แล้วค้นหา 20 queries จากมุมของแต่ละหน่วยงาน

| ขั้นตอน | p50 | p95 |
|---|---|---|
| Qdrant hybrid search + PostgreSQL hydration | 9.7 ms | 12.1 ms |
| Neo4j graph traversal | 6.5 ms | 9.4 ms |
| รวมต่อ query (รวม embedding API) | 472 ms | 493 ms |

**ผลลัพธ์:**
- **ไม่มีข้อมูลรั่ว:** **0 / 20 queries** ไม่มีข้อมูลของหน่วยงานอื่นปนมาในผลค้นหา
- **ฐานข้อมูลเร็วมาก:** การค้นในฐานข้อมูลทั้งหมดใช้ **~16 ms** ต่อ query
- **คอขวดคือ embedding:** ~95% ของเวลารวมคือการเรียก embedding API ภายนอก ถ้าจะเร่งความเร็วต้องเริ่มที่ embedding (เช่น cache หรือ host โมเดลเอง)
- **ไม่ทิ้งข้อมูลจำลองไว้:** benchmark ลบข้อมูลจำลองทิ้งเองเมื่อจบ ฐานข้อมูลจริงไม่ปนเปื้อน

**End-to-end QA:**
- **เวลาต่อคำถาม:** 40–135 วินาที
- **ทำงานพร้อมกัน:** 3 คำถามพร้อมกันเสร็จใน 136 วินาที (ขนานกันจริง) server ยังตอบ healthcheck ได้ภายใน 0.03 วินาทีระหว่างนั้น

---

## 6. Workflow Evaluation (RAG Triad)

**Dataset:** 40 คำถามกฎหมายจัดซื้อจัดจ้าง พร้อม ground truth ระดับเอกสาร + มาตรา (ประเมินได้ 39 ข้อ อีก 1 ข้อระบบตอบว่าไม่พบกฎหมาย)
**ที่มา:** `logs/agentic_triad_report.json`

### Retrieval Layer (k = 20)
| Metric | ค่า | ความหมาย |
|---|---|---|
| **Hit@20** | **0.949** | 95% ของคำถาม ดึงมาตราที่ถูกต้องได้อย่างน้อย 1 มาตรา |
| **Recall@20** | **0.809** | ครอบคลุมมาตราที่ถูกต้องทั้งหมด 81% |
| **MRR@20** | **0.743** | มาตราที่ถูกต้องตัวแรก อยู่อันดับ 1–2 โดยเฉลี่ย |
| Adjusted Precision@20 | 0.838 | ความแม่นยำเมื่อเทียบกับจำนวนมาตราที่ต้องหา |

### Generation Layer (LLM-as-a-Judge)
| Metric | ค่า | ความหมาย |
|---|---|---|
| **Faithfulness** | **0.833** | คำตอบอิงตามหลักฐานที่ค้นได้ |
| **Completeness** | **0.851** | ตอบครบทุกประเด็นของคำถาม |
| **Answer Relevancy** | **0.923** | คำตอบตรงกับสิ่งที่ถาม |

### การปรับปรุงเครื่องมือประเมิน (ระหว่างรอบนี้)
- **จับคู่จาก label ของ chunk:** มาตราที่ถูกนับต้องเป็นมาตรานั้นจริง ไม่ใช่แค่มีคำว่า "มาตรา N" ในเนื้อหา (ตัวเก่านับเกินจริงในบางกรณี)
- **รองรับชื่อเอกสารที่ถูกตัดสั้น** และ ground truth ที่แยกอนุมาตรา (เช่น `(๕)` ต่อจากมาตรา 96)
- **รายงานหลายระดับ:** @5 / @10 / @20 และ **context recall** (recall บน context ทั้งหมดที่ generator ได้รับ) ให้คะแนน retrieval สอดคล้องกับสิ่งที่ generator ใช้จริง

> **หมายเหตุ:** รอบประเมินล่าสุดหลัง cleanup ได้คะแนน retrieval ต่ำผิดปกติ ตรวจพบว่า reranker ไม่ได้ทำงานเพราะ dependency หายไปตอน cleanup แก้แล้วพร้อมเพิ่มการตรวจใน `/ready` และ test อยู่ระหว่างรันประเมินใหม่ด้วยเครื่องมือที่ปรับปรุงแล้ว

---

## 7. งานสนับสนุนอื่น ๆ

| เรื่อง | สิ่งที่ทำ |
|---|---|
| API สำหรับ orchestrator | Request `query` + `org_id`; response มีเฉพาะที่จำเป็น (`answer`, `citations` พร้อมไฟล์/หน้า, `unresolved_issues`, `grounded`) ขนาดลดจาก ~8.4 KB เหลือ ~5 KB |
| MCP | แก้ endpoint ที่ใช้งานไม่ได้, ไม่ block server ระหว่างทำงาน, contract เดียวกับ REST |
| Concurrency | จำกัด QA พร้อมกัน (ค่าเริ่มต้น 4) เกินแล้วตอบ 503 + `Retry-After` |
| Reliability | Timeout ต่อ LLM call, จำกัดขนาด cache, rotate log, ไม่เปิดเผยรายละเอียด error |
| Reproducibility | Pin ทุก dependency (`requirements.lock`), ชื่อ config มาตรฐานเดียว |
| Tests | 33 automated tests (รวม RLS 6 tests) + MCP smoke test 17 checks ผ่านทั้งหมด |

---

## 8. ความเสี่ยงและขั้นตอนถัดไป

| เรื่อง | รายละเอียด | แผน |
|---|---|---|
| คำตอบไม่คงที่ | คำถามเดิมบางข้อสลับระหว่าง "ตอบได้" กับ "ไม่พบกฎหมาย" ในแต่ละรอบ | ลด variance ของขั้นแยกประเด็น / retrieval |
| กฎหมายที่ถูกยกเลิก | LLM บางครั้งอ้างระเบียบสำนักนายกฯ พ.ศ. 2535 ที่ถูกยกเลิกแล้ว | รอแนวทางจาก DGA |
| Authentication | API ยังไม่มีการยืนยันตัวตน | เพิ่มก่อนเปิดให้หลายหน่วยงานใช้จริง |
| Infrastructure | DB เปิด port และใช้รหัสผ่าน default | ปิด port, ตั้งรหัสผ่านใหม่, backup ก่อน production |

**ขั้นตอนถัดไป:**
1. รัน RAG Triad ใหม่ด้วยเครื่องมือประเมินที่ปรับปรุงแล้ว และเทียบกับ baseline
2. ทีมทดลองใช้งานบนเซิร์ฟเวอร์ทดสอบ เก็บ feedback
3. แก้ปัญหาคำตอบไม่คงที่
4. เตรียม production: authentication, ความปลอดภัยของ infrastructure
