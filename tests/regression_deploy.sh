#!/usr/bin/env bash
# Deployment regression for ProcurementOCR + ProcurementQA_Agent (multi-tenant).
# Usage: bash tests/regression_deploy.sh   (run on the server; creates and removes its own test data)
QA=${QA:-http://localhost:8000}
OCR=${OCR:-http://localhost:8001}
A=REGTEST_A; B=REGTEST_B
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "PASS  $1"; else fail=$((fail+1)); echo "FAIL  $1 (got '$2', want '$3')"; fi; }
code() { curl -s -o /dev/null -w "%{http_code}" "$@"; }
jget() { python3 -c "import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
neo() { docker exec procurement-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain "$1" | tail -1 | tr -d '"'; }
NEO4J_PASSWORD=$(grep ^NEO4J_PASSWORD "$(dirname "$0")/../.env" | cut -d= -f2- | tr -d '"')

DOC='{"title":"REG TOR","source_id":"reg-tor","markdown":"<!-- Page 1 of 2 -->\n\nใช้วิธีเฉพาะเจาะจงตามมาตรา 56 (2) (ข) แห่งพระราชบัญญัติการจัดซื้อจัดจ้างฯ ค่าปรับร้อยละ 0.2 ต่อวัน\n\n<!-- Page 2 of 2 -->\n\nตรวจรับตามระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างฯ ข้อ 175 และมาตรา 3 แห่งประมวลรัษฎากร"}'

echo "--- QA: tenant documents"
R=$(curl -s -X POST $QA/api/v1/documents -H "Content-Type: application/json" -H "X-Organization-Id: $A" -d "$DOC")
DOC_ID=$(echo "$R" | jget 'd["doc_id"]')
check "ingest returns graph synced" "$(echo "$R" | jget 'd["graph_status"]')" "synced"
check "ingest links 2 statute citations (มาตรา 56, ข้อ 175; not ประมวลรัษฎากร)" "$(echo "$R" | jget 'd["graph_citations"]')" "2"
check "documents POST without tenant -> 400/422" "$(code -X POST $QA/api/v1/documents -H 'Content-Type: application/json' -d "$DOC" | sed 's/422/400/')" "400"
check "owner lists 1 doc" "$(curl -s $QA/api/v1/documents -H "X-Organization-Id: $A" | jget 'd["count"]')" "1"
check "other tenant lists 0 docs" "$(curl -s $QA/api/v1/documents -H "X-Organization-Id: $B" | jget 'd["count"]')" "0"
check "invalid org id -> 400" "$(code $QA/api/v1/documents -H 'X-Organization-Id: bad id!')" "400"
check "re-ingest same source_id replaces (still 1 doc)" \
  "$(curl -s -X POST $QA/api/v1/documents -H "Content-Type: application/json" -H "X-Organization-Id: $A" -d "$DOC" >/dev/null; curl -s $QA/api/v1/documents -H "X-Organization-Id: $A" | jget 'd["count"]')" "1"
check "neo4j: 1 TenantDocument, 2 chunks for tenant" "$(neo "MATCH (d:TenantDocument {org_id:'$A'})-[:CONTAINS]->(t) RETURN count(DISTINCT d)+'/'+count(t)")" "1/2"
check "cross-tenant delete -> 404" "$(code -X DELETE "$QA/api/v1/documents/$DOC_ID" -H "X-Organization-Id: $B")" "404"
check "tenant with no data: /search -> 200" "$(code -X POST $QA/api/v1/search -H 'Content-Type: application/json' -H "X-Organization-Id: $B" -d '{"query":"ค่าปรับ"}')" "200"

echo "--- QA: search isolation"
S=$(curl -s -X POST $QA/api/v1/search -H 'Content-Type: application/json' -H "X-Organization-Id: $B" -d '{"query":"REG TOR ค่าปรับร้อยละ 0.2 ต่อวัน"}')
check "tenant B never sees tenant A chunks" "$(echo "$S" | grep -c "tdoc:$A")" "0"
check "header beats body org_id (REST)" "$(curl -s -X POST $QA/api/v1/search -H 'Content-Type: application/json' -H "X-Organization-Id: $B" -d "{\"query\":\"REG TOR ค่าปรับ\",\"org_id\":\"$A\"}" | grep -c "tdoc:$A")" "0"
check "DGA statutes still present" "$(neo "MATCH (c:StatuteClause {org_id:'DGA'}) RETURN count(c)")" "2633"

echo "--- QA: graph expansion (item 7)"
G='{"title":"REG GRAPH","source_id":"reg-graph","markdown":"<!-- Page 1 of 1 -->\n\nตรวจรับตามระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างฯ ข้อ 175 รหัสทดสอบ ZXQ-REG"}'
GID=$(curl -s -X POST $QA/api/v1/documents -H "Content-Type: application/json" -H "X-Organization-Id: DGA" -d "$G" | jget 'd["doc_id"]')
check "DGA tenant chunk retrieves cited ข้อ 175 via graph" "$(docker exec procurement-mcp python -c "
from core.graph_construct.feature_graph import search_similar_nodes_direct as f
_, laws = f(None, None, 'ระเบียบกระทรวงการคลัง ข้อ 175 รหัสทดสอบ ZXQ-REG', top_k=5, org_id='DGA')
ids = [l.get('id') or '' for l in laws]
print(bool('$GID') and any(i.startswith('$GID') for i in ids) and any('ข้อ ๑๗๕' in (l.get('entry') or '') for l in laws))" 2>/dev/null | tail -1)" "True"
curl -s -o /dev/null -X DELETE "$QA/api/v1/documents/$GID" -H "X-Organization-Id: DGA"

echo "--- QA: ceiling consistency (item 4)"
check "/verify uses 500,000 ceiling" "$(curl -s -X POST $QA/api/v1/verify -H 'Content-Type: application/json' -d '{"procurement_item":"คอมพิวเตอร์","proposed_method":"วิธีเฉพาะเจาะจง","estimated_budget":800000}' | grep -c '500,000')" "1"
check "guardrail flags 1M general ceiling" "$(docker exec procurement-mcp python -c "from core.agent.guardrail import GroundingGuardrail as G; print(G.audit('วิธีเฉพาะเจาะจงใช้ได้ไม่เกินหนึ่งล้านบาท',[])['passed'])")" "False"
check "guardrail accepts MoE special case" "$(docker exec procurement-mcp python -c "from core.agent.guardrail import GroundingGuardrail as G; print(G.audit('สถานศึกษาสังกัดกระทรวงศึกษาธิการใช้วิธีเฉพาะเจาะจงได้ไม่เกินหนึ่งล้านบาท',[])['passed'])")" "True"

echo "--- OCR"
docker exec procurement-ocr-service bash -c 'cd /tmp && printf "ทดสอบ regression ภาษาไทย\n" > reg.txt && soffice --headless --convert-to docx reg.txt >/dev/null 2>&1'
docker cp procurement-ocr-service:/tmp/reg.docx /tmp/reg.docx
JID=$(curl -s -X POST $OCR/api/v1/ocr/jobs -H "X-Organization-Id: $A" -F file=@/tmp/reg.docx | jget 'd["job_id"]')
# the QA push runs right after the job is marked completed, so wait for its status too
for i in $(seq 90); do J=$(curl -s $OCR/api/v1/ocr/jobs/$JID -H "X-Organization-Id: $A"); ST=$(echo "$J" | jget 'd["status"]'); QS=$(echo "$J" | jget 'd["qa_ingest_status"]'); [ "$ST" = failed ] && break; [ "$ST" = completed -a "$QS" != None ] && break; sleep 2; done
check "OCR job completes" "$ST" "completed"
check "OCR result keeps Thai text" "$(curl -s $OCR/api/v1/ocr/jobs/$JID/result -H "X-Organization-Id: $A" | grep -c 'ภาษาไทย')" "1"
check "OCR job hidden from other tenant" "$(code $OCR/api/v1/ocr/jobs/$JID -H "X-Organization-Id: $B")" "404"
check "OCR /jobs/path outside INPUT_DIR rejected" "$(code -X POST $OCR/api/v1/ocr/jobs/path -H 'Content-Type: application/json' -H "X-Organization-Id: $A" -d '{"file_path":"/etc/passwd"}' | grep -c '^4')" "1"
check "OCR invalid org -> 400" "$(code $OCR/api/v1/ocr/jobs/$JID -H 'X-Organization-Id: bad id!')" "400"
check "OCR -> QA push ingested" "$QS" "ingested"
check "OCR qa-ingest retry" "$(curl -s -X POST $OCR/api/v1/ocr/jobs/$JID/qa-ingest -H "X-Organization-Id: $A" | jget 'd["qa_ingest_status"]')" "ingested"
docker restart procurement-ocr-service >/dev/null; for i in $(seq 30); do curl -sf $OCR/health >/dev/null && break; sleep 2; done
check "OCR job survives restart" "$(curl -s $OCR/api/v1/ocr/jobs/$JID -H "X-Organization-Id: $A" | jget 'd["status"]')" "completed"

echo "--- cleanup"
for id in $(curl -s $QA/api/v1/documents -H "X-Organization-Id: $A" | jget '" ".join(x["doc_id"] for x in d["documents"])'); do
  curl -s -o /dev/null -X DELETE "$QA/api/v1/documents/$id" -H "X-Organization-Id: $A"; done
check "delete removes graph nodes" "$(neo "MATCH (n) WHERE n.org_id='$A' AND (n:TenantDocument OR n:TenantChunk) RETURN count(n)")" "0"
docker exec procurement-ocr-service rm -rf /app/data_ocr/$A /tmp/reg.txt /tmp/reg.docx
rm -f /tmp/reg.docx

echo "=== $pass passed, $fail failed"
[ $fail -eq 0 ]
