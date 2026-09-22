# Thai Legal QA Benchmark Evaluation Report
- **Total Input Entries:** 40
- **Evaluated Samples:** 36 (Active QA pairs)
- **Skipped NO_LAW_FOUND:** 4 (Exempt from benchmark evaluation as requested)
- **Skipped Errors:** 0
- **Embedding Model:** `unsloth/embeddinggemma-300m`
- **Benchmark Judge Model:** `google/gemini-3.8-flash` (Thai Legal QA Benchmark Rubric 0-5)
## 1. Overall Metric Averages
| Metric | Score | Description |
| :--- | :---: | :--- |
| **Strict Hit Rate [Doc & Section]** | **34/36 (94.4%)** | Strict retrieval Recall@k |
| **Document Hit Rate** | **36/36 (100.0%)** | Document matching rate |
| **Section Hit Rate** | **34/36 (94.4%)** | Statutory clause matching rate |
| **Average Legal QA Score** | **3.83 / 5.0** | Overall Thai Legal Benchmark Quality |
| **Good or Perfect (>=4/5)** | **75.0%** | Percentage of answers with Good to Perfect rating |
| **Low Score (<3/5)** | **2 (5.6%)** | Number and percentage of questions scoring below 3 |
| **Semantic Cosine Similarity** | **0.7742** | Dense legal embedding similarity |
| **ROUGE-L F1** | **0.3886** | Longest Common Subsequence F1 |
| **ROUGE-L Recall** | **0.5843** | Ground truth coverage |
| **ROUGE-L Precision** | **0.3037** | Agent precision against ground truth |
| **Word Length Ratio** | **2.17x** | Agent words / Ground truth words (verbosity) |
| **Citation Recall (Judge)** | **68.6%** | Ground truth statutory clause coverage |
| **Exact All-or-Nothing Citation** | **47.2%** | Percentage with complete (100%) citation match |
| **Deterministic Citation Recall** | **68.6%** | Rule-based exact section identifier recall |
| **Deterministic Citation Precision** | **59.0%** | Rule-based exact section identifier precision |
| **Rule / Penalty Correctness** | **86.1%** | Accuracy of numbers, periods, thresholds, penalties |
| **Completeness vs GT** | **66.7%** | Full coverage of principles, conditions, and exceptions |
| **No Hallucination Rate** | **94.4%** | Freedom from fabricated laws or hallucinated claims |

## 2. Per-Question Detailed Breakdown

| Row | Question | Status | SemSim | ROUGE-L F1 | Score | Label | Cite(Recall) | Rule | Comp | NoHal | Critique / Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | หากเกิดภัยพิบัติหรือเหตุฉุกเฉินที่ต้อ... | COMPLIANT | 0.8220 | 0.3091 | 5/5 | Perfect | 1.00 | YES | YES | YES | คำตอบถูกต้องสมบูรณ์แบบ ครอบคลุมทั้งแนวทางการจัด... |
| 2 | การจัดซื้อจัดจ้างพัสดุจากรัฐวิสาหกิจห... | COMPLIANT | 0.7242 | 0.4370 | 4/5 | Good | 0.67 | YES | YES | YES | คำตอบมีความถูกต้อง แม่นยำ และครบถ้วนในสาระสำคัญ... |
| 3 | การจัดซื้อจัดจ้างกรณีมีผู้ประกอบการที... | COMPLIANT | 0.7793 | 0.2574 | 3/5 | Partial | 0.67 | YES | NO | YES | ตอบวิธีเฉพาะเจาะจงและการเชิญมาเจรจาตกลงราคาได้ถ... |
| 4 | คณะกรรมการตรวจรับพัสดุในงานซื้อหรือจ้... | COMPLIANT | 0.8951 | 0.4348 | 2/5 | Poor | 0.33 | NO | NO | NO | ตอบองค์ประกอบคณะกรรมการได้ถูกต้อง (ประธาน 1 คน ... |
| 5 | ผู้ที่ได้รับการแต่งตั้งเป็นกรรมการพิจ... | VIOLATION | 0.9123 | 0.5797 | 4/5 | Good | 0.50 | YES | YES | YES | คำตอบถูกต้องครบถ้วนในสาระสำคัญ โดยตอบว่าไม่ได้แ... |
| 6 | การประชุมและลงมติของคณะกรรมการตรวจรับ... | COMPLIANT | 0.7218 | 0.2034 | 2/5 | Poor | 0.33 | NO | NO | YES | คำตอบขาดสาระสำคัญหลักของการประชุมและการลงมติไปเ... |
| 8 | สินค้าหรือบริการที่ได้รับการขึ้นบัญชี... | COMPLIANT | 0.7903 | 0.2342 | 3/5 | Partial | 0.20 | YES | NO | YES | Candidate ตอบสาระสำคัญเรื่องวิธีการจัดซื้อจัดจ้... |
| 9 | การจัดซื้อยาในบัญชียาหลักแห่งชาติ หรื... | COMPLIANT | 0.7525 | 0.2353 | 4/5 | Good | 0.25 | YES | NO | YES | Candidate ตอบวิธีจัดซื้อได้ถูกต้อง (วิธีเฉพาะเจ... |
| 10 | กรณีใดบ้างที่การจัดซื้อจัดจ้างเพื่อกา... | COMPLIANT | 0.6559 | 0.1912 | 4/5 | Good | 0.67 | YES | YES | YES | Candidate ตอบได้ถูกต้องและลงลึกในรายละเอียดของป... |
| 11 | พัสดุส่งเสริมและพัฒนาด้านการเกษตร เช่... | COMPLIANT | 0.8207 | 0.2769 | 4/5 | Good | 0.50 | YES | YES | YES | คำตอบตอบได้ถูกต้องในสาระสำคัญทั้งหมด โดยระบุการ... |
| 13 | ในกรณีที่คู่สัญญาไม่สามารถส่งมอบงานได... | COMPLIANT | 0.9456 | 0.6122 | 4/5 | Good | 0.50 | YES | YES | YES | คำตอบระบุอัตราค่าปรับและเงื่อนไขได้อย่างถูกต้อง... |
| 14 | การวางหลักประกันการปฏิบัติตามสัญญา (P... | COMPLIANT | 0.8921 | 0.5676 | 3/5 | Partial | 0.33 | YES | NO | YES | ตอบตัวเลขหลักถูกต้อง คือ อัตราร้อยละ 5 และกำหนด... |
| 15 | การบอกเลิกสัญญาตาม พ.ร.บ. มีหลักเกณฑ์... | COMPLIANT | 0.7961 | 0.3194 | 4/5 | Good | 0.50 | YES | YES | YES | คำตอบมีความถูกต้องครบถ้วนในสาระสำคัญ ตอบได้ครอบ... |
| 16 | ราคากลางงานก่อสร้างคำนวณตามหลักเกณฑ์ใ... | COMPLIANT | 0.8083 | 0.4962 | 4/5 | Good | 0.67 | YES | YES | NO | คำตอบอธิบายสาระสำคัญได้ครบถ้วนและถูกต้องตามเกณฑ... |
| 17 | โครงการจัดซื้อจัดจ้างที่ต้องจัดทำข้อต... | COMPLIANT | 0.8383 | 0.4706 | 4/5 | Good | 0.33 | YES | YES | YES | คำตอบระบุวัตถุประสงค์และบุคคลผู้ร่วมลงนามทั้ง 3... |
| 18 | ผู้มีสิทธิอุทธรณ์ผลการจัดซื้อจัดจ้างต... | COMPLIANT | 0.7791 | 0.6098 | 3/5 | Partial | 0.25 | NO | NO | YES | ตอบประเด็นผู้รับอุทธรณ์ (หน่วยงานของรัฐ) และระย... |
| 19 | ผู้ยื่นข้อเสนอไม่มีสิทธิอุทธรณ์ในกรณี... | COMPLIANT | 0.7932 | 0.2706 | 4/5 | Good | 1.00 | YES | NO | YES | Candidate ตอบได้ถูกต้อง แม่นยำ และลงรายละเอียดข... |
| 20 | หน่วยงานของรัฐสามารถระบุยี่ห้อของพัสด... | COMPLIANT | 0.6869 | 0.4308 | 5/5 | Perfect | 1.00 | YES | YES | YES | คำตอบถูกต้องสมบูรณ์แบบ สื่อสารหลักการและข้อยกเว... |
| 21 | การจัดซื้อจัดจ้างพัสดุโดยวิธีคัดเลือก... | COMPLIANT | 0.7382 | 0.7170 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบในส่วนเนื้อหาถูกต้องสมบูรณ์แบบครบถ้วนทั้งเ... |
| 22 | คณะกรรมการซื้อหรือจ้างในแต่ละคณะตามระ... | COMPLIANT | 0.8448 | 0.5833 | 5/5 | Perfect | 1.00 | YES | YES | YES | คำตอบถูกต้องสมบูรณ์แบบทั้งในแง่องค์ประกอบคณะกรร... |
| 23 | ในการจัดซื้อจัดจ้างครั้งเดียวกัน กรรม... | VIOLATION | 0.7753 | 0.5556 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบตอบได้ถูกต้องสมบูรณ์ตามหลักเกณฑ์ในสาระสำคั... |
| 24 | การลงมติของคณะกรรมการตรวจรับพัสดุในกา... | COMPLIANT | 0.8401 | 0.3793 | 5/5 | Perfect | 1.00 | YES | YES | YES | คำตอบถูกต้องสมบูรณ์ ชี้แจงหลักเกณฑ์การลงมติของค... |
| 25 | หน่วยงานของรัฐจะต้องกำหนดหลักประกันกา... | COMPLIANT | 0.7688 | 0.3265 | 5/5 | Perfect | 1.00 | YES | YES | YES | คำตอบถูกต้องสมบูรณ์แบบ ระบุเงื่อนไขวงเงินเกินกว... |
| 26 | หากผู้ยื่นข้อเสนอหรือคู่สัญญาเป็นหน่ว... | COMPLIANT | 0.6389 | 0.3200 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบตอบได้ถูกต้องครบถ้วนตามสาระสำคัญและเฉลยมาต... |
| 27 | หน่วยงานของรัฐจะต้องคืนหลักประกันการเ... | COMPLIANT | 0.8374 | 0.5208 | 4/5 | Good | 0.00 | YES | YES | YES | คำตอบในส่วนของเนื้อหาสาระสำคัญ ระยะเวลา 15 วัน ... |
| 28 | การจ่ายเงินค่าพัสดุล่วงหน้า สามารถจ่า... | COMPLIANT | 0.7333 | 0.2095 | 3/5 | Partial | 0.50 | YES | NO | YES | Candidate ตอบรายละเอียดของอัตราร้อยละตามระเบียบ... |
| 29 | อัตราค่าปรับกรณีคู่สัญญาไม่ปฏิบัติตาม... | COMPLIANT | 0.7278 | 0.3443 | 3/5 | Partial | 0.00 | YES | YES | YES | Candidate ตอบสาระสำคัญและอัตราค่าปรับของงานก่อส... |
| 30 | หากคู่สัญญาส่งมอบงานล่าช้าจนมีค่าปรับ... | COMPLIANT | 0.7690 | 0.2692 | 4/5 | Good | 1.00 | YES | NO | YES | คำตอบระบุตัวเลขเกณฑ์ค่าปรับเกินกว่าร้อยละ 10 แล... |
| 31 | คู่สัญญาของรัฐสามารถนำงานตามสัญญาไปจ้... | COMPLIANT | 0.7465 | 0.3789 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบถูกต้องในสาระสำคัญครบถ้วน ทั้งหลักการห้ามจ... |
| 32 | ผู้มีสิทธิอุทธรณ์ผลการจัดซื้อจัดจ้างต... | COMPLIANT | 0.6950 | 0.3030 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบระบุระยะเวลา 7 วันทำการ และจุดเริ่มต้นนับร... |
| 33 | หน่วยงานของรัฐจะสามารถลงนามในสัญญาจัด... | COMPLIANT | 0.7932 | 0.4819 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบถูกต้องครบถ้วนในสาระสำคัญและเงื่อนไขตามมาต... |
| 34 | หน่วยงานของรัฐได้รับการยกเว้นไม่ต้องจ... | COMPLIANT | 0.7174 | 0.3306 | 4/5 | Good | 1.00 | NO | YES | YES | คำตอบระบุข้อยกเว้นตามมาตรา 11 แห่ง พ.ร.บ.การจัด... |
| 35 | ในกรณีใดที่หน่วยงานของรัฐสามารถแต่งตั... | COMPLIANT | 0.7818 | 0.3051 | 3/5 | Partial | 0.50 | NO | NO | YES | ผู้ตอบระบุหลักการตาม พ.ร.บ. มาตรา 100 วรรคสาม ถ... |
| 36 | การซื้อหรือจ้างด้วยวิธีประกวดราคาอิเล... | COMPLIANT | 0.7062 | 0.4286 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบระบุวงเงินเกิน 500,000 บาทขึ้นไปและลักษณะพ... |
| 37 | วิธีตลาดอิเล็กทรอนิกส์ (Electronic Ma... | COMPLIANT | 0.5647 | 0.2047 | 4/5 | Good | 1.00 | YES | YES | YES | คำตอบครอบคลุมสาระสำคัญครบถ้วนตาม Ground Truth ท... |
| 38 | การจัดซื้อจัดจ้างด้วยวิธี e-bidding ก... | COMPLIANT | 0.7806 | 0.3944 | 4/5 | Good | 1.00 | YES | NO | YES | คำตอบระบุเกณฑ์วงเงินงบประมาณและเงื่อนไขได้อย่าง... |

## 3. Skipped NO_LAW_FOUND Entries (Exempt from Scoring)

| Row | Question | Status | Reason / Fallback Answer |
| :---: | :--- | :---: | :--- |
| - | หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะ... | NO_LAW_FOUND | ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหาร... |
| - | คณะกรรมการกำหนดราคากลางงานก่อสร้างมีองค์ประกอบอ... | NO_LAW_FOUND | ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหาร... |
| - | กรณีใดที่หน่วยงานของรัฐสามารถจัดทำข้อตกลงเป็นหน... | NO_LAW_FOUND | ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหาร... |
| - | พัสดุที่ได้รับการรับรองว่าเป็นพัสดุที่ผลิตภายใน... | NO_LAW_FOUND | ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหาร... |