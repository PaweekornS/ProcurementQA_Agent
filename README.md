# LegalGraphRAG: Multi-Agent Corrective RAG (CRAG) for Thai Government Procurement Law

> **An advanced legal question-answering and statutory retrieval system for Thai Government Procurement laws and regulations.**
> Powered by an active **Multi-Agent Corrective RAG (CRAG)** architecture combining **Multi-Aspect Hybrid Retrieval (Dense Vector + Thai BM25 + GPU Cross-Encoder Reranking)** and **Knowledge Graph Traversal**.

---

## 🚀 **Key Highlights**

- ✅ **Multi-Agent CRAG Architecture**:
  1. **Issue Decomposer & Intent Classifier (Agent 1)**: Decomposes complex procurement inquiries into atomic sub-questions (`sub-issues`), preventing dominant topics from masking secondary issues during search.
  2. **Multi-Aspect Hybrid Retrieval**: Concurrently queries dense vector embeddings, tokenized Thai BM25, GPU Cross-Encoder reranker (`BAAI/bge-reranker-v2-m3`), and graph clusters with reciprocal score fusion.
  3. **Legal Synthesizer & Adjudicator (Agent 2)**: Formulates grounded legal judgments, strictly separating a concise `direct_answer` from detailed `legal_reasoning`.
  4. **Completeness & Grounding Auditor (Agent 3)**: Audits whether all sub-issues are comprehensively addressed and explicitly substantiated by retrieved statutory clauses.
  5. **Targeted Query Refiner (Agent 4) with Active Feedback Loop (`max_retry=1`)**: Automatically constructs focused search queries and traverses 1-hop graph neighbors (`RELATED_TO`, `RELATES_TO_LAW`) for missing statutory aspects, triggering a secondary retrieval pass.
- ✅ **Clean Direct Answer**: `direct_answer` provides an unambiguous, straightforward summary free of statutory section numbers (e.g. no "มาตรา" or "ข้อ" clutter), keeping all legal citations structured in `applicable_laws`.
- ✅ **3-Tier NO_LAW_FOUND Guardrails**: Detects out-of-scope inquiries across three defensive layers: pre-LLM retrieval emptiness, LLM prompt guidelines, and post-processing override.
- ✅ **Comprehensive Benchmark Evaluator**: Evaluates legal answer quality using an LLM Judge (0–5 rubric) measuring:
  - Strict Hit Rate (Doc & Section Recall@k)
  - Citation Precision, Recall, and F1
  - Rule & Penalty Correctness
  - Completeness vs Ground Truth
  - Hallucination-Free Rate
  - Verbosity / Word Length Ratio

---

## 🧩 **Project Structure**

```text
LegalGraphRAG_procurement/
├── core/
│   ├── LegalGraphRAG.py       # Main LegalGraphRAG engine and LegalGraphRAGConfig
│   ├── crag/                  # Multi-Agent Corrective RAG Package
│   │   ├── classifier.py      # Agent 1: Issue Decomposer & Intent Classifier
│   │   ├── synthesizer.py     # Agent 2: Legal Synthesizer & Adjudicator
│   │   ├── auditor.py         # Agent 3: Completeness & Grounding Auditor
│   │   ├── refiner.py         # Agent 4: Query Refiner & Graph Neighbor Search
│   │   └── pipeline.py        # CRAG Orchestrator (Multi-Aspect Search & Retry Loop)
│   ├── graph_construct/       # Knowledge graph construction and retrieval
│   │   ├── feature_graph.py   # Hybrid Search (Dense + BM25 + Cross-Encoder Reranker)
│   │   └── graph_db.py        # NetworkX In-Memory Graph Database
│   ├── judge/                 # Legal judgment & guardrails (judge_crime.py)
│   ├── preprocess/            # Procurement entity & feature extraction
│   ├── prompt/                # Unified prompt registry
│   │   ├── crag/              # Prompts for Decomposer, Auditor, and Refiner
│   │   ├── judge/             # Prompts for Legal Synthesizer
│   │   └── preprocess/        # Prompts for Procurement Features
│   └── utils/                 # Utilities and pipeline helper functions
├── configs/
│   └── thai_procurement.env   # Model, API keys, retrieval top-k, and reranker settings
├── datas/
│   ├── law_to_crime.json      # Knowledge base of Thai procurement statutory clauses (2,486 nodes)
│   └── cases_with_feature.json# Repository of past procurement consultations and advisory cases
├── datasets/
│   └── crime_data_THAI_small.json # Benchmark dataset of 40 Thai procurement inquiry test cases
├── evaluation/
│   └── evaluate_results.py    # Benchmark evaluation script (LLM Judge + statistical metrics)
├── outputs/
│   └── openrouter_graph_db.pkl# Cached in-memory graph and vector index for instant loading
├── run.py                     # Main CLI execution pipeline
└── README.md
```

---

## 🛠️ **Installation & Usage**

### 1️⃣ Dependencies

Install required packages (Python 3.10+):

```bash
pip install -r requirements.txt
```

> **Note for GPU Reranker**:
> Ensure CUDA-compatible `torch` and `sentence-transformers` are installed for accelerating `BAAI/bge-reranker-v2-m3`.

---

### 2️⃣ Environment Configuration (`configs/thai_procurement.env`)

Configure your environment settings in `.env` or `configs/thai_procurement.env`:

```ini
# OpenRouter / OpenAI API Configuration
model_name=openrouter:google/gemini-2.5-flash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Embedding Service (Ollama / Local)
embedding_api_url=http://localhost:11434/api/embed
embedding_model=unsloth/embeddinggemma-300m

# Graph & Retrieval Settings
graph_db_path=./outputs/openrouter_graph_db.pkl
top_retrieve_top_k=3
direct_retrieve_top_k=5
reranker_model=BAAI/bge-reranker-v2-m3
reranker_device=cuda:0
reranker_threshold=-4.0

# CRAG Multi-Agent Loop
crag_enabled=true
crag_max_retry=1
```

---

### 3️⃣ Running LegalGraphRAG (`run.py`)

Execute the pipeline on the evaluation dataset. Use `--no-build-graph` to load the pre-built knowledge graph:

```bash
python run.py --no-build-graph
```

Execution steps performed automatically:

1. Loads the NetworkX In-Memory Knowledge Graph from `outputs/openrouter_graph_db.pkl`.
2. Builds and caches the Thai BM25 index alongside dense embeddings.
3. Dispatches inquiries through the **CRAG Multi-Agent Loop**.
4. Outputs structured legal predictions and `crag_meta` diagnostics to `outputs/THAI/openrouter_results.json`.

---

### 4️⃣ Benchmark Evaluation (`evaluate_results.py`)

Evaluate the generated answers against ground-truth legal principles using the LLM Judge:

```bash
python evaluation/evaluate_results.py
```

Outputs generated:

- `outputs/THAI/eval_results.json`: Detailed per-question metric logs.
- `outputs/THAI/eval_summary.md`: Comprehensive evaluation report with summary tables, 0–5 quality scores, completeness breakdown, and qualitative critiques.

---

## 📊 **Evaluation Metrics**

| Category                    | Metric                                     | Description                                                                                     |
| :-------------------------- | :----------------------------------------- | :---------------------------------------------------------------------------------------------- |
| **Retrieval Quality** | **Strict Hit Rate**                  | Exact recall matching BOTH the statutory document and section number (Doc AND Section Recall@k) |
| **Retrieval Quality** | **Section Hit Rate**                 | Proportion of inquiries where the required section is retrieved in Top-K candidates             |
| **Legal QA Score**    | **Average Score (0–5)**             | Overall response quality based on the Thai Legal QA rubric (Good / Perfect:$\ge$ 4/5)         |
| **Citation Quality**  | **Citation Precision / Recall / F1** | Accuracy and completeness of cited statutory clauses                                            |
| **Substance & Fact**  | **Rule / Penalty Correctness**       | Accuracy of statutory numbers, budget thresholds, deadlines, and penalties                      |
| **Substance & Fact**  | **Completeness vs GT**               | Full coverage of sub-questions, statutory conditions, and exceptions                            |
| **Trustworthiness**   | **No Hallucination Rate**            | Freedom from fabricated clauses or unsupported legal assertions                                 |
| **Conciseness**       | **Word Length Ratio**                | Ratio of agent word count relative to ground truth (measuring conciseness)                      |

---

## 📄 **License & Acknowledgments**

This project extends the original LegalGraphRAG framework, re-architecting it for the **Public Procurement and Supplies Administration Act, B.E. 2560 (2017)** (พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560) and related Ministry of Finance regulations of Thailand.
