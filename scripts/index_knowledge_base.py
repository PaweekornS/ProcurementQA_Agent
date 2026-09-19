#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
index_knowledge_base.py

Standalone Production Pipeline for Data Ingestion & Knowledge Graph / VectorDB Indexing:
1. Ingests statutory markdown files (typhoon_ocr) with structure-aware Page Macro Chunking (CRAG).
2. Ingests authoritative FAQ Q&A pairs (Excel) as knowledge nodes.
3. Generates corpus files (law_to_crime.json, cases_with_feature.json).
4. Builds Dense Embeddings & Knowledge Graph DB (openrouter_graph_db.pkl).
5. Pre-builds Sparse BM25 index with PyThaiNLP (attacut engine).

Usage:
    # Run full ingestion and indexing:
    python scripts/index_knowledge_base.py

    # Force rebuild even if index exists:
    python scripts/index_knowledge_base.py --force

    # Specify custom input data and output paths:
    python scripts/index_knowledge_base.py --procurement_dir ./procurement_data --output_path ./outputs/openrouter_graph_db.pkl
"""

import os
import sys
import argparse
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
import scripts.prepare_thai_corpus as corpus_builder


def run_indexing_pipeline(
    dotenv_path: str = "configs/thai_procurement.env",
    procurement_dir: str = "./procurement_data",
    output_db_path: str = None,
    force_rebuild: bool = False,
    skip_corpus_prep: bool = False,
):
    print("=" * 60)
    print("LEGAL-GRAPH-RAG: KNOWLEDGE BASE INDEXING PIPELINE")
    print("=" * 60)
    t_start = time.time()

    # 1. Load system configuration
    if not os.path.exists(dotenv_path):
        fallback_env = "configs/default.env"
        if os.path.exists(fallback_env):
            dotenv_path = fallback_env
            
    print(f"[Config] Loading settings from: {dotenv_path}")
    config = LegalGraphRAGConfig.from_env_file(dotenv_path)

    if output_db_path:
        config.graph.graph_db_path = output_db_path
    elif not config.graph.graph_db_path:
        config.graph.graph_db_path = "./outputs/openrouter_graph_db.pkl"

    os.makedirs(os.path.dirname(os.path.abspath(config.graph.graph_db_path)), exist_ok=True)

    # 2. Stage 1: Data Preprocessing & Macro Chunking (CRAG)
    if not skip_corpus_prep:
        print("\n" + "-" * 50)
        print("[STAGE 1] INGESTION & STRUCTURE-AWARE MACRO CHUNKING")
        print("-" * 50)
        
        typhoon_dir = os.path.join(procurement_dir, "typhoon_ocr")
        faq_file = os.path.join(procurement_dir, "FAQ_กรมบัญชีกลาง.xlsx")
        output_datas_dir = "./datas"
        os.makedirs(output_datas_dir, exist_ok=True)

        print(f"Ingesting statutes from: {typhoon_dir}")
        print(f"Ingesting FAQ pairs from: {faq_file}")

        chunks = corpus_builder.load_raw_chunks(
            typhoon_dir=typhoon_dir if os.path.exists(typhoon_dir) else None,
            faq_file=faq_file if os.path.exists(faq_file) else None,
        )

        law_output = os.path.join(output_datas_dir, "law_to_crime.json")
        cases_output = os.path.join(output_datas_dir, "cases_with_feature.json")

        corpus_builder.build_law_to_crime(chunks, law_output)
        corpus_builder.build_cases_with_feature(faq_file, cases_output)
        print("[Done] Stage 1 complete: Corpus files prepared with Macro Chunks.")
    else:
        print("\n[Skip] Skipping Stage 1 corpus preparation as requested.")

    # 3. Stage 2: Graph Database & Vector Construction
    print("\n" + "-" * 50)
    print("[STAGE 2] KNOWLEDGE GRAPH & VECTOR EMBEDDING INDEXING")
    print("-" * 50)
    print(f"Target Graph DB: {config.graph.graph_db_path}")

    if not force_rebuild and os.path.exists(config.graph.graph_db_path):
        size_mb = os.path.getsize(config.graph.graph_db_path) / (1024 * 1024)
        print(f"Graph DB already exists ({size_mb:.2f} MB). Use --force to overwrite.")
        return

    # Initialize builder
    config.graph.auto_build = False
    config.graph.auto_save = False
    rag_builder = LegalGraphRAG(config=config)

    print("Building knowledge nodes, relations, and embeddings...")
    rag_builder.build_graph(force_rebuild=True)

    # 4. Stage 3: Verification
    print("\n" + "-" * 50)
    print("[STAGE 3] INDEX VERIFICATION")
    print("-" * 50)
    if os.path.exists(config.graph.graph_db_path):
        size_mb = os.path.getsize(config.graph.graph_db_path) / (1024 * 1024)
        print(f"[Success] Created graph database: {config.graph.graph_db_path} ({size_mb:.2f} MB)")
    else:
        print("[Error] Graph database file was not created!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"[COMPLETE] INDEXING PIPELINE FINISHED in {time.time() - t_start:.2f}s!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Knowledge Base Ingestion & Vector / Graph Indexing Pipeline"
    )
    parser.add_argument(
        "--config",
        dest="dotenv_path",
        type=str,
        default="configs/thai_procurement.env",
        help="Path to environment config file",
    )
    parser.add_argument(
        "--procurement_dir",
        type=str,
        default="./procurement_data",
        help="Directory containing typhoon_ocr/ and FAQ_กรมบัญชีกลาง.xlsx",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Custom output path for graph db .pkl file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild graph even if output exists",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip re-chunking files and only rebuild vector/graph from ./datas/",
    )

    args = parser.parse_args()

    run_indexing_pipeline(
        dotenv_path=args.dotenv_path,
        procurement_dir=args.procurement_dir,
        output_db_path=args.output_path,
        force_rebuild=args.force,
        skip_corpus_prep=args.skip_corpus,
    )


if __name__ == "__main__":
    main()
