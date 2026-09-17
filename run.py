import argparse
import os
import sys
import json
from tqdm import tqdm
import multiprocessing
import time
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig


def load_test_cases(datasets: str, datasets_path: str = "./datasets") -> List[Dict[str, Any]]:
    if os.path.exists(datasets) and os.path.isfile(datasets):
        case_file = datasets
    else:
        case_file = os.path.join(datasets_path, f"crime_data_{datasets}_small.json")
        if not os.path.exists(case_file):
            alt_file = os.path.join(datasets_path, datasets)
            if os.path.exists(alt_file) and os.path.isfile(alt_file):
                case_file = alt_file
            else:
                raise FileNotFoundError(f"Test dataset not found: {case_file}")
    
    with open(case_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    return cases


def process_cases_worker(
    cases: List[Dict[str, Any]],
    config_dict: Dict[str, Any],
    device: str,
    output_file: str,
    model_name: str
):
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    config = LegalGraphRAGConfig.from_dict(config_dict)
    
    config.model.device = device
    config.model.model_name = model_name
    # Workers load the graph read-only; the parent process owns persistence.
    config.graph.auto_build = False
    config.graph.auto_save = False
    
    rag = LegalGraphRAG(config=config)
    
    results = []
    section_hits = 0
    category_hits = 0
    
    try:
        for case in tqdm(cases, desc=f"Processing on {device} with {model_name}"):
            question = case.get("fact", "")
            true_category = case.get("crime", [])
            true_section = case.get("laws", [])
            ground_truth = case.get("ground_truth", "")
            
            case_res = rag.analyze_case(case)
            
            pred_answer = ""
            pred_direct_answer = ""
            pred_laws = []
            pred_category = []
            exceptions = ""
            
            if case_res and isinstance(case_res, list) and len(case_res) > 0:
                judge_result = case_res[0].get("judge_result", {})
                pred_answer = judge_result.get("answer", "")
                pred_direct_answer = judge_result.get("direct_answer", "")
                pred_laws = list(judge_result.get("applicable_laws", judge_result.get("law_article", [])))
                pred_category = list(judge_result.get("category", judge_result.get("charge_name", [])))
                exceptions = judge_result.get("exceptions_or_conditions", "")
                
                # Also include used laws from graph traversal
                used_laws = case_res[0].get("used_laws", [])
                for ul in used_laws:
                    entry = ul.get("entry", "")
                    if entry and entry not in pred_laws:
                        pred_laws.append(entry)

            # Check Section hit (does any predicted law contain true_section?)
            is_section_hit = False
            for ts in true_section:
                ts_clean = str(ts).strip()
                if not ts_clean:
                    continue
                for pl in pred_laws:
                    if ts_clean in str(pl) or str(pl) in ts_clean:
                        is_section_hit = True
                        break
                if is_section_hit:
                    break
            if is_section_hit:
                section_hits += 1

            # Check Category hit
            is_category_hit = False
            for tc in true_category:
                tc_clean = str(tc).strip()
                if not tc_clean:
                    continue
                for pc in pred_category:
                    if tc_clean in str(pc) or str(pc) in tc_clean:
                        is_category_hit = True
                        break
                if is_category_hit:
                    break
            if is_category_hit:
                category_hits += 1

            results.append({
                "id": case.get("id"),
                "question": question,
                "direct_answer": pred_direct_answer,
                "answer": pred_answer,
                "ground_truth": ground_truth,
                "expected_section": true_section,
                "predicted_laws": pred_laws,
                "is_section_hit": is_section_hit,
                "expected_category": true_category,
                "predicted_category": pred_category,
                "is_category_hit": is_category_hit,
                "exceptions_or_conditions": exceptions,
                # Backward compatibility keys
                "fact": question,
                "law_article": true_section,
                "judge_res": case_res,
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return section_hits, category_hits, len(cases)
    
    finally:
        if hasattr(rag, 'model') and hasattr(rag.model, 'release_model'):
            try:
                rag.model.release_model()
            except:
                pass


def run_evaluation(
    model_name: str,
    datasets: str = "THAI",
    dotenv_path: str = "configs/thai_procurement.env",
    devices: Optional[List[str]] = None,
    datasets_path: str = "./datasets",
    build_graph: bool = True,
    force_rebuild: bool = False,
    limit: Optional[int] = None
):
    config = LegalGraphRAGConfig.from_env_file(dotenv_path)
    
    output_dir = os.path.join(config.data.output_dir, datasets)
    os.makedirs(output_dir, exist_ok=True)
    if datasets_path == "./datasets" and config.data.datasets_path:
        datasets_path = config.data.datasets_path
    if not config.graph.graph_db_path:
        config.graph.graph_db_path = os.path.join(output_dir, f"{model_name}_graph_db.pkl")
        print(f"graph_db_path not configured; using {config.graph.graph_db_path}")
    
    # Build graph before starting parallel processes
    if build_graph:
        print("="*60)
        print("Building graph database...")
        print("="*60)
        
        # Use first device for graph construction (if devices specified), otherwise use config device
        if devices and len(devices) > 0:
            build_device = devices[0]
        else:
            build_device = config.model.device
        
        # Create configuration for graph construction (using first device)
        build_config = LegalGraphRAGConfig.from_dict(config.to_dict())
        build_config.model.device = build_device
        build_config.model.model_name = model_name
        # run_evaluation controls graph construction explicitly below.
        build_config.graph.auto_build = False
        build_config.graph.auto_save = False
        
        # Create LegalGraphRAG instance and build graph
        print(f"Using device {build_device} for graph construction...")
        rag_builder = LegalGraphRAG(config=build_config)
        rag_builder.build_graph(force_rebuild=force_rebuild)
        
        # Release model resources used for graph construction
        if hasattr(rag_builder, 'model') and hasattr(rag_builder.model, 'release_model'):
            try:
                rag_builder.model.release_model()
            except:
                pass
        
        print("="*60)
        print("Graph database ready!")
        print("="*60)
        print()
    elif not os.path.exists(config.graph.graph_db_path):
        raise FileNotFoundError(
            f"Graph database not found: {config.graph.graph_db_path}. "
            "Run without --no-build-graph first."
        )
    
    test_cases = load_test_cases(datasets, datasets_path)
    print(f"Loaded {len(test_cases)} test cases from {datasets} dataset")
    if limit is not None and limit > 0:
        test_cases = test_cases[:limit]
        print(f"Limiting evaluation to first {len(test_cases)} test cases (--limit {limit})")
    
    if devices is None:
        if config.model.device and config.model.device != "auto":
            devices = [config.model.device]
        else:
            try:
                import torch
                devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
            except ImportError:
                devices = ["cpu"]
    if not devices or len(devices) == 0:
        devices = ["cpu"]
    num_processes = len(devices)
    
    chunks = [[] for _ in range(num_processes)]
    for i, case in enumerate(test_cases):
        chunk_index = i % num_processes
        chunks[chunk_index].append(case)
    
    print(f"Split {len(test_cases)} cases into {num_processes} processes")
    for i, chunk in enumerate(chunks):
        print(f"  Process {i} ({devices[i]}): {len(chunk)} cases")
    
    config_dict = config.to_dict()
    
    pool = multiprocessing.Pool(processes=num_processes)
    async_results = []
    
    time_before = time.time()
    
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"{model_name}_results_part_{i}.json")
        async_results.append(
            pool.apply_async(
                process_cases_worker,
                args=(
                    chunk,
                    config_dict,
                    devices[i],
                    output_file,
                    model_name,
                ),
            )
        )
    
    pool.close()
    pool.join()
    
    time_after = time.time()
    elapsed_time = time_after - time_before
    
    total_section_hits = 0
    total_category_hits = 0
    total_cases = 0
    for res in async_results:
        sec_hits, cat_hits, count = res.get()
        total_section_hits += sec_hits
        total_category_hits += cat_hits
        total_cases += count
    
    sec_rate = (total_section_hits / total_cases * 100) if total_cases > 0 else 0.0
    cat_rate = (total_category_hits / total_cases * 100) if total_cases > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {datasets}")
    print(f"Total questions evaluated: {total_cases}")
    print(f"Section Retrieval Hit Rate: {total_section_hits}/{total_cases} ({sec_rate:.1f}%)")
    print(f"Category Match Rate: {total_category_hits}/{total_cases} ({cat_rate:.1f}%)")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    combined_results = []
    for i in range(len(chunks)):
        part_file = os.path.join(output_dir, f"{model_name}_results_part_{i}.json")
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                part_data = json.load(f)
                combined_results.extend(part_data)
            os.remove(part_file)
    
    combined_file = os.path.join(output_dir, f"{model_name}_results_combined.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined QA results saved to {combined_file}")
    
    stats_file = os.path.join(output_dir, f"{model_name}_stats.json")
    stats = {
        "model_name": model_name,
        "dataset": datasets,
        "total_cases": total_cases,
        "section_hits": total_section_hits,
        "section_hit_rate": sec_rate,
        "category_hits": total_category_hits,
        "category_hit_rate": cat_rate,
        "correct_count": total_section_hits,
        "elapsed_time": elapsed_time,
        "output_file": combined_file
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal Case Analysis with Different Models using LegalGraphRAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for analysis (e.g. openrouter, google/gemma-3-4b-it, qwen3, gpt4o_mini, etc.)",
    )
    default_dotenv = "configs/thai_procurement.env" if os.path.exists("configs/thai_procurement.env") else ".env"
    parser.add_argument(
        "--dotenv_path",
        type=str,
        default=default_dotenv,
        help="Path to the .env file (default: configs/thai_procurement.env)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="THAI",
        help="Dataset name (default: THAI)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="# of query to inference",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=None,
        help="Path to datasets directory (default: ./datasets)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use (e.g., cuda:2 cuda:3)",
    )
    parser.add_argument(
        "--no-build-graph",
        action="store_true",
        help="Skip graph construction (assume graph already exists)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild graph even if it already exists",
    )
    
    args = parser.parse_args()
    
    multiprocessing.set_start_method("spawn", force=True)
    
    run_evaluation(
        model_name=args.model,
        datasets=args.datasets,
        dotenv_path=args.dotenv_path,
        devices=args.devices,
        datasets_path=args.datasets_path if args.datasets_path else "./datasets",
        build_graph=not args.no_build_graph,
        force_rebuild=args.force_rebuild,
        limit=args.limit
    )
