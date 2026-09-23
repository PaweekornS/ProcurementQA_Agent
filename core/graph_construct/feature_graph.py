import os
import ast
import hashlib
import threading
from collections import defaultdict
import numpy as np
import requests
import re
from .graph_db import GraphDBManager
from tqdm import tqdm


_embedding_provider = "local"
_embedding_api_url = "http://localhost:11434/api/embed"
_embedding_model = "unsloth/embeddinggemma-300m"
_tokenmind_api_key = None
_tokenmind_base_url = "https://tokenmind.abdul.in.th/v1"
_tokenmind_model = "BAAI/bge-m3"
_embedding_dim = 768
_local_embedder = None
_local_embedder_lock = threading.Lock()
_vector_cache = {}
_http_embedder_available = None


def _init_vector_cache():
    global _vector_cache, _embedding_dim
    if _vector_cache:
        return
    # Vector cache is maintained in-memory during execution


def configure_embedding(
    api_url=None,
    model=None,
    provider=None,
    tokenmind_api_key=None,
    tokenmind_base_url=None,
    tokenmind_model=None
):
    """Configure the embedding backend used by graph construction and retrieval."""
    global _embedding_api_url, _embedding_model, _embedding_provider
    global _tokenmind_api_key, _tokenmind_base_url, _tokenmind_model
    global _local_embedder, _http_embedder_available, _embedding_dim

    if provider:
        _embedding_provider = provider.lower()
        if _embedding_provider == "tokenmind":
            _embedding_dim = 1024
    if api_url:
        _embedding_api_url = api_url
    if model:
        if _embedding_model != model:
            _local_embedder = None
            _http_embedder_available = None
        _embedding_model = model
    if tokenmind_api_key:
        _tokenmind_api_key = tokenmind_api_key
    if tokenmind_base_url:
        _tokenmind_base_url = tokenmind_base_url.rstrip("/")
    if tokenmind_model:
        _tokenmind_model = tokenmind_model
    _init_vector_cache()


def batch_embed_tokenmind(texts, batch_size=16):
    """Batch embed texts using Tokenmind API"""
    api_key = _tokenmind_api_key or os.getenv("TOKENMIND_API_KEY") or os.getenv("tokenmind_api_key")
    base_url = (_tokenmind_base_url or os.getenv("TOKENMIND_BASE_URL") or "https://tokenmind.abdul.in.th/v1").rstrip("/")
    model = _tokenmind_model or os.getenv("TOKENMIND_EMBEDDING_MODEL") or "BAAI/bge-m3"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = [str(t)[:2000] for t in texts[i:i + batch_size]]
        try:
            payload = {"model": model, "input": chunk}
            resp = requests.post(f"{base_url}/embeddings", headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            res_data = resp.json()
            items = res_data.get("data", [])
            for item in items:
                all_embeddings.append(item["embedding"])
        except Exception as e:
            print(f"[feature_graph] Tokenmind batch embed failed at offset {i}: {e}, falling back to sequential")
            for t in chunk:
                all_embeddings.append(get_embedding(t))
    return all_embeddings


def get_embedding(text):
    global _embedding_dim, _http_embedder_available, _local_embedder
    global _embedding_provider, _tokenmind_api_key, _tokenmind_base_url, _tokenmind_model

    if not text:
        return [0.0] * _embedding_dim

    _init_vector_cache()
    # 1. Check in-memory cache
    if text in _vector_cache:
        return _vector_cache[text]
    if text[:200] in _vector_cache:
        return _vector_cache[text[:200]]

    # 2. Try Tokenmind API if configured as active provider
    provider = (_embedding_provider or os.getenv("embedding_provider", "local")).lower()
    if provider == "tokenmind":
        api_key = _tokenmind_api_key or os.getenv("TOKENMIND_API_KEY") or os.getenv("tokenmind_api_key")
        base_url = (_tokenmind_base_url or os.getenv("TOKENMIND_BASE_URL") or "https://tokenmind.abdul.in.th/v1").rstrip("/")
        model = _tokenmind_model or os.getenv("TOKENMIND_EMBEDDING_MODEL") or "BAAI/bge-m3"

        if api_key and base_url:
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "input": str(text)[:2000]
                }
                response = requests.post(f"{base_url}/embeddings", json=payload, headers=headers, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                data = result.get('data') or result.get('embeddings')
                if data and len(data) > 0:
                    item = data[0]
                    vec = item['embedding'] if isinstance(item, dict) and 'embedding' in item else item
                    _embedding_dim = len(vec)
                    _vector_cache[text] = vec
                    return vec
            except Exception as e:
                print(f"[feature_graph] Tokenmind embedding failed: {e}")

    # 3. Try HTTP embedding endpoint (e.g. Ollama with unsloth/embeddinggemma-300m)
    if _http_embedder_available is not False and _embedding_api_url:
        try:
            data = {
                "model": _embedding_model,
                "input": text
            }
            response = requests.post(_embedding_api_url, json=data, timeout=1.0)
            response.raise_for_status()
            result = response.json()
            embeddings = result.get('embeddings') or result.get('data')
            if embeddings and embeddings[0]:
                if isinstance(embeddings[0], dict) and 'embedding' in embeddings[0]:
                    vec = embeddings[0]['embedding']
                else:
                    vec = embeddings[0]
                _embedding_dim = len(vec)
                _vector_cache[text] = vec
                _http_embedder_available = True
                return vec
        except Exception:
            _http_embedder_available = False

    # 4. Try local SentenceTransformer as fallback
    if _local_embedder is None:
        with _local_embedder_lock:
            if _local_embedder is None:
                try:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    embed_name = os.getenv("embedding_model") or os.getenv("EMBEDDING_MODEL") or _embedding_model
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    _local_embedder = SentenceTransformer(embed_name, device=device)
                except Exception as e:
                    print(f"[feature_graph] Failed loading local embedder: {e}")
                    _local_embedder = False

    if _local_embedder:
        try:
            vec = _local_embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            emb = vec.tolist()
            _embedding_dim = len(emb)
            _vector_cache[text] = emb
            return emb
        except Exception:
            pass

    # 5. Deterministic fallback pseudo-vector based on hash matching active dimension
    h = hashlib.sha256(text.encode('utf-8')).digest()
    dim = _embedding_dim
    np.random.seed(int.from_bytes(h[:4], 'big'))
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    emb = v.tolist()
    _vector_cache[text] = emb
    return emb


def summarize_texts(model, text):
    try:
        from core.prompt import get_prompt
        prefix = ""
        try:
            prefix = get_prompt("SUMMARIZE_TEXTS_INPUT_PREFIX")
        except Exception:
            pass
        return model.generate_response(
            get_prompt("SUMMARIZE_TEXTS_PROMPT") + prefix + text,
            max_length=512
        ).strip()
    except Exception as e:
        print(f"[summarize_texts fallback] {e}")
        return text[:300]


def rerank_clusters(model, clusters, query_text):
    if not clusters:
        return []
    # Try GPU Cross-Encoder Reranker first
    try:
        from .hybrid_reranker import get_reranker
        reranker = get_reranker()
        if reranker and reranker.model is not None:
            candidates = [{'code': c['code'], 'description': c.get('summary', '')} for c in clusters]
            reranked = reranker.rerank(query_text, candidates, top_k=len(candidates))
            return [c['code'] for c in reranked]
    except Exception:
        pass

    # Fallback to LLM if requested
    try:
        from core.prompt import get_prompt
        cluster_summaries = "\n".join(
            [f"code{c['code']}：{c['summary']}\n" for c in clusters])
        prompt = get_prompt("RERANK_CLUSTERS_PROMPT_TEMPLATE").format(
            cluster_summaries=cluster_summaries,
            query_text=query_text
        )
        response = model.generate_response(prompt, max_length=512)
        match = re.search(r"\[([\d,\s]+)\]", response)
        if match:
            ranked_codes = [int(code.strip()) for code in match.group(1).split(',') if code.strip().isdigit()]
            if ranked_codes:
                return ranked_codes
    except Exception:
        pass

    return [c['code'] for c in clusters]


def rerank(model, query_text, neighbors):
    if not neighbors:
        return []

    # Use GPU Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
    try:
        from .hybrid_reranker import get_reranker
        reranker_model = os.getenv("reranker_model", "BAAI/bge-reranker-v2-m3")
        reranker_device = os.getenv("reranker_device", "cuda:0")
        reranker_thresh = float(os.getenv("reranker_threshold", "0.20"))
        
        reranker = get_reranker(model_name=reranker_model, device=reranker_device, threshold=reranker_thresh)
        if reranker and reranker.model is not None:
            reranked = reranker.rerank(query_text, neighbors, top_k=len(neighbors), threshold=reranker_thresh)
            for idx, n in enumerate(reranked):
                n['rank'] = idx + 1
            return reranked
    except Exception as e:
        print(f"[Reranker fallback] GPU reranker unavailable: {e}")

    # Fallback to LLM Prompting if CrossEncoder is not available
    try:
        from core.prompt import get_prompt
        neighbor_summaries = "\n".join(
            [f"code{n.get('rank', i+1)}：{n.get('description', '')}\n" for i, n in enumerate(neighbors)])
        prompt = get_prompt("RERANK_PROMPT_TEMPLATE").format(
            neighbor_summaries=neighbor_summaries,
            query_text=query_text
        )
        response = model.generate_response(prompt, max_length=512)
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            list_str = response[first_bracket:last_bracket + 1]
            ranked_indices = eval(list_str)
            neighbors = [n for n in neighbors if n.get('rank') in ranked_indices]
            neighbors = sorted(neighbors, key=lambda x: ranked_indices.index(x.get('rank')))
            return neighbors
    except Exception as e:
        print(f"Error parsing LLM rerank response: {e}")

    return neighbors[:3]


def store_nodes_with_embeddings(nodes_data):
    """
    nodes_data: Dict with keys 'case', 'law', 'crime', each containing a list of node dicts
    """
    store_nodes(nodes_data)
    build_relationships()


def store_nodes(nodes_data):
    """
    存储节点到内存图数据库
    nodes_data: Dict with keys 'case', 'law', 'crime', each containing a list of node dicts
    """
    db = GraphDBManager.get_db()
    case_nodes_data, law_nodes_data, crime_nodes_data = nodes_data[
        'case'], nodes_data['law'], nodes_data['crime']

    # 存储Case节点
    for node in tqdm(case_nodes_data, desc="Storing case nodes"):
        db.add_node(
            node['id'], 
            'Cases',
            {
                'description': node.get('description'),
                'embedding': node.get('embedding'),
                'caseId': node.get('caseId'),
                'crime': node.get('crime'),
                'law': node.get('law')
            }
        )

    # 存储Law节点
    for node in tqdm(law_nodes_data, desc="Storing law nodes"):
        db.add_node(
            node['id'],
            'Laws',
            {
                'entry': node.get('entry'),
                'description': node.get('description'),
                'embedding': node.get('embedding'),
                'crimes': node.get('crimes'),
                'judge_dep': node.get('judge_dep'),
                'related_laws': node.get('related_laws'),
                'insights': ''
            }
        )

    # 存储Crime节点
    for node in tqdm(crime_nodes_data, desc="Storing crime nodes"):
        db.add_node(
            node['id'],
            'Crimes',
            {
                'description': node.get('description'),
                'embedding': node.get('embedding')
            }
        )


def build_relationships():
    """
    在图中建立节点之间的关系
    """
    db = GraphDBManager.get_db()
    
    # 创建Case到Law的关系（基于entry匹配）
    # 从图中检索所有Case节点及其law属性
    case_nodes = db.get_nodes_by_type('Cases')
    
    for case_node in tqdm(case_nodes, desc="Linking cases to laws"):
        case_id = case_node['id']
        law_entries = case_node.get('law')
        
        if not law_entries:
            continue

        for law_entry in law_entries:
            law_entry_str = str(law_entry).strip()
            if not law_entry_str:
                continue
            # 找到对应的Law节点
            law_found = False
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Laws':
                    entry_val = str(node_info['data'].get('entry', '')).strip()
                    if entry_val == law_entry_str:
                        db.add_edge(case_id, node_id, 'RELATES_TO_LAW')
                        law_found = True
                        break
                    elif law_entry_str in entry_val or entry_val in law_entry_str:
                        db.add_edge(case_id, node_id, 'RELATES_TO_LAW')
                        law_found = True
                        break

    # 创建Law到Crime的关系（基于罪名描述匹配）
    # 从图中检索所有Law节点及其crimes属性
    law_nodes = db.get_nodes_by_type('Laws')

    for law_node in tqdm(law_nodes, desc="Linking laws to crimes"):
        law_id = law_node['id']
        crime_descriptions = law_node.get('crimes')
        
        if not crime_descriptions:
            continue

        for crime_desc in crime_descriptions:
            # 检查是否已存在关系
            existing_neighbors = db.get_neighbors(law_id, 'RELATED_CRIME')
            if existing_neighbors:
                # 检查是否已经有匹配的crime
                found = False
                for crime_id in existing_neighbors:
                    crime_data = db.get_node(crime_id)
                    if crime_data and crime_data.get('description') == crime_desc:
                        found = True
                        break
                if found:
                    continue

            # 尝试精确匹配
            crime_found = False
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Crimes' and node_info['data'].get('description') == crime_desc:
                    db.add_edge(law_id, node_id, 'RELATED_CRIME', {'match_type': 'exact'})
                    crime_found = True
                    break

            # 如果精确匹配失败，尝试模糊匹配
            if not crime_found:
                for node_id, node_info in db.nodes_data.items():
                    if node_info['type'] == 'Crimes':
                        desc = node_info['data'].get('description', '')
                        if crime_desc in desc or desc in crime_desc:
                            db.add_edge(law_id, node_id, 'RELATED_CRIME', {'match_type': 'fuzzy'})
                            break
    
    # 仅针对中国刑法数据集（所有entry均为数字）删除entry <= 101的法条总则节点
    law_entries = [node_info['data'].get('entry') for node_info in db.nodes_data.values() if node_info['type'] == 'Laws']
    is_cail_corpus = (
        len(law_entries) > 0
        and all(str(e).isdigit() for e in law_entries if e is not None)
        and any(int(e) <= 101 for e in law_entries if e is not None and str(e).isdigit())
    )
    if is_cail_corpus:
        nodes_to_delete = []
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Laws':
                entry = node_info['data'].get('entry')
                try:
                    if entry is not None and int(entry) <= 101:
                        nodes_to_delete.append(node_id)
                except (ValueError, TypeError):
                    pass
        
        node_count = len(nodes_to_delete)
        if node_count > 0:
            print(f"将要删除 {node_count} 个Law节点及其所有关系")
            for node_id in nodes_to_delete:
                db.graph.remove_node(node_id)
                del db.nodes_data[node_id]
                for node_type in db.embeddings:
                    if node_id in db.embeddings[node_type]:
                        del db.embeddings[node_type][node_id]
                        db._update_vector_index(node_type)
            print(f"已成功删除 {node_count} 个Law节点及其所有关系")


def run_knn(top_k=3):
    db = GraphDBManager.get_db()
    
    # 获取所有Cases节点
    case_nodes = db.get_nodes_by_type('Cases')
    if len(case_nodes) < 2:
        return
    
    # 获取所有Cases的embeddings
    case_embeddings = []
    case_ids = []
    for node in case_nodes:
        emb = node.get('embedding')
        if emb is not None:
            case_embeddings.append(np.array(emb))
            case_ids.append(node['id'])
    
    if len(case_embeddings) < 2:
        return
    
    case_embeddings = np.array(case_embeddings)
    
    # 计算相似度矩阵并创建SIMILAR_TO关系
    for i in tqdm(range(len(case_ids)), desc="Running KNN"):
        similarities = []
        for j in range(len(case_ids)):
            if i != j:
                sim = db.cosine_similarity(case_embeddings[i], case_embeddings[j])
                similarities.append((j, sim))
        
        # 选择top_k个最相似的
        similarities.sort(key=lambda x: x[1], reverse=True)
        for j, score in similarities[:top_k]:
            db.add_edge(case_ids[i], case_ids[j], 'SIMILAR_TO', {'score': score})


def extract_doc_name(data):
    """Extract clean legal document/file name from a Law node's properties."""
    rel = data.get('related_laws', [])
    if isinstance(rel, str):
        try:
            rel = ast.literal_eval(rel)
        except Exception:
            rel = [rel]
    if isinstance(rel, list):
        for r in rel:
            if isinstance(r, str) and (r.endswith('.md') or '/' in r or '\\' in r):
                base = os.path.basename(r)
                return base[:-3] if base.endswith('.md') else base
    entry = data.get('entry', '')
    if '|' in entry:
        return entry.split('|')[0].strip()
    crimes = data.get('crimes', [])
    if crimes:
        return crimes[-1] if isinstance(crimes, list) else str(crimes)
    return "Unknown_Document"


def create_clusters(model=None):
    """
    Construct hierarchical legal document clusters grouped by source file/document.
    Replaces slow Louvain/KNN + LLM summary with deterministic, instant document grouping.
    Hierarchy: Document Cluster -> Law sections (Laws nodes).
    """
    db = GraphDBManager.get_db()
    
    # 1. Group all Laws nodes by source document/file name
    doc_laws = defaultdict(list)
    for node_id, node_info in db.nodes_data.items():
        if node_info['type'] == 'Laws':
            doc_name = extract_doc_name(node_info['data'])
            doc_laws[doc_name].append(node_id)
            
    print(f"Grouped into {len(doc_laws)} legal document clusters.")
    
    # 2. For each document, create a Cluster node
    for doc_name, law_ids in tqdm(doc_laws.items(), desc="Creating document clusters"):
        cluster_id = f"doc_{hashlib.md5(doc_name.encode('utf-8')).hexdigest()[:12]}"
        
        # Calculate cluster embedding using document title
        doc_embedding = get_embedding(doc_name)
        
        # Store Cluster node with document metadata
        db.add_node(
            cluster_id,
            'Cluster',
            {
                'name': doc_name,
                'summary': doc_name,
                'doc_name': doc_name,
                'embedding': doc_embedding,
                'law_count': len(law_ids),
            }
        )
        
        # Connect Laws nodes to this Document Cluster
        for law_id in law_ids:
            db.add_edge(law_id, cluster_id, 'BELONGS_TO')
            db.add_edge(cluster_id, law_id, 'CONTAINS_LAW')
            db.update_node(law_id, {'communityId': cluster_id})
            
    # 3. Connect Cases to clusters if case cites or relates to laws in that document
    for node_id, node_info in db.nodes_data.items():
        if node_info['type'] == 'Cases':
            law_neighbors = db.get_neighbors(node_id, 'RELATES_TO_LAW')
            connected_clusters = set()
            for law_id in law_neighbors:
                for c_id in db.get_neighbors(law_id, 'BELONGS_TO'):
                    if c_id not in connected_clusters:
                        db.add_edge(node_id, c_id, 'BELONGS_TO')
                        connected_clusters.add(c_id)


def search_similar_nodes_top(model, query_embedding, query_text, top_k=5):
    """
    Search similar nodes using Hierarchical Document Clusters:
    1. Find top matching legal documents (Cluster nodes) via vector similarity.
    2. Traverse from top documents to their constituent law sections (Laws nodes).
    3. Rerank candidate laws using Cross-Encoder / vector similarity.
    """
    db = GraphDBManager.get_db()
    
    # 1. Find most similar Document Clusters
    cluster_results = db.find_similar_nodes(query_embedding, 'Cluster', top_k=min(6, top_k * 2))
    if not cluster_results:
        return [], [], []

    clusters = []
    for ids, record in enumerate(cluster_results):
        clusters.append({
            'code': ids,
            'cluster_id': record['id'],
            'summary': record.get('summary', record.get('name', '')),
            'similarity': record.get('similarity', 0.0)
        })

    cluster_ids = rerank_clusters(model, clusters, query_text)
    if not cluster_ids:
        cluster_ids = [0]

    # 2. Collect candidate laws belonging to top matching Document Clusters
    candidate_laws = []
    seen_law_ids = set()
    cluster_cases = []
    seen_case_ids = set()

    for c_idx in cluster_ids[:3]:
        if c_idx >= len(clusters):
            continue
        cluster_node_id = clusters[c_idx]['cluster_id']
        
        # Collect member laws from CONTAINS_LAW or BELONGS_TO
        law_neighbors = db.get_neighbors(cluster_node_id, 'CONTAINS_LAW')
        if not law_neighbors:
            for nid, ninfo in db.nodes_data.items():
                if ninfo['type'] == 'Laws' and cluster_node_id in db.get_neighbors(nid, 'BELONGS_TO'):
                    law_neighbors.append(nid)

        for law_id in law_neighbors:
            if law_id not in seen_law_ids:
                law_data = db.get_node(law_id)
                if law_data:
                    emb = law_data.get('embedding')
                    sim = 0.0
                    if emb is not None and query_embedding is not None:
                        sim = db.cosine_similarity(query_embedding, np.array(emb))
                    candidate_laws.append({
                        'id': law_id,
                        'entry': law_data.get('entry', ''),
                        'description': law_data.get('description', ''),
                        'crimes': law_data.get('crimes', []),
                        'judge_dep': law_data.get('judge_dep', []),
                        'related_laws': law_data.get('related_laws', []),
                        'insights': law_data.get('insights', ''),
                        'similarity': sim
                    })
                    seen_law_ids.add(law_id)

        # Collect cases linked to this cluster if any
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Cases' and node_id not in seen_case_ids:
                if cluster_node_id in db.get_neighbors(node_id, 'BELONGS_TO'):
                    cluster_cases.append({
                        'id': node_id,
                        'description': node_info['data'].get('description', ''),
                        'caseId': node_info['data'].get('caseId', ''),
                    })
                    seen_case_ids.add(node_id)

    # 3. Sort candidate laws by similarity and rerank
    candidate_laws.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
    laws_pool = candidate_laws[:top_k * 4]

    for idx, law in enumerate(laws_pool):
        law['rank'] = idx + 1

    reranked_laws = rerank(model, query_text, laws_pool)
    for law in reranked_laws:
        if 'rerank_score' not in law:
            law['rerank_score'] = law.get('similarity', 0.0)

    return clusters, cluster_cases[:top_k], reranked_laws[:top_k]


_bm25_initialized = False
_bm25_build_lock = threading.Lock()


def _ensure_bm25_index(db):
    """Build BM25 index on all Laws and Cases nodes in the graph if not already built."""
    global _bm25_initialized
    if _bm25_initialized:
        return

    with _bm25_build_lock:
        if _bm25_initialized:
            return
        try:
            from .hybrid_reranker import get_bm25_index
            bm25_idx = get_bm25_index()
            docs = []
            for node_id, node_info in db.nodes_data.items():
                ntype = node_info.get('type')
                data = node_info.get('data', {})
                if ntype in ('Laws', 'Cases'):
                    # Gather descriptive text
                    text = data.get('description', '')
                    if ntype == 'Laws':
                        entry = data.get('entry', '')
                        if entry:
                            text = f"{entry}\n{text}"
                    docs.append({
                        'id': node_id,
                        'type': ntype,
                        'text': text,
                        'data': data
                    })
            if docs:
                bm25_idx.build_index(docs)
                print(f"[HybridRetrieval] Built Thai BM25 index with {len(docs)} legal and case nodes.")
            _bm25_initialized = True

            # Warm up GPUReranker safely in the same lock so workers don't race on GPU allocation
            from .hybrid_reranker import get_reranker
            reranker_model = os.getenv("reranker_model", "BAAI/bge-reranker-v2-m3")
            reranker_device = os.getenv("reranker_device", "cuda:0")
            reranker_thresh = float(os.getenv("reranker_threshold", "0.20"))
            get_reranker(model_name=reranker_model, device=reranker_device, threshold=reranker_thresh)
        except Exception as e:
            print(f"[HybridRetrieval] Could not initialize BM25/Reranker index: {e}")


def search_similar_nodes_direct(model, query_embedding, query_text, top_k=5):
    db = GraphDBManager.get_db()
    use_hybrid = os.getenv("hybrid_retrieval", "True").lower() == "true"
    dense_top_k = int(os.getenv("dense_top_k", "30"))
    bm25_top_k = int(os.getenv("bm25_top_k", "30"))
    reranker_thresh = float(os.getenv("reranker_threshold", "0.20"))
    reranker_model = os.getenv("reranker_model", "BAAI/bge-reranker-v2-m3")
    reranker_device = os.getenv("reranker_device", "cuda:0")

    if not use_hybrid:
        # Fallback to classical dense-only Cases retrieval
        neighbor_results = db.find_similar_nodes(query_embedding, 'Cases', top_k=top_k)
        if not neighbor_results:
            return [], []
        neighbors = []
        for record in neighbor_results:
            neighbors.append({
                'id': record['id'],
                'description': record.get('description', ''),
                'caseId': record.get('caseId', ''),
                'similarity': record['similarity'],
                'type': 'Cases'
            })
        neighbors = rerank(model, query_text, neighbors)
        cases, laws = [], []
        for neighbor in neighbors:
            law_neighbors = db.get_neighbors(neighbor['id'], 'RELATES_TO_LAW')
            for law_id in law_neighbors:
                law_data = db.get_node(law_id)
                if law_data:
                    laws.append({'id': law_id, **law_data})
            cases.append(neighbor)
        return cases, laws

    # --- HYBRID RETRIEVAL BRANCH ---
    _ensure_bm25_index(db)

    # 1. Sparse Search (BM25 with PyThaiNLP + Numeric Query Expansion)
    from .hybrid_reranker import get_bm25_index, get_reranker, weighted_rrf, expand_numeric_query
    bm25_idx = get_bm25_index()
    bm25_query = expand_numeric_query(query_text)
    sparse_raw = bm25_idx.search(bm25_query, top_k=bm25_top_k)
    sparse_results = []
    for doc, score in sparse_raw:
        sparse_results.append(({
            'id': doc['id'],
            'type': doc['type'],
            'description': doc.get('text', ''),
            'data': doc.get('data', {})
        }, score))

    # 2. Dense Search (Embedding Cosine Similarity) on Laws and Cases
    dense_results = []
    law_records = db.find_similar_nodes(query_embedding, 'Laws', top_k=dense_top_k)
    for rec in law_records:
        dense_results.append(({
            'id': rec['id'],
            'type': 'Laws',
            'description': rec.get('description', ''),
            'data': rec
        }, rec.get('similarity', 0.0)))

    case_records = db.find_similar_nodes(query_embedding, 'Cases', top_k=dense_top_k)
    for rec in case_records:
        dense_results.append(({
            'id': rec['id'],
            'type': 'Cases',
            'description': rec.get('description', ''),
            'data': rec
        }, rec.get('similarity', 0.0)))

    # 3. Weighted RRF Fusion
    fused_candidates = weighted_rrf(
        dense_results,
        sparse_results,
        dense_weight=1.0,
        sparse_weight=1.0,
        rrf_k=int(os.getenv("rrf_k", "60"))
    )

    # 4. GPU/CPU Cross-Encoder Reranker with Relevance Gate (>= threshold)
    # Default candidate pool: 50 on GPU, 15 on CPU for fast sub-8s latency
    default_pool_size = 15 if "cpu" in str(reranker_device).lower() else 50
    pool_size = int(os.getenv("rerank_pool_size", default_pool_size))
    rerank_pool = fused_candidates[:pool_size]
    reranker = get_reranker(model_name=reranker_model, device=reranker_device, threshold=reranker_thresh)
    if reranker and reranker.model is not None:
        top_candidates = reranker.rerank(
            query_text,
            rerank_pool,
            top_k=top_k,
            threshold=reranker_thresh
        )
    else:
        top_candidates = fused_candidates[:top_k]

    # 5. Graph Traversal & Context Augmentation
    cases = []
    laws = []
    seen_law_ids = set()
    seen_case_ids = set()

    for cand in top_candidates:
        node_id = cand.get('id')
        node_type = cand.get('type')
        data = cand.get('data', {})

        if node_type == 'Laws':
            if node_id not in seen_law_ids:
                laws.append({
                    'id': node_id,
                    'entry': data.get('entry'),
                    'description': data.get('description'),
                    'crimes': data.get('crimes'),
                    'judge_dep': data.get('judge_dep'),
                    'related_laws': data.get('related_laws'),
                    'insights': data.get('insights', ''),
                    'rerank_score': cand.get('rerank_score', 1.0)
                })
                seen_law_ids.add(node_id)

            # Traverse to related Laws connected in Graph
            neighbors = db.get_neighbors(node_id, 'RELATED_TO')
            for n_id in neighbors:
                if n_id not in seen_law_ids:
                    n_data = db.get_node(n_id)
                    if n_data:
                        laws.append({
                            'id': n_id,
                            'entry': n_data.get('entry'),
                            'description': n_data.get('description'),
                            'crimes': n_data.get('crimes'),
                            'judge_dep': n_data.get('judge_dep'),
                            'related_laws': n_data.get('related_laws'),
                            'insights': n_data.get('insights', '')
                        })
                        seen_law_ids.add(n_id)

        elif node_type == 'Cases':
            if node_id not in seen_case_ids:
                cases.append({
                    'id': node_id,
                    'description': data.get('description', cand.get('description', '')),
                    'caseId': data.get('caseId', ''),
                    'rerank_score': cand.get('rerank_score', 1.0),
                })
                seen_case_ids.add(node_id)

            # Traverse from Case to Laws via RELATES_TO_LAW
            law_neighbors = db.get_neighbors(node_id, 'RELATES_TO_LAW')
            for law_id in law_neighbors:
                if law_id not in seen_law_ids:
                    law_data = db.get_node(law_id)
                    if law_data:
                        laws.append({
                            'id': law_id,
                            'entry': law_data.get('entry'),
                            'description': law_data.get('description'),
                            'crimes': law_data.get('crimes'),
                            'judge_dep': law_data.get('judge_dep'),
                            'related_laws': law_data.get('related_laws'),
                            'insights': law_data.get('insights', '')
                        })
                        seen_law_ids.add(law_id)

    return cases, laws


def query_similar_nodes_naive(model, query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    db = GraphDBManager.get_db()
    neighbor_results = db.find_similar_nodes(query_embedding, 'Cases', top_k=top_k)

    if not neighbor_results:
        return []

    neighbors = []
    for record in neighbor_results:
        neighbors.append({
            'id': record['id'],
            'description': record.get('description', ''),
            'caseId': record.get('caseId', ''),
            'similarity': record['similarity']
        })
    neighbors = sorted(
        neighbors, key=lambda x: x['similarity'], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor['rank'] = ids + 1

    return neighbors


def query_similar_nodes(model, query_text, retrieve_config):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return {}, [], []

    # 调用两个查询函数
    if retrieve_config["top_retrieve"]:
        top_result_clusters, top_result_cases, top_result_laws = search_similar_nodes_top(
            model, query_embedding, query_text, top_k=retrieve_config["top_retrieve_top_k"])
    else:
        top_result_clusters, top_result_cases, top_result_laws = [], [], []
    if retrieve_config["direct_retrieve"]:
        direct_result_cases, direct_result_laws = search_similar_nodes_direct(
            model, query_embedding, query_text, top_k=retrieve_config["direct_retrieve_top_k"])
    else:
        direct_result_cases, direct_result_laws = [], []

    # 整理结果
    result_cases = []
    seen_ids_cases = set()  # 用于去重
    result_laws = []
    seen_ids_laws = set()  # 用于去重

    # 1. Process search_similar_nodes_direct first (BM25 + Dense + GPU Reranker)
    if direct_result_laws:
        for neighbor in direct_result_laws:
            if neighbor.get('id') and neighbor['id'] not in seen_ids_laws:
                result_laws.append({
                    'id': neighbor['id'],
                    'entry': neighbor.get('entry', ''),
                    'description': neighbor.get('description', ''),
                    'crimes': neighbor.get('crimes', []),
                    'judge_dep': neighbor.get('judge_dep', []),
                    'related_laws': neighbor.get('related_laws', []),
                    'rerank_score': neighbor.get('rerank_score', 0.0)
                })
                seen_ids_laws.add(neighbor['id'])

    # 2. Process search_similar_nodes_top (cluster-level graph traversal)
    if top_result_laws:
        for neighbor in top_result_laws:
            if neighbor.get('id') and neighbor['id'] not in seen_ids_laws:
                result_laws.append({
                    'id': neighbor['id'],
                    'entry': neighbor.get('entry', ''),
                    'description': neighbor.get('description', ''),
                    'crimes': neighbor.get('crimes', []),
                    'judge_dep': neighbor.get('judge_dep', []),
                    'related_laws': neighbor.get('related_laws', []),
                    'rerank_score': neighbor.get('rerank_score', 0.0)
                })
                seen_ids_laws.add(neighbor['id'])

    result_laws.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)

    # 3. Process direct cases
    if direct_result_cases:
        for idx, neighbor in enumerate(direct_result_cases):
            if neighbor.get('id') and neighbor['id'] not in seen_ids_cases:
                result_cases.append({
                    'id': neighbor['id'],
                    'description': neighbor.get('description', ''),
                    'caseId': neighbor.get('caseId', ''),
                    'rank': neighbor.get('rank', idx + 1),
                    'rerank_score': neighbor.get('rerank_score', 0.0)
                })
                seen_ids_cases.add(neighbor['id'])

    # 4. Process top cases
    if top_result_cases:
        for idx, neighbor in enumerate(top_result_cases):
            if neighbor.get('id') and neighbor['id'] not in seen_ids_cases:
                result_cases.append({
                    'id': neighbor['id'],
                    'description': neighbor.get('description', ''),
                    'caseId': neighbor.get('caseId', ''),
                    'rank': neighbor.get('rank', idx + 1),
                    'rerank_score': neighbor.get('rerank_score', 0.0)
                })
                seen_ids_cases.add(neighbor['id'])

    result_cases.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
    original_retrieved_res = {
        "top": {
            "clusters": top_result_clusters,
            "cases": top_result_cases,
            "laws": top_result_laws
        },
        "direct": {
            "cases": direct_result_cases,
            "laws": direct_result_laws
        },
        "augmented": []
    }

    return original_retrieved_res, result_cases, result_laws


def query_similar_laws_naive(query_text, top_k=1):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    db = GraphDBManager.get_db()
    law_results = db.find_similar_nodes(query_embedding, 'Laws', top_k=top_k)

    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for law_record in law_results:
        entry = law_record.get('entry')
        if entry is not None and entry not in seen_law_ids:
            result_laws.append({
                'entry': entry,
                'similarity': law_record['similarity']
            })
            seen_law_ids.add(entry)

    return result_laws


def query_similar_laws(crime_list, top_k=1):
    """
    Query law nodes related to the most similar crime nodes based on a list of crime descriptions.

    Args:
        crime_list (list[str]): List of crime descriptions as strings.
        top_k (int): Number of top similar crime nodes to retrieve per crime description.

    Returns:
        list[dict]: List of law nodes with their details, deduplicated.
    """
    db = GraphDBManager.get_db()
    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for crime in crime_list:
        # Convert crime description to embedding
        crime_embedding = get_embedding(crime)
        if crime_embedding is None:
            continue  # Skip if embedding generation fails

        # Query the most similar crime nodes
        crime_results = db.find_similar_nodes(crime_embedding, 'Crimes', top_k=top_k)

        # Process each similar crime node
        for crime_record in crime_results:
            crime_id = crime_record['id']
            crime_similarity = crime_record['similarity']

            # Query law nodes related to this crime node
            # 找到所有指向该crime的Laws节点
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Laws':
                    neighbors = db.get_neighbors(node_id, 'RELATED_CRIME')
                    if crime_id in neighbors:
                        law_data = node_info['data']
                        law_id = node_id
                        if law_id not in seen_law_ids:
                            result_laws.append({
                                'id': law_id,
                                'entry': law_data.get('entry'),
                                'description': law_data.get('description'),
                                'crimes': law_data.get('crimes'),
                                'judge_dep': law_data.get('judge_dep'),
                                'related_laws': law_data.get('related_laws'),
                                'insights': law_data.get('insights', ''),
                                'crime_similarity': crime_similarity
                            })
                            seen_law_ids.add(law_id)

    # Sort results by crime similarity (descending) and assign ranks
    result_laws = sorted(
        result_laws, key=lambda x: x['crime_similarity'], reverse=True)
    for rank, law in enumerate(result_laws, 1):
        law['rank'] = rank

    return result_laws


def update_insights_in_graph(law_id, insights):
    db = GraphDBManager.get_db()
    db.update_node(law_id, {'insights': insights})


def construct_feature_graph(model, nodes_data):
    import torch
    GraphDBManager.initialize()

    case_nodes_data = nodes_data['case']
    law_nodes_data = nodes_data['law']
    crime_nodes_data = nodes_data['crime']

    # Check if Tokenmind embedding provider is active
    active_provider = (_embedding_provider or os.getenv("embedding_provider", "local")).lower()
    if active_provider == "tokenmind":
        print(f"[GraphConstruct] Acceleration: Batch encoding nodes with Tokenmind API ({_tokenmind_model})...")
        case_texts = [str(n.get('description', ''))[:1500] for n in case_nodes_data]
        if case_texts:
            case_embs = batch_embed_tokenmind(case_texts, batch_size=16)
            for i, emb in enumerate(case_embs):
                case_nodes_data[i]['embedding'] = emb

        law_texts = [str(n.get('description', ''))[:1500] for n in law_nodes_data]
        if law_texts:
            law_embs = batch_embed_tokenmind(law_texts, batch_size=16)
            for i, emb in enumerate(law_embs):
                law_nodes_data[i]['embedding'] = emb

        crime_texts = [str(n.get('description', ''))[:1500] for n in crime_nodes_data]
        if crime_texts:
            crime_embs = batch_embed_tokenmind(crime_texts, batch_size=16)
            for i, emb in enumerate(crime_embs):
                crime_nodes_data[i]['embedding'] = emb

        use_gpu_build = True
    else:
        # Attempt fast batch encoding on GPU if CUDA is available
        use_gpu_build = torch.cuda.is_available()
        embedder_model_name = _embedding_model or "unsloth/embeddinggemma-300m"

        if use_gpu_build:
            print(f"[GraphConstruct] Acceleration: Encoding nodes with '{embedder_model_name}' on GPU (cuda:0, fp16)...")
            try:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(
                    embedder_model_name,
                    device="cuda:0",
                    model_kwargs={"dtype": torch.float16}
                )
                embedder.max_seq_length = 512

                # 1. Batch encode Cases
                case_texts = [str(n.get('description', ''))[:1500] for n in case_nodes_data]
                if case_texts:
                    case_embs = embedder.encode(case_texts, batch_size=8, device="cuda:0", normalize_embeddings=True, show_progress_bar=True)
                    for i, emb in enumerate(case_embs):
                        case_nodes_data[i]['embedding'] = emb.tolist()

                # 2. Batch encode Laws
                law_texts = [str(n.get('description', ''))[:1500] for n in law_nodes_data]
                if law_texts:
                    law_embs = embedder.encode(law_texts, batch_size=8, device="cuda:0", normalize_embeddings=True, show_progress_bar=True)
                    for i, emb in enumerate(law_embs):
                        law_nodes_data[i]['embedding'] = emb.tolist()

                # 3. Batch encode Crimes
                crime_texts = [str(n.get('description', ''))[:1500] for n in crime_nodes_data]
                if crime_texts:
                    crime_embs = embedder.encode(crime_texts, batch_size=8, device="cuda:0", normalize_embeddings=True, show_progress_bar=True)
                    for i, emb in enumerate(crime_embs):
                        crime_nodes_data[i]['embedding'] = emb.tolist()

                # Free GPU memory completely for Reranker and inference
                del embedder
                torch.cuda.empty_cache()
                print("[GraphConstruct] Finished GPU batch encoding. Released GPU memory.")
            except Exception as e:
                print(f"[GraphConstruct] GPU batch encoding failed ({e}), falling back to sequential get_embedding...")
                use_gpu_build = False

    if not use_gpu_build:
        # Fallback to sequential get_embedding (e.g. via HTTP Ollama)
        for i, node in enumerate(tqdm(case_nodes_data, desc="Generating case embeddings")):
            node_embedding = get_embedding(node['description'])
            if node_embedding is not None:
                case_nodes_data[i]['embedding'] = node_embedding

        for i, node in enumerate(tqdm(law_nodes_data, desc="Generating law embeddings")):
            node_embedding = get_embedding(node['description'])
            if node_embedding is not None:
                law_nodes_data[i]['embedding'] = node_embedding

        for i, node in enumerate(tqdm(crime_nodes_data, desc="Generating crime embeddings")):
            node_embedding = get_embedding(node['description'])
            if node_embedding is not None:
                crime_nodes_data[i]['embedding'] = node_embedding

    # Store nodes and embeddings in graph DB
    store_nodes_with_embeddings(nodes_data)

    # Create hierarchical legal document clusters (grouped by file name, no KNN required)
    create_clusters(model)
