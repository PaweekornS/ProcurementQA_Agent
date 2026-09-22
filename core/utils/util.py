from core.preprocess.get_features import get_features
from core.preprocess.case_seg import segment_case_text_withname

from core.graph_construct.feature_graph import query_similar_nodes, query_similar_laws, query_similar_laws_naive, query_similar_nodes_naive, update_insights_in_graph

from core.judge.judge_law import judge_law
from core.judge.judge_crime import judge_crime, judge_crime_all

import json
from core.prompt import get_prompt


def filter_facts(retrieved_laws, retrieved_facts):
    """
    过滤retrieved_facts，只保留law中包含至少一个retrieved_laws中条目的fact

    参数:
    retrieved_laws: 法律条文列表，每个元素包含id字段
    retrieved_facts: 事实列表，每个元素包含law字段（法律id列表）

    返回:
    过滤后的facts列表
    """
    # 提取所有retrieved_laws的id集合
    law_ids = {str(law['id']) for law in retrieved_laws}

    # 过滤facts，只保留law字段中至少有一个id在law_ids中的fact
    filtered_facts = [
        fact for fact in retrieved_facts
        if any(law_id in law_ids for law_id in fact.get('law', []))
    ]

    return filtered_facts


def concat_feature_descriptions(description, raw_text=""):
    """
    Concat feature descriptions for retrieval query.
    Extracts key procurement points in Thai and combines with original query text for optimal BM25/Dense matching.
    """
    if isinstance(description, str):
        return description.strip()

    parts = []
    # Primary procurement issues
    criminal_acts = description.get("criminal_acts", [])
    if criminal_acts:
        parts.append("ประเด็น: " + ", ".join(str(x) for x in criminal_acts))

    # Procurement object, items, TOR, or budget
    victim_property = description.get("victim_property_details", [])
    if victim_property:
        parts.append("พัสดุ/ขอบเขต: " + ", ".join(str(x) for x in victim_property))

    # Intent, conditions, or exceptions
    intent = description.get("intent_remorse", [])
    if intent:
        parts.append("เงื่อนไข: " + ", ".join(str(x) for x in intent))

    # Defendant / entity info
    defendant = description.get("defendant_info", [])
    if defendant:
        parts.append("หน่วยงาน/ผู้เกี่ยวข้อง: " + ", ".join(str(x) for x in defendant))

    feature_summary = " ".join(parts).strip()
    # If original inquiry text is available, prioritize it directly as query
    # to avoid BM25 term frequency dilution from repetitive metadata labels
    if raw_text and raw_text.strip():
        return raw_text.strip()
    return feature_summary if feature_summary else str(raw_text)


def retrieve_law(chatbot, case):
    fact = case["description"][:1024]
    name = case["name"]
    response = chatbot.generate_response(
        get_prompt("RETRIEVE_LAW_PROMPT").format(name=name, fact=fact), max_length=256)
    try:
        first = response.find('[')
        last = response.rfind(']') + 1
        crimes = eval(response[first:last])
    except (ValueError, SyntaxError):
        return []
    laws = query_similar_laws(crimes, top_k=1)
    # print(f"Fact: {fact}\nPredicted Crimes: {crimes}\nRetrieved Laws: {laws}\n")
    return laws


def retrieve(chatbot, cases, law_to_crime, cases_db, retrieve_config):
    features = cases.get("feature", {})
    query_text = concat_feature_descriptions(features, raw_text=cases.get("description", ""))
    original_retrieved_res, retrieved_facts, retrieved_laws = query_similar_nodes(
        chatbot, query_text, retrieve_config)

    # In Thai Procurement domain, knowledge base is predominantly statutory Laws (2,486 nodes)
    # alongside FAQ cases (29 nodes). Proceed if either laws or facts are retrieved.
    if not retrieved_facts and not retrieved_laws:
        return {}, [], []

    augmented_laws = []
    if retrieve_config["augment_retrieve"]:
        augmented_laws = retrieve_law(chatbot, cases)
        original_retrieved_res["augmented"] = augmented_laws
    else:
        augmented_laws = []
    retrieved_laws = retrieved_laws + augmented_laws
    for item in retrieved_facts:
        for case in cases_db:
            if str(case.get("id", "")) == str(item.get("caseId", "")):
                item["crime"] = case.get("crime", [])
                item["law"] = case.get("law", [])
                break
    final_retrieved_laws = []
    seen_law_ids = set()
    for law in retrieved_laws:
        if law["id"] in seen_law_ids:
            continue
        seen_law_ids.add(law["id"])
        if isinstance(law.get("judge_dep"), str):
            try:
                law["judge_dep"] = eval(law["judge_dep"])
            except Exception:
                law["judge_dep"] = []
        elif not isinstance(law.get("judge_dep"), list):
            law["judge_dep"] = []
            
        if isinstance(law.get("related_laws"), str):
            try:
                law["related_laws"] = eval(law["related_laws"])
            except Exception:
                law["related_laws"] = []
        elif not isinstance(law.get("related_laws"), list):
            law["related_laws"] = []
            
        final_retrieved_laws.append(law)

    return original_retrieved_res, final_retrieved_laws, retrieved_facts


def naive_retrieve(chatbot, cases, law_to_crime, cases_db):
    features = cases.get("feature", {})
    query_text = concat_feature_descriptions(features, raw_text=cases.get("description", ""))
    retrieved_facts = query_similar_nodes_naive(
        chatbot, query_text, top_k=5)

    if not retrieved_facts:
        return None, None

    retrieved_laws = query_similar_laws_naive(
        concat_feature_descriptions(features), top_k=5)
    retrieved_laws = [str(law['entry']) for law in retrieved_laws]
    for item in retrieved_facts:
        for case in cases_db:
            if str(case.get("id", "")) == str(item.get("caseId", "")):
                item["crime"] = case.get("crime", [])
                item["law"] = case.get("law", [])
                retrieved_laws.extend(case.get("law", []))
                break
    retrieved_laws = list(set(retrieved_laws))
    final_retrieved_laws = []
    for x in retrieved_laws:
        try:
            for item in law_to_crime:
                item_id = str(item.get("id", ""))
                x_str = str(x)
                if item_id == x_str or (x_str and (x_str in item_id or item_id in x_str)):
                    for entry in item.get("items", []):
                        final_retrieved_laws.append(
                            {"id": item["id"], "text": entry["text"], "crime": entry["crime"], "judge_dep": entry["judge_dep"], "related_laws": entry["related_laws"]})
                    break
        except Exception:
            continue

    return final_retrieved_laws, retrieved_facts


def locate_law(law, laws):
    for item in laws:
        if law["id"] == item["id"]:
            return item
    return law["text"]


def analyze_case(chatbot, case, law_to_crime, cases_db, retrieve_config, crag_config=None):
    names = case.get("name")
    if not names:
        names = ["ผู้สอบถาม"]
    elif isinstance(names, str):
        names = [names]

    raw_fact = case.get("fact", "")
    # Optimization: Use concise procurement inquiry directly to save ~30-50s per query
    if len(raw_fact) < 800:
        case_by_defendant = [{"name": names[0], "description": raw_fact}]
    else:
        case_by_defendant = segment_case_text_withname(
            chatbot, raw_fact[:1024], names)
        if not case_by_defendant:
            case_by_defendant = [{"name": names[0], "description": raw_fact[:1024]}]

    use_crag = True
    max_retry = 1
    if crag_config is not None:
        use_crag = bool(crag_config.get("enabled", True))
        max_retry = int(crag_config.get("max_retry", 1))

    if use_crag:
        from core.crag.pipeline import CRAGPipeline
        crag_pipe = CRAGPipeline(chatbot, retrieve_config=retrieve_config, max_retry=max_retry)
        for item in case_by_defendant:
            crag_pipe.process_case_item(item, law_to_crime, cases_db)
        return case_by_defendant

    for item in case_by_defendant:
        item["feature"] = get_features(chatbot, item)
        original_retrieved_res, retrieved_laws, retrieved_facts = retrieve(
            chatbot, item, law_to_crime, cases_db, retrieve_config)
        if not (retrieved_laws or retrieved_facts):
            fallback_msg = "ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหารือนี้ในฐานข้อมูลการจัดซื้อจัดจ้างภาครัฐ (เนื่องจากไม่อยู่ในขอบเขตของ พ.ร.บ. การจัดซื้อจัดจ้างฯ ระเบียบกระทรวงการคลัง หรือประกาศที่จัดเก็บไว้ในคลังข้อมูล)"
            item["judge_result"] = {
                "status": "NO_LAW_FOUND",
                "direct_answer": fallback_msg,
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "law_article": [],
            }
            item["retrieved_laws"] = []
            item["retrieved_facts"] = []
            item["original_retrieved_res"] = original_retrieved_res
            item["used_laws"] = []
            item["used_facts"] = []
            continue
            
        # Use top reranked laws (up to 8) plus graph-traversed neighbor laws
        max_laws = retrieve_config.get("direct_retrieve_top_k", 8)
        law_used = retrieved_laws[:max_laws]
            
        fact_used = filter_facts(law_used, retrieved_facts) if retrieved_facts else []
        judge_result = judge_crime_all(
            chatbot, law_used, fact_used, item['description'])
        item["judge_result"] = judge_result
        item["retrieved_laws"] = retrieved_laws
        item["retrieved_facts"] = retrieved_facts
        item["original_retrieved_res"] = original_retrieved_res
        item["used_laws"] = law_used
        item["used_facts"] = fact_used

    return case_by_defendant
