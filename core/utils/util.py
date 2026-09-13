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


def concat_feature_descriptions(description):
    res = ""
    res += "被告信息：" + ", ".join(description.get("defendant_info", [])) + "。"
    res += "犯罪行为：" + ", ".join(description.get("criminal_acts", [])) + "。"
    res += "犯罪对象特征：" + \
        ", ".join(description.get("victim_property_details", [])) + "。"
    res += "犯罪意图及悔罪表现：" + \
        ", ".join(description.get("intent_remorse", [])) + "。"
    return res


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
    features = cases["feature"]
    original_retrieved_res, retrieved_facts, retrieved_laws = query_similar_nodes(
        chatbot, concat_feature_descriptions(features), retrieve_config)

    if not retrieved_facts:
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
    features = cases["feature"]
    retrieved_facts = query_similar_nodes_naive(
        chatbot, concat_feature_descriptions(features), top_k=5)

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


def analyze_case(chatbot, case, law_to_crime, cases_db, retrieve_config):
    names = case.get("name")
    if not names:
        names = ["ผู้สอบถาม"]
    elif isinstance(names, str):
        names = [names]

    case_by_defendant = segment_case_text_withname(
        chatbot, case["fact"][:1024], names)
    if not case_by_defendant:
        case_by_defendant = [{"name": names[0], "description": case["fact"][:1024]}]

    for item in case_by_defendant:
        item["feature"] = get_features(chatbot, item)
        original_retrieved_res, retrieved_laws, retrieved_facts = retrieve(
            chatbot, item, law_to_crime, cases_db, retrieve_config)
        if not (retrieved_laws or retrieved_facts):
            item["judge_result"] = {"charge_name": [], "law_article": [], "term_of_imprisonment": {}}
            item["retrieved_laws"] = []
            item["retrieved_facts"] = []
            item["original_retrieved_res"] = original_retrieved_res
            item["used_laws"] = []
            item["used_facts"] = []
            continue
            
        law_used = []
        for law in retrieved_laws:
            used, _ = judge_law(
                chatbot, item['description'], law)
            if used:
                law_used.append(law)
        
        # If no laws confirmed by judge_law, fall back to top retrieved laws
        if not law_used and retrieved_laws:
            law_used = retrieved_laws[:3]
            
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
