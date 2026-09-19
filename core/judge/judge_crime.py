import json
import re
from core.prompt import get_prompt


def format_law(law_used):
    res = ""
    for law in law_used:
        law["crimes"] = law.get("crimes", [])
        topics = [c.replace("\n", " ") for c in law["crimes"] if c]
        entry = law.get("entry", "")
        desc = law.get("description", "")
        res += f"ข้อกฎหมาย/ระเบียบ: {entry}\nหมวดหมู่/ประเด็น: {', '.join(topics)}\nเนื้อหาข้อกฎหมาย:\n{desc}\n---\n"

    return res


def format_fact(facts):
    res = ""
    for fact in facts:
        topics = fact.get("crime", [])
        desc = fact.get("description", "")
        res += f"ประเด็นที่เกี่ยวข้อง: {', '.join(topics)}\nรายละเอียด/แนวทาง: {desc}\n"
    return res


def judge_crime(chatbot, law_used, retrieved_facts, case_description):
    response = chatbot.generate_response(
        get_prompt("JUDGE_CRIME_PROMPT").format(
            law=format_law(law_used),
            case=case_description
        ),
        max_length=4096
    )
    try:
        first = response.rfind('[')
        last = response.rfind(']') + 1
        response = response.replace('，', ',')
        response = list(set(eval(response[first:last])))
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = []
    response = [str(x).strip() for x in response if str(x).strip()]
    return response


def judge_crime_all(chatbot, law_used, retrieved_facts, case_description):
    prompt_str = get_prompt("JUDGE_CRIME_ALL_PROMPT")
    formatted_law = format_law(law_used)
    formatted_facts = format_fact(retrieved_facts) if retrieved_facts else ""

    if "{law}" in prompt_str and "{case}" in prompt_str:
        if "{facts}" in prompt_str:
            full_prompt = prompt_str.format(
                law=formatted_law,
                facts=formatted_facts,
                case=case_description
            )
        else:
            full_prompt = prompt_str.format(
                law=formatted_law,
                case=case_description
            )
    else:
        input_template = get_prompt("JUDGE_CRIME_ALL_INPUT_TEMPLATE")
        full_prompt = prompt_str + input_template.format(
            law=formatted_law,
            case=case_description
        )

    response = chatbot.generate_response(full_prompt, max_length=4096)
    
    # Robust LLM JSON parsing
    parsed = {}
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    
    if first != -1 and last != -1 and last > first:
        json_str = cleaned[first:last + 1]
        try:
            parsed = json.loads(json_str)
        except Exception:
            # Try repairing common LLM trailing commas: e.g. [1, 2,] or {"a": 1,}
            fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                parsed = json.loads(fixed)
            except Exception:
                pass
                
    if not parsed or not isinstance(parsed, dict):
        # Fallback: regex field extraction if json is broken or truncated
        parsed = {}
        m_dir = re.search(r'["\']direct_answer["\']\s*:\s*["\'](.*?)["\']\s*[,}]', cleaned, re.DOTALL)
        if m_dir:
            parsed["direct_answer"] = m_dir.group(1).strip()
            
        m_ans = re.search(r'["\']answer["\']\s*:\s*["\'](.*?)["\']\s*[,}]', cleaned, re.DOTALL)
        if m_ans:
            parsed["answer"] = m_ans.group(1).strip()
            
        m_laws = re.search(r'["\']applicable_laws["\']\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        if m_laws:
            parsed["applicable_laws"] = [s.strip(" \"'\n\r") for s in m_laws.group(1).split(",") if s.strip(" \"'\n\r")]
            
        m_cat = re.search(r'["\']category["\']\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        if m_cat:
            parsed["category"] = [s.strip(" \"'\n\r") for s in m_cat.group(1).split(",") if s.strip(" \"'\n\r")]
            
        if not parsed.get("answer"):
            parsed["answer"] = cleaned

    # Guarantee QA fields
    if "direct_answer" not in parsed:
        parsed["direct_answer"] = ""
    if "answer" not in parsed or not parsed["answer"]:
        parsed["answer"] = parsed.get("legal_opinion") or parsed.get("direct_answer") or response.strip()
    if not parsed.get("direct_answer") and parsed.get("answer"):
        first_line = parsed["answer"].strip().split("\n")[0]
        parsed["direct_answer"] = first_line[:200]

    if "applicable_laws" not in parsed:
        if "law_article" in parsed:
            raw_laws = parsed["law_article"] if isinstance(parsed["law_article"], list) else [str(parsed["law_article"])]
        else:
            raw_laws = [law.get("entry", "") for law in law_used]
    else:
        raw_laws = parsed["applicable_laws"] if isinstance(parsed["applicable_laws"], list) else [str(parsed["applicable_laws"])]
    parsed["applicable_laws"] = list(dict.fromkeys([str(x).strip() for x in raw_laws if str(x).strip()]))

    if "category" not in parsed:
        if "charge_name" in parsed:
            raw_cats = parsed["charge_name"] if isinstance(parsed["charge_name"], list) else [str(parsed["charge_name"])]
        else:
            raw_cats = []
    else:
        raw_cats = parsed["category"] if isinstance(parsed["category"], list) else [str(parsed["category"])]
    parsed["category"] = list(dict.fromkeys([str(x).strip() for x in raw_cats if str(x).strip()]))

    if "exceptions_or_conditions" not in parsed:
        parsed["exceptions_or_conditions"] = ""

    # Map backwards for legacy consumers
    parsed["law_article"] = parsed["applicable_laws"]
    parsed["charge_name"] = parsed["category"]

    return parsed
