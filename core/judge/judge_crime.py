import json
import re
from core.prompt import get_prompt


TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def format_law(law_used):
    res = ""
    for law in law_used:
        law["crimes"] = law.get("crimes", [])
        topics = [c.replace("\n", " ") for c in law["crimes"] if c]
        entry = law.get("entry", "")
        desc = law.get("description", "")
        if desc:
            desc = str(desc).translate(TH_TO_AR)
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


FALLBACK_NO_LAW_ANSWER = "ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหารือนี้ในฐานข้อมูลการจัดซื้อจัดจ้างภาครัฐ (เนื่องจากไม่อยู่ในขอบเขตของ พ.ร.บ. การจัดซื้อจัดจ้างฯ ระเบียบกระทรวงการคลัง หรือประกาศที่จัดเก็บไว้ในคลังข้อมูล)"


def judge_crime_all(chatbot, law_used, retrieved_facts, case_description):
    if not law_used and not retrieved_facts:
        return {
            "status": "NO_LAW_FOUND",
            "direct_answer": FALLBACK_NO_LAW_ANSWER,
            "decisive_quotes": [],
            "applicable_laws": [],
            "exceptions_or_conditions": "",
            "law_article": []
        }

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
        m_st = re.search(r'["\']status["\']\s*:\s*["\'](.*?)["\']\s*[,}]', cleaned, re.IGNORECASE)
        if m_st:
            parsed["status"] = m_st.group(1).strip().upper()

        m_dir = re.search(r'["\']direct_answer["\']\s*:\s*["\'](.*?)["\']\s*[,}]', cleaned, re.DOTALL)
        if m_dir:
            parsed["direct_answer"] = m_dir.group(1).strip()
            
        m_laws = re.search(r'["\']applicable_laws["\']\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        if m_laws:
            parsed["applicable_laws"] = [s.strip(" \"'\n\r") for s in m_laws.group(1).split(",") if s.strip(" \"'\n\r")]
            
        m_quote = re.search(r'["\'](?:decisive_quotes|decisive_quote)["\']\s*:\s*(?:\[(.*?)\]|["\'](.*?)["\'])', cleaned, re.DOTALL)
        if m_quote:
            raw_q = m_quote.group(1) or m_quote.group(2) or ""
            parsed["decisive_quotes"] = [{"law": "General", "quote": raw_q.strip()}] if raw_q.strip() else []

    # Check status and detect NO_LAW_FOUND state or degenerate backtick output
    raw_status = str(parsed.get("status", "")).strip().upper()
    direct_ans = str(parsed.get("direct_answer", "")).strip()

    # Detect degenerate backtick loops, empty responses, or corrupt LLM formatting
    is_degenerate = (
        not cleaned
        or direct_ans.startswith("```")
        or set(direct_ans.replace(" ", "").replace("\n", "")) <= {'`'}
        or (len(direct_ans) == 0 and not parsed.get("applicable_laws"))
        or (response.count("```") > 10 and not parsed.get("direct_answer"))
    )

    is_no_law = (
        raw_status == "NO_LAW_FOUND"
        or is_degenerate
        or direct_ans.startswith("ไม่พบข้อกฎหมาย")
        or direct_ans.startswith("ไม่อยู่ในขอบเขต")
    )

    if is_no_law:
        parsed["status"] = "NO_LAW_FOUND"
        parsed["direct_answer"] = FALLBACK_NO_LAW_ANSWER
        parsed["decisive_quotes"] = []
        parsed["applicable_laws"] = []
        parsed["law_article"] = []
        parsed["exceptions_or_conditions"] = ""
        parsed.pop("legal_reasoning", None)
        return parsed

    # Normalize state for compliant/violation cases
    if raw_status in ["COMPLIANT", "VIOLATION"]:
        parsed["status"] = raw_status
    elif "VIOLATION" in raw_status or "ฝ่าฝืน" in raw_status or "ผิด" in raw_status:
        parsed["status"] = "VIOLATION"
    else:
        parsed["status"] = "COMPLIANT"

    # Guarantee QA fields
    if "direct_answer" not in parsed:
        parsed["direct_answer"] = ""
    
    # Remove legal_reasoning from output
    parsed.pop("legal_reasoning", None)

    # Normalize decisive_quotes (supporting list of dicts, dict, or string)
    quotes = parsed.get("decisive_quotes") or parsed.get("decisive_quote") or []
    if isinstance(quotes, str):
        quotes = [{"law": "General", "quote": quotes.strip()}] if quotes.strip() else []
    elif isinstance(quotes, dict):
        quotes = [{"law": k, "quote": str(v).strip()} for k, v in quotes.items()]
    elif isinstance(quotes, list):
        norm_quotes = []
        for q in quotes:
            if isinstance(q, dict):
                norm_quotes.append(q)
            elif isinstance(q, str) and q.strip():
                norm_quotes.append({"law": "General", "quote": q.strip()})
        quotes = norm_quotes
    parsed["decisive_quotes"] = quotes

    if "applicable_laws" not in parsed:
        if "law_article" in parsed:
            raw_laws = parsed["law_article"] if isinstance(parsed["law_article"], list) else [str(parsed["law_article"])]
        else:
            raw_laws = [law.get("entry", "") for law in law_used]
    else:
        raw_laws = parsed["applicable_laws"] if isinstance(parsed["applicable_laws"], list) else [str(parsed["applicable_laws"])]
    parsed["applicable_laws"] = list(dict.fromkeys([str(x).strip() for x in raw_laws if str(x).strip()]))

    if "exceptions_or_conditions" not in parsed:
        parsed["exceptions_or_conditions"] = ""

    # Map backwards for legacy consumers
    parsed["law_article"] = parsed["applicable_laws"]

    return parsed
