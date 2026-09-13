import json
import re
from core.prompt import get_prompt


def get_features(model, cases):
    fact = cases.get("description", "")
    name = cases.get("name", "ผู้สอบถาม")

    input_template = get_prompt("GET_FEATURES_INPUT_TEMPLATE")
    try:
        formatted_input = input_template.format(name=name, fact=fact)
    except KeyError:
        formatted_input = f"\nคำถาม/ข้อหารือ: {fact}"

    prompt_formatted = get_prompt("GET_FEATURES_PROMPT") + formatted_input
    response = model.generate_response(prompt_formatted)
    
    json_data = {}
    # Strip markdown code blocks if present
    cleaned = re.sub(r"^```(?:json)?", "", response.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    
    first = cleaned.find("{")
    last = cleaned.rfind("}") + 1
    if first != -1 and last != -1:
        json_str = cleaned[first:last]
        try:
            json_data = json.loads(json_str)
        except Exception:
            pass

    # Ensure required keys for graph traversal
    if not isinstance(json_data, dict):
        json_data = {}

    if "defendant_info" not in json_data:
        json_data["defendant_info"] = ["หน่วยงานของรัฐ / ผู้สอบถาม"]
    if "criminal_acts" not in json_data:
        # Represents inquiry actions/topics in procurement
        json_data["criminal_acts"] = [fact[:80]] if fact else ["การจัดซื้อจัดจ้าง"]
    if "victim_property_details" not in json_data:
        json_data["victim_property_details"] = ["พัสดุ / ขอบเขตงาน (TOR) / สัญญา"]
    if "intent_remorse" not in json_data:
        json_data["intent_remorse"] = []

    return json_data
