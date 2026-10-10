import json
from core.prompts import get_prompt


def summarize_question(model, case_text, criminals):
    cases = []
    for name in criminals:
        prompt2_formatted = get_prompt("SUMMARIZE_QUESTION_PROMPT").format(fact=case_text, name=name)
        response2 = model.generate_response(prompt2_formatted, max_length=1024)
        if response2 != "":
            cases.append({"name": name, "description": response2.strip()})
        else:
            cases.append({"name": name, "description": case_text.strip()})

    return cases
