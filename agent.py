import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="lokal-kein-key-noetig",
)


def get_current_time() -> str:
    """Gibt das aktuelle lokale Datum und die Uhrzeit zurück."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_expression(expression: str) -> str:
    """Berechnet einen einfachen mathematischen Ausdruck sicher."""
    allowed_chars = set("0123456789+-*/(). ")
    if not expression or not set(expression).issubset(allowed_chars):
        return "Ungültiger Ausdruck."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Rechenfehler: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Gibt aktuelles Datum und aktuelle Uhrzeit zurück.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Berechnet einen mathematischen Ausdruck wie '25 * 4 + 10'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Der mathematische Ausdruck, z. B. '(120 / 3) + 15'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate_expression": calculate_expression,
}


def run_agent(prompt: str, model: str = "llama3.2") -> str:
    messages = [
        {
            "role": "system",
            "content": "Du bist ein lokaler KI-Agent. Nutze verfügbare Tools, wenn Zeit oder Rechnungen gefragt sind.",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            args = json.loads(raw_args)

            tool_fn = AVAILABLE_TOOLS.get(fn_name)
            tool_result = tool_fn(**args) if tool_fn else f"Tool '{fn_name}' nicht gefunden."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )

        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return final_response.choices[0].message.content

    return msg.content


if __name__ == "__main__":
    test_prompt = "Wie spät ist es und was ergibt (48 * 6) + 12?"
    try:
        print(run_agent(test_prompt))
    except Exception as e:
        print("Fehler:", e)