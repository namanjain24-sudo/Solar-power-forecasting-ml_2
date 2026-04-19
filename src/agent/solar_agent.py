from src.utils.forecast_summary import generate_summary
from src.rag.retriever import query_docs
from groq import Groq
import os

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_reason(summary, knowledge):
    prompt = f"""
You are an expert in solar energy grid management.

IMPORTANT:
- Do NOT assume values are extreme unless explicitly stated
- Base reasoning only on patterns (high, low, variability)

Forecast Summary:
{summary}

Retrieved Knowledge:
{knowledge}

Tasks:
1. Determine risk level (Low / Medium / High)
2. Explain reasoning clearly
3. Provide actionable optimization recommendations

Return STRICT JSON:

{{
  "risk": "Low/Medium/High",
  "reasoning": "explanation",
  "recommendations": ["point1", "point2"]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def run_agent(predictions):
    # Step 1: Summary
    summary = generate_summary(predictions)

    # Step 2: Retrieve knowledge
    knowledge_raw = query_docs(summary)
    knowledge = knowledge_raw if isinstance(knowledge_raw, list) else [knowledge_raw]

    # Step 3: LLM reasoning
    llm_output = llm_reason(summary, knowledge)

    # Step 4: Safe parsing (important)
    import json

    try:
        parsed = json.loads(llm_output)
    except:
        parsed = {
            "risk": "Unknown",
            "reasoning": llm_output,
            "recommendations": ["LLM output parsing failed"]
        }

    return {
        "summary": summary,
        "risk": parsed.get("risk", "Unknown"),
        "reasoning": parsed.get("reasoning", ""),
        "knowledge": knowledge,
        "recommendations": parsed.get("recommendations", [])
    }