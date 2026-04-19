from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_reason(summary, knowledge):
    prompt = f"""
You are an expert in solar energy grid management.

Given:
1. Forecast summary:
{summary}

2. Retrieved knowledge:
{knowledge}

Tasks:
- Identify risk level (Low/Medium/High)
- Explain reasoning
- Give optimization strategies

Return strictly in this format:

Risk: <value>

Reasoning:
<explanation>

Recommendations:
- point 1
- point 2
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content