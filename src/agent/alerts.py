from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_alert(summary, risk):
    prompt = f"""
You are a grid monitoring AI.

Given:
Summary: {summary}
Risk: {risk}

Generate a SHORT alert message (1-2 lines max).
Be practical and actionable.
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return res.choices[0].message.content.strip()
