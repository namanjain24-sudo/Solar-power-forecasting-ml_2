from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_agent(question, summary, knowledge):
    prompt = f"""
You are a solar grid optimization expert.

Context:
Summary: {summary}
Knowledge: {knowledge}

User Question:
{question}

Answer clearly and practically.
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return res.choices[0].message.content.strip()
