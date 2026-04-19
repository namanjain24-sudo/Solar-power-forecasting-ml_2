from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_agent(question, summary, knowledge):
    prompt = f"""
You are a solar grid optimization expert.

Context:
Summary: {summary}
Knowledge: {knowledge}

IMPORTANT RULES:
1. If the User Question is unrelated to solar power forecasting, grid management, or the provided Context (e.g., general programming questions like "how to print hello world", or casual chat, or recipes), you MUST refuse to answer. 
2. Politely state that you are a specialized assistant and can only answer questions related to solar power forecasting and grid optimization.
3. Do not provide code unless it is specifically about solar grid context.

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
