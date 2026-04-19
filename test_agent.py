from src.agent.solar_agent import run_agent
from src.rag.retriever import load_docs

# load knowledge base
load_docs()

# dummy predictions
predictions = [20, 40, 80, 90, 30, 25]

result = run_agent(predictions)

print(result)
