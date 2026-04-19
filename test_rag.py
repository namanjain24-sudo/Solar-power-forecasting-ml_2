from src.rag.retriever import load_docs, query_docs

load_docs()

query = "What to do during high solar generation?"

results = query_docs(query)

print("\n=== Retrieved Knowledge ===")
for r in results:
    print("-", r)