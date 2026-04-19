import chromadb
from sentence_transformers import SentenceTransformer

# init embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# init DB
client = chromadb.Client()
collection = client.get_or_create_collection(name="solar_knowledge")


def load_docs():
    with open("data/docs/grid_rules.txt", "r") as f:
        text = f.read()

    # simple chunking
    chunks = [c for c in text.split("\n") if len(c.strip()) > 20]

    for i, chunk in enumerate(chunks):
        if chunk.strip():
            embedding = model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"id_{i}"]
            )


def query_docs(query):
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results["documents"][0]