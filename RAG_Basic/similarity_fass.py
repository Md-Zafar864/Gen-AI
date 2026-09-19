from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Company documents
documents = [
    "Python is a popular programming language.",
    "Neural networks are used in deep learning.",
    "Transformers use self attention to process sequences.",
    "Football is a popular sport played around the world.",
    "Machine learning models learn patterns from data."
]


# Convert documents into embeddings
document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")


# Convert query into embedding
query = "How do neural networks work?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")


# Normalize vectors
faiss.normalize_L2(document_embeddings)
faiss.normalize_L2(query_embedding)


# Number of dimensions
dimension = document_embeddings.shape[1]

print("Embedding dimension:", dimension)


# Create FAISS index
index = faiss.IndexFlatIP(dimension)


# Add document vectors to the index
index.add(document_embeddings)


# Search
k = 3

scores, indices = index.search(
    query_embedding,
    k
)


# Display results
print("\n==============================")
print(f"TOP {k} RESULTS")
print("==============================")


for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(documents[index_id])