
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import time

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

index_name = "genai-learning-01"

# Create index if it does not already exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    time.sleep(10)

print("Connected to Pinecone!")


# Connect to the index
index = pc.Index(index_name)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Documents
documents = [
    "Python is a popular programming language.",
    "Neural networks are used in deep learning.",
    "Transformers use self attention to process sequences.",
    "Football is a popular sport played around the world.",
    "Machine learning models learn patterns from data."
]


# Create document embeddings
document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
)


# Prepare records
records = []

for i, embedding in enumerate(document_embeddings):
    records.append({
        "id": f"doc-{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": documents[i]
        }
    })


# Upload vectors to Pinecone
index.upsert(vectors=records)

print("Vectors uploaded successfully!")


# User query
query = "How do neural networks work?"


# Create query embedding
query_embedding = model.encode(query).tolist()


# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)


# Display results
print("\n==============================")
print("TOP 3 RESULTS")
print("==============================")

for rank, match in enumerate(results["matches"], start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
