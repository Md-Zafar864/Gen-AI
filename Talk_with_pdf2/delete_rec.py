from pinecone import Pinecone

# Connect to Pinecone
pc = Pinecone(api_key="")

# Connect to existing index
index = pc.Index("gen-ai")

# Delete all vectors from the index
index.delete(delete_all=True, namespace="")

print("All records deleted successfully.")