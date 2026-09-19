from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Employees can reset their password from the account settings page.",
    "Employees receive 20 days of paid leave every year.",
    "The company provides health insurance to all full-time employees.",
    "The office cafeteria is open from 9 AM to 6 PM.",
    "Employees can have half day 4 time in a month",
    "Employees have to serve 2 month notice period before leaving"
]

embeddings = model.encode(documents)

query = "I forgot my password. How can I change it?"

query_embedding = model.encode(query)

similarities = cosine_similarity(
    [query_embedding],
    embeddings
)

print("Similarity scores:", similarities)

print(similarities.shape)
k=3
idx=np.argsort(similarities)[::-1][:k]
print(idx)
