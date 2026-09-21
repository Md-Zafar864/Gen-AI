import sys

sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------
# Import required libraries
# --------------------------------------------------

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------

reader = PdfReader("Doc.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# print(text)
# print("extracted text length", len(text))


# --------------------------------------------------
# 2. Split text into chunks
# --------------------------------------------------

chunk_size = 500
overlap = 50

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)

    start += chunk_size - overlap


# print("Number of chunks:", len(chunks))
# print("\nFirst chunk:")
# print(chunks[0])


# --------------------------------------------------
# 3. Convert chunks into embeddings
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embedd = model.encode(chunks)

print(embedd.shape)


# --------------------------------------------------
# 4. Connect to Pinecone
# --------------------------------------------------

pc = Pinecone(api_key="")

index = pc.Index("gen-ai")


# --------------------------------------------------
# 5. Create records for Pinecone
# --------------------------------------------------

record = []

for i, embed in enumerate(embedd):

    record.append({
        "id": f"chunk_{i}",
        "values": embed.tolist(),
        "metadata": {
            "text": chunks[i],
            "source": "Doc.pdf"
        }
    })


# --------------------------------------------------
# 6. Upload records to Pinecone
# --------------------------------------------------

index.upsert(vectors=record)


# --------------------------------------------------
# 7. Create embedding for the question
# --------------------------------------------------

question = "What is the maximum ambulance coverage?"

query_embed = model.encode(question)


# --------------------------------------------------
# 8. Search Pinecone for relevant chunks
# --------------------------------------------------

results = index.query(
    vector=query_embed.tolist(),
    top_k=3,
    include_metadata=True
)


# --------------------------------------------------
# 9. Display retrieved results
# --------------------------------------------------

# for match in results["matches"]:
#     print("Score:", match["score"])
#     print("Text:", match["metadata"]["text"])
#     print()


# --------------------------------------------------
# 10. Build context from retrieved chunks
# --------------------------------------------------

context = ""

for match in results["matches"]:
    context += match["metadata"]["text"] + "\n\n"

print("conext is ----- \n", context)


# --------------------------------------------------
# 11. Connect to Gemini
# --------------------------------------------------

from google import genai

client = genai.Client(api_key="")


# --------------------------------------------------
# 12. Create prompt using retrieved context
# --------------------------------------------------

prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}
"""


# --------------------------------------------------
# 13. Generate final answer using Gemini
# --------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)

print(response.text)