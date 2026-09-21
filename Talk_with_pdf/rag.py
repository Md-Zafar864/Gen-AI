import sys
sys.stdout.reconfigure(encoding="utf-8")

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

reader = PdfReader("Doc.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# print(text)
# print("extracted text length", len(text))



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

model=SentenceTransformer("all-MiniLM-L6-v2")

embedd=model.encode(chunks)
print(embedd.shape)


pc=Pinecone(api_key="")

index=pc.Index("gen-ai")

record=[]

for i, embed in enumerate(embedd):
    record.append({
        "id": f"chunk_{i}",
        "values": embed.tolist(),
        "metadata" : {
            "text":chunks[i],
            "source":"Doc.pdf"
        }
    })

index.upsert(vectors=record)

question="What is the maximum ambulance coverage?"
query_embed=model.encode(question)

results = index.query(
    vector=query_embed.tolist(),
    top_k=3,
    include_metadata=True
)

# for match in results["matches"]:
#     print("Score:", match["score"])
#     print("Text:", match["metadata"]["text"])
#     print()

context=""
for match in results["matches"]:
    context += match["metadata"] ["text"] + "\n\n"

print("conext is ----- \n", context)

from google import genai
client = genai.Client(api_key="")
prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}
"""
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)

print(response.text)