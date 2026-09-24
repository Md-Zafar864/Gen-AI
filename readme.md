reader.pages
Page.extract_text()
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from google import genai


reader = PdfReader("Doc.pdf")

pagess=[]

for page_number, page in enumerate(reader.pages, start=1):
    text=page.extract_text()

    pagess.append({
        "text" : text,
        "page_number" : page_number
    })

# print(pagess[3])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks=[]

for page in pagess:
    page_chunks=splitter.split_text(page["text"])

    for chunk in page_chunks:
        chunks.append({
            "text":chunk,
            "page_number":page["page_number"]
        })


print("Number of chunks:", len(chunks))
print("First chunk:", chunks[0])

model=SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]


embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

pc=Pinecone(api_key="")

index=pc.Index("gen-ai")

record=[]

for i, embedding in enumerate(embeddings):
    record.append({
        "id":f"chunks_{i}",
        "value": embedding.tolist(),
        "metadata":{
            "text":chunks[i]["text"],
            "sources":"Doc.pdf",
            "page_number":chunk[i][page_number]
        }
    })

    index.upsert(vector=record)

question = "What is the maximum ambulance coverage?"
query_embed = model.encode(question)

result=index.query(
    vector=query_embed.tolist(),
    top_k=3,
    include_metadata=True
)

context=""

for match in result["matches"]:
    context+=match["metadata"]["text"] + "\n\n"

print(context)

client=genai.Client(api_key="")

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