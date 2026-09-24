import sys
sys.stdout.reconfigure(encoding="utf-8")

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from google import genai


# =========================================================
# 1. Gemini client
# =========================================================

client = genai.Client(api_key="")


# =========================================================
# 2. Query rewriting function
# =========================================================

def rewrite_query(question, conversation):

    history = ""

    for item in conversation:
        history += f"User: {item['question']}\n"
        history += f"Assistant: {item['answer']}\n"

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history only to resolve references such as:
"it", "that plan", "the second one", "this", etc.

Do not answer the question.
Return only the rewritten search query.

Conversation history:
{history}

Latest question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text.strip()


# =========================================================
# 3. Read PDF
# =========================================================

reader = PdfReader("Doc.pdf")

pagess = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    pagess.append({
        "text": text,
        "page_number": page_number
    })


# =========================================================
# 4. Split pages into chunks
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []

for page in pagess:

    page_chunks = splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append({
            "text": chunk,
            "page_number": page["page_number"]
        })


print("Number of chunks:", len(chunks))
print("First chunk:", chunks[0])


# =========================================================
# 5. Load embedding model
# =========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# 6. Create embeddings for all chunks
# =========================================================

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)


# =========================================================
# 7. Connect to Pinecone
# =========================================================

pc = Pinecone(api_key="")

index = pc.Index("gen-ai")


# =========================================================
# 8. Prepare records
# =========================================================

records = []

for i, embedding in enumerate(embeddings):

    records.append({
        "id": f"chunks_{i}",

        "values": embedding.tolist(),

        "metadata": {
            "text": chunks[i]["text"],
            "source": "Doc.pdf",
            "page_number": chunks[i]["page_number"]
        }
    })


# =========================================================
# 9. Upload chunks to Pinecone
# =========================================================

index.upsert(vectors=records)


# =========================================================
# 10. Load CrossEncoder reranker
# =========================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =========================================================
# 11. Conversation memory
# =========================================================

conversation = []


# =========================================================
# 12. User question
# =========================================================

question = "What is the maximum ambulance coverage?"


# =========================================================
# 13. Rewrite query
# =========================================================

rewritten_query = rewrite_query(
    question,
    conversation
)

print("\nOriginal question:")
print(question)

print("\nRewritten query:")
print(rewritten_query)


# =========================================================
# 14. Convert rewritten query into embedding
# =========================================================

query_embed = model.encode(rewritten_query)


# =========================================================
# 15. Search Pinecone
# =========================================================

result = index.query(
    vector=query_embed.tolist(),
    top_k=5,
    include_metadata=True
)


# =========================================================
# 16. Filter weak matches
# =========================================================

MIN_SCORE = 0.5

relevant_chunks = [
    match
    for match in result["matches"]
    if match["score"] >= MIN_SCORE
]


print("\nRelevant chunks:", len(relevant_chunks))


# =========================================================
# 17. Create pairs for CrossEncoder
# =========================================================

pairs = []

for match in relevant_chunks:

    pairs.append([
        rewritten_query,
        match["metadata"]["text"]
    ])


# =========================================================
# 18. Calculate reranker scores
# =========================================================

if pairs:

    scores = reranker.predict(pairs)

else:

    scores = []


# =========================================================
# 19. Combine chunks with reranker scores
# =========================================================

reranked = []

for match, score in zip(relevant_chunks, scores):

    reranked.append(
        (match, score)
    )


# =========================================================
# 20. Sort by reranker score
# =========================================================

reranked.sort(
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 21. Take top 3 chunks
# =========================================================

top_chunks = reranked[:3]


# =========================================================
# 22. Build context
# =========================================================

context = ""


for i, (match, score) in enumerate(top_chunks, start=1):

    context += f"""
[Source {i}]
Document: {match["metadata"]["source"]}
Page: {match["metadata"]["page"]}
Content:
{match["metadata"]["text"]}

"""


# =========================================================
# 23. Generate final answer
# =========================================================

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


answer = response.text.strip()


print("\nAnswer:")
print(answer)


# =========================================================
# 24. Update conversation
# =========================================================

conversation.append({
    "question": question,
    "answer": answer
})


print("\nConversation:")
print(conversation)