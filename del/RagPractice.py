# import sys

# sys.stdout.reconfigure(encoding="utf-8")

# from pypdf import PdfReader

# reader=PdfReader("doc.pdf")

# text=""

# for page in reader.pages:
#     text+=page.extract_text()

# # print(text)


# chunk_size = 500
# overlap = 50

# chunks = []

# start = 0

# while start < len(text):
#     end = start + chunk_size
#     chunk = text[start:end]
#     chunks.append(chunk)

#     start += chunk_size - overlap

# print(len(chunks))

# from sentence_transformers import SentenceTransformer

# model= SentenceTransformer("all-MiniLM-L6-v2")

# embedding=model.encode(chunks)

# from pinecone import Pinecone

# pc=Pinecone(api_key="")

# index=pc.Index("gen-ai")


students = [
    {
        "name": "Zafar",
        "details": {
            "age": 20,
            "city": "Delhi"
        },
        "skills": ["Python", "NLP", "RAG"]
    },

    {
        "name": "Rahul",
        "details": {
            "age": 21,
            "city": "Mumbai"
        },
        "skills": ["Java", "SQL", "React"]
    }
]

for st in students:
    print(st["skills"])