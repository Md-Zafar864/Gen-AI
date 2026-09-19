from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
text = [
    "Employees can reset their password from the account settings page.",
    "Employees receive 20 days of paid leave every year.",
    "The company provides health insurance to all full-time employees.",
    "The office cafeteria is open from 9 AM to 6 PM."
]

embed=model.encode(text)

print(embed.shape)