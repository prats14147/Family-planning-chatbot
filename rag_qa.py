# rag_qa.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/faiss_index.bin")
df = pd.read_pickle("data/qa.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

def rag_answer(query):
    query_vec = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_vec, k=1)

    best_index = indices[0][0]
    best_distance = distances[0][0]

    # convert L2 distance to similarity score
    score = 1 / (1 + best_distance)

    answer = df.iloc[best_index]["Answers"]

    return answer, score
