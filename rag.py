import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load CSV
df = pd.read_csv("data/qa.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
questions = df["Questions"].tolist()
embeddings = model.encode(questions, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/faiss_index.bin")

# Save dataframe for use later
df.to_pickle("data/qa.pkl")

print("FAISS index created successfully!")
