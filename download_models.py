from transformers import pipeline
from sentence_transformers import SentenceTransformer

print("Downloading emotion model...")
pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("Downloading sentence embedder...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Done.")