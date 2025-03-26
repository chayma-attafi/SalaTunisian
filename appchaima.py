from fastapi import FastAPI
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from mousalslel import semantic_search, latin_to_arabic_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/mousalsel-recommend")
def recommend_series(query: str):
    """API endpoint to recommend a series based on user query."""
    arabic_query = latin_to_arabic_text(query)
    recommended_series = semantic_search(arabic_query)
    return {"title": recommended_series["title"], "description": recommended_series["description"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5177)
