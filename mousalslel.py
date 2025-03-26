import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

# Load SentenceTransformer model for embedding
st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Latinized Tunisian to Arabic decoding dictionary
# latin_to_arabic = {
#     "7": "ح", "3": "ع", "5": "خ", "6": "ط", "ta": "ق", "8": "ق", "2": "أ", "4": "د", "kh": "خ", 
#     "t": "ت", "b": "ب", "g": "ج", "s": "س", "z": "ز", "f": "ف", "k": "ك", "l": "ل", "m": "م", 
#     "n": "ن", "h": "ه", "w": "و", "y": "ي", "o": "و"
# }

# def decode_tunisian_latin(text):
#     """Decode Latinized Tunisian (with numbers and letters) into Arabic."""
#     for key in sorted(latin_to_arabic.keys(), key=len, reverse=True):  # Replace multi-char mappings first
#         text = text.replace(key, latin_to_arabic[key])
#     return text
# Dictionnaire de translittération latine vers arabe
latin_to_arabic = {
    "a": "ا", "b": "ب", "te": "ت", "th": "ث", "j": "ج", "7": "ح", "kh": "خ",
    "d": "د", "dh": "ذ", "r": "ر", "z": "ز", "s": "س", "ch": "ش",
    "sa": "ص", "D": "ض", "t": "ط", "Z": "ظ", "3": "ع", "gh": "غ",
    "f": "ف", "9": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "h": "ه", "w": "و", "y": "ي", "ou": "و", "i": "ي","5":"خ"
}
 
# Fonction de conversion
def latin_to_arabic_text(text):
    arabic_word = ""
    i = 0
    while i < len(text):
        # Vérifier les lettres doubles comme "ch", "kh", "th", "sa", "gh", "dh"
        if i < len(text) - 1 and text[i:i+2].lower() in latin_to_arabic:
            arabic_char = latin_to_arabic[text[i:i+2].lower()]
            arabic_word += arabic_char if text[i:i+2].islower() else arabic_char.upper()
            i += 2  # Sauter deux lettres
        elif text[i].lower() in latin_to_arabic:
            arabic_char = latin_to_arabic[text[i].lower()]
            arabic_word += arabic_char if text[i].islower() else arabic_char.upper()
            i += 1
        else:
            arabic_word += text[i]  # Conserver les caractères inconnus (espaces, ponctuation, etc.)
            i += 1
    return arabic_word
 


def generate_llm_response(user_input):
    api_key = "GROQ_API_KEY"  # Replace with your actual API key
    url = "https://api.groq.com/openai/v1/chat/completions"  # Groq API endpoint

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",  # Model to use
        "messages": [{"role": "user", "content": user_input}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print("Error making API request:", e)
        return "Error in API request"

# Series Data
# Load series data from the JSON file
with open('series_data.json', 'r') as f:
    series_data = json.load(f)


# Precompute embeddings for series descriptions
series_embeddings = [st_model.encode(series["description"]) for series in series_data]

def expand_query_with_llm(query):
    """Use LLaMA 3 API to expand the query for better search performance."""
    prompt = f"Rewrite this user query to be clearer and more descriptive for a TV series recommendation system:\n\nQuery: {query}\n\nImproved Query:"
    expanded_query = generate_llm_response(prompt)
    return expanded_query.strip()

def semantic_search(query):
    """Perform semantic search using SentenceTransformer with LLaMA 3 query enhancement."""
    # Decode the query from Latinized Tunisian to Arabic
    #decoded_query = decode_tunisian_latin(query)


    # Expand the query using LLaMA 3 API
    improved_query = expand_query_with_llm(query)
    
    
    # Encode the improved query
    query_embedding = st_model.encode(improved_query)
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], series_embeddings)
    
    # Get the most similar series
    most_similar_idx = np.argmax(similarities)
    
    return series_data[most_similar_idx]

# Test the search function
#latin_word = "n7eb mosalsel tounsi fih rajel smin 3ayli"
#arabic_word = latin_to_arabic_text(latin_word)
#print("jomla motarjma",arabic_word)  # Résultat attendu : "وجهك تطبع ف فلوس"
query = "i want series about music teacher"
recommended_series = semantic_search(query)
print(f"Recommended Series: {recommended_series['title']}")
