import os
import openai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

def initialize_api_and_db():
    """Initialize API key and MongoDB connection."""
    api_key = os.getenv("API_KEY")
    openai.api_key = api_key

    client = MongoClient("")
    db = client[""]
    collection = db[""]
    return collection

def initialize_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_missing_embeddings(collection, model):
    """
    Generate embeddings for reviews that do not have them
    and update them in the database.
    """
    reviews = list(collection.find({"embedding": {"$exists": False}}, 
                                   {"_id": 1, "ratingText": 1, "ratingValue": 1, "date": 1, "locationId": 1}))
    
    for review in reviews:
        review_id = review["_id"]
        rating_text = review.get("ratingText", "")
        rating_value = str(review.get("ratingValue", ""))
        date = review.get("date", "")
        locationId = review.get("locationId", "")
        
        combined_text = f"{rating_text} {rating_value} {date} {locationId}".strip()

        if combined_text:
            embedding = model.encode(combined_text).tolist()
            collection.update_one({"_id": review_id}, {"$set": {"embedding": embedding}})

    print("Embeddings generated and stored successfully for missing entries.")

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_reviews(query, locationId, collection, model, top_k=50):
    """
    Retrieve top-k relevant reviews based on cosine similarity,
    filtering by locationId.
    """
    query_embedding = model.encode(query).tolist()
    reviews = list(collection.find({"locationId": locationId}, {"_id": 1, "ratingText": 1, "ratingValue": 1, "date": 1, "locationId": 1, "embedding": 1}))
    results = []

    for review in reviews:
        if "embedding" in review:
            stored_embedding = np.array(review["embedding"])
            similarity = cosine_similarity(query_embedding, stored_embedding)

            rating_text = review.get("ratingText", "")
            rating_value = review.get("ratingValue", "")
            date = review.get("date", "")
            locationId = review.get("locationId", "")
            combined_text = f"{rating_text} {rating_value} {date} {locationId}".strip()

            results.append({"text": combined_text, "similarity": similarity})

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def generate_response(query, reviews):
    """
    Generate a response using GPT-4 based on the retrieved reviews,
    displaying only the review text without any numbering or prefixes.
    """
    if not reviews:
        return "No relevant reviews found in the database."
    context = "\n\n".join([review['text'] for review in reviews])
    prompt = f"""
    You are an AI assistant. Use the following reviews to answer the query concisely and informatively.

    Query: {query}

    Reviews:
    {context}

    Answer the query:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

def rag_pipeline(user_query, locationId, collection, model, top_k=50):
    """
    Complete RAG pipeline: retrieve reviews and generate a response,
    filtering by locationId.
    """
    relevant_reviews = retrieve_relevant_reviews(user_query, locationId, collection, model, top_k=top_k)
    response = generate_response(user_query, relevant_reviews)
    return response

if __name__ == "__main__":
    collection = initialize_api_and_db()
    model = initialize_model()
    generate_missing_embeddings(collection, model)

    locationId = int(input("Enter Location ID: "))
    user_query = input("Enter Your Query: ")
    response = rag_pipeline(user_query, locationId, collection, model)
    print("RAG Response:")
    print(response)