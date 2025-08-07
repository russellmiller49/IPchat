import os

import chromadb
import google.generativeai as genai


def gemini_embed(text: str):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return result["embedding"]


client = chromadb.Client()
collection = client.get_collection("interventional_pulm")


def answer_query(query: str) -> str:
    """Retrieve context and query Gemini to answer."""
    query_vec = gemini_embed(query)
    results = collection.query(query_embeddings=[query_vec], n_results=3)
    context = "\n".join(results["documents"][0])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(f"Context:\n{context}\n\nQuestion: {query}")
    return response.text


if __name__ == "__main__":
    while True:
        question = input("Question: ")
        if not question:
            break
        print(answer_query(question))
