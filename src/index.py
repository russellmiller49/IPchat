import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai


def gemini_embed(text: str):
    """Return an embedding vector from the Gemini API."""
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return result["embedding"]


client = chromadb.Client()
collection = client.create_collection(
    name="interventional_pulm",
    embedding_function=embedding_functions.Function(gemini_embed),
)


def index_text(doc_id: str, text: str) -> None:
    collection.add(documents=[text], ids=[doc_id])


if __name__ == "__main__":
    text_dir = Path("data/pdfs")
    for txt_file in text_dir.glob("*.txt"):
        index_text(txt_file.stem, txt_file.read_text(encoding="utf-8"))
