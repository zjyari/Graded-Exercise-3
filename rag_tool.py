### COMPLETE THE CODE ###


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever

# Local MiniLM embedding model (fetched from ModelScope so Hugging Face is not required)
MODELS_DIR = Path(__file__).resolve().parent / "models"
EMBED_MODEL_PATH = str(MODELS_DIR / "all-MiniLM-L6-v2")


def build_retriever(k: int = 4) -> VectorStoreRetriever:
    """Build a vector retriever over the single-document knowledge base.

    The knowledge base is loaded through the provided loader (load_policy_documents),
    embedded with a local sentence-transformers model, and indexed with Chroma.
    """
    documents = load_policy_documents()  # the only allowed way to load the document
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever
