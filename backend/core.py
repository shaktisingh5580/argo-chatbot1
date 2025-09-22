# backend/core.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env.local for local development
load_dotenv(dotenv_path='.env.local')

def get_db_engine():
    # Render provides a DATABASE_URL. We use it if it exists.
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Fallback for local development
        db_password = os.getenv("DB_PASSWORD")
        if not db_password: raise ValueError("DB_PASSWORD or DATABASE_URL not found in environment.")
        db_url = f"postgresql://postgres:{db_password}@localhost:5432/argo_data"
    return create_engine(db_url)

def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: raise ValueError("OPENROUTER_API_KEY not found in environment.")
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        default_headers={"HTTP-Referer": "argo-react-app-deployment"}
    )

def get_vector_store():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="chroma_db", 
        embedding_function=embedding_function,
        collection_name="argo_float_metadata"
    )