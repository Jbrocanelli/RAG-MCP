from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

_vector_db = None

def get_or_create_vector_db(persist_directory: str, documents: list = None, model_name: str = "all-MiniLM-L6-v2"):
      global _vector_db
      embedding = HuggingFaceEmbeddings(model_name=model_name)
      if Path(persist_directory).exists():
            _vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
      else:
            if not documents:
                  raise ValueError("Documents are empty, cannot create vector db")
            _vector_db = Chroma.from_documents(
          documents=documents,
          embedding=embedding,
          persist_directory=persist_directory,
      )
      return _vector_db
