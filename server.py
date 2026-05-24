from mcp.server.fastmcp import FastMCP
from pathlib import Path
from src.ingestion import load_documents, load_single_document, split_documents
from src.vectorstore import get_or_create_vector_db
from src.retriever import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()

PERSIST_DIR = "data/vectorstore"

mcp = FastMCP("RAG-Pipeline",host=os.getenv("MCP_HOST"),json_response=True)

_chain = None

@mcp.tool()
def ask(prompt: str) -> str:
      """Answer questions using the document knowledge base. 
    Use this tool for any question that could be answered by the indexed documents."""
      try:
        result = get_chain().invoke(prompt)
        sources = set(doc.metadata["file_name"] for doc in result["source_documents"])
        return f"{result['answer']}\n\nSources: {', '.join(sources)}"
      except Exception as e:
          return f"Error querying database : {str(e)}"

@mcp.tool()
def add_document(file_path: str) -> str:
    """Ingest a new document into the knowledge base. 
    Use this when the user wants to add a file to the RAG pipeline."""
    global _chain
    try:
        path = Path(file_path)
        documents = load_documents(str(path)) if path.is_dir() else load_single_document(str(path))
        chunks = split_documents(documents)
        get_or_create_vector_db(persist_directory=PERSIST_DIR).add_documents(chunks)
        _chain = None
        return f"Ingested {len(chunks)} chunks"
    except Exception as e:
        return f"Error ingesting documents: {str(e)}"

@mcp.resource("documents://list")
def list_documents() -> str:
    """List all documents currently indexed in the knowledge base."""
    if not Path(PERSIST_DIR).exists():
        return f"No documents available"
    documents = get_or_create_vector_db(persist_directory=PERSIST_DIR).get()
    files = set(metadata["file_name"] for metadata in documents["metadatas"])
    return "\n".join(files)

@mcp.prompt()
def response_style(question: str, style: str = "concise") -> str:
    styles = {
        "concise": "Answer in 2-3 sentences. Stick to what the documents say. If the answer isn't inthe documents, say so.",
        "in-depth": "Give a thorough explanation with technical detail. Reference specific concepts from the documents. If the answer isn't in the documents, say so.",
        "simple": "Explain this simply, as if to someone with no ML background. Use analogies where helpful. If the answer isn't in the documents, say so.",
    }

    return f"{styles.get(style, styles["concise"])} Question: {question}"
    
def get_chain():
    global _chain
    if _chain is None:
        documents = load_documents("data")
        split_docs = split_documents(documents)
        vector_db = get_or_create_vector_db(PERSIST_DIR, documents=split_docs)
        _chain = create_retrieval_chain(vector_db)
    return _chain


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
