from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

LOADERS = {
      ".pdf": {"glob": "**/*.pdf", "loader_cls": PyMuPDFLoader, "loader_kwargs": {}},
      ".txt": {"glob": "**/*.txt", "loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
      ".md": {"glob": "**/*.md", "loader_cls": UnstructuredMarkdownLoader, "loader_kwargs": {}}
  }

def load_documents(directory: str):
    doc_dir = Path(directory)

    if not doc_dir.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    all_docs = []

    for key, loader in LOADERS.items():
        documents = DirectoryLoader(
        doc_dir,
        glob=loader["glob"],
        loader_cls=loader["loader_cls"],
        loader_kwargs=loader["loader_kwargs"]
        ).load()

        for d in documents:
            d.metadata["file_name"] = Path(d.metadata["source"]).name 
            d.metadata["file_type"] = key.lstrip(".")

        all_docs.extend(documents)

    return all_docs

def load_single_document(file_path: str):
    path = Path(file_path)
    config = LOADERS.get(path.suffix)

    if config is None:
        raise ValueError(f"Unsuported file type {path.suffix}")
    
    docs = config["loader_cls"](str(path), **config["loader_kwargs"]).load()

    for d in docs:
        d.metadata["file_name"] = Path(d.metadata["source"]).name 
        d.metadata["file_type"] = path.suffix.lstrip(".")

    return docs


def split_documents(documents: list, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter.split_documents(documents)
