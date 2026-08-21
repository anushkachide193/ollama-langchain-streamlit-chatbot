from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def extract_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError(
            "No readable text found in the PDF."
        )

    return text.strip()


def split_text(text):
    """Split text into smaller chunks for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    return splitter.split_text(text)


def create_vector_text(text):
    """Create FAISS vector store from text."""
    chunks = split_text(text)

    if not chunks:
        raise ValueError(
            "No text available to create vector store."
        )

    documents = [
        Document(
            page_content=chunk
        )
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore
