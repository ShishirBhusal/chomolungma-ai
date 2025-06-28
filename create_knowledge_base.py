import os
import shutil
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# Ensure your GOOGLE_API_KEY is set in your .env file

SOURCE_DIRECTORY = "knowledge_base_source_data"
PERSIST_DIRECTORY = "chroma_db_aca" # ACA for "Annapurna Conservation Area"

# --- Main Ingestion Pipeline ---
def create_and_store_knowledge_base():
    """
    Scans a directory for documents, processes them based on file type,
    chunks them, creates embeddings, and stores them in a new ChromaDB database.
    """
    print("--- Starting Knowledge Base Ingestion Pipeline ---")

    # Clean up old database directory if it exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Found old database directory. Deleting '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    # 1. Load documents from the source directory
    all_docs = []
    print(f"Scanning directory: '{SOURCE_DIRECTORY}'...")

    if not os.path.isdir(SOURCE_DIRECTORY):
        print(f"Error: Source directory '{SOURCE_DIRECTORY}' not found.")
        return

    for filename in os.listdir(SOURCE_DIRECTORY):
        file_path = os.path.join(SOURCE_DIRECTORY, filename)
        
        try:
            if filename.endswith(".pdf"):
                print(f"  - Loading PDF: {filename}")
                loader = PyMuPDFLoader(file_path)
                all_docs.extend(loader.load())
            elif filename.endswith(".txt"):
                print(f"  - Loading Text File: {filename}")
                # Use utf-8 encoding for broad compatibility
                loader = TextLoader(file_path, encoding='utf-8')
                all_docs.extend(loader.load())
            else:
                print(f"  - Skipping unsupported file type: {filename}")
        except Exception as e:
            print(f"  - Error loading {filename}: {e}")
            continue # Move to the next file

    if not all_docs:
        print("\nError: No documents were successfully loaded. Halting pipeline.")
        return
    
    print(f"\nSuccessfully loaded {len(all_docs)} document pages/files.")

    # 2. Split the documents into chunks
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    # This is a crucial step to verify our metadata is preserved.
    # The 'source' metadata comes directly from the loader.
    print(f"\nExample chunk metadata: {chunks[0].metadata}")

    # 3. Initialize the embedding model
    print("Initializing embedding model (models/embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. Create the ChromaDB vector store
    print(f"Creating and storing embeddings in ChromaDB at '{PERSIST_DIRECTORY}'...")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    
    print("\n--- Knowledge Base Ingestion Complete! ---")
    print(f"Vector database is ready at '{PERSIST_DIRECTORY}'.")

# --- Run the script ---
if __name__ == "__main__":
    create_and_store_knowledge_base()