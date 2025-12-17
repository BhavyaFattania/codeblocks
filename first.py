import os
import chromadb
from dotenv import load_dotenv

# --- LlamaIndex Imports ---
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# ... (rest of the imports) ...

# --- 1. ENVIRONMENT SETUP (Adjusted for Notebook) ---
load_dotenv()

# NOTE: The GROQ_API_KEY is already set in the previous cell.
os.getenv("GROQ_API_KEY")== os.environ.get("GROQ_API_KEY")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")


print("✅ Environment Variables Available.")
# --- 2. CONFIGURE CORE COMPONENTS ---

# A. Configure LLM for Generation
Settings.llm = Groq(model="openai/gpt-oss-120b")

# B. Configure Embedding Model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
print("✅ LLM and Embedding Models Configured (Groq & BGE).")

# --- 3. INDEXING AND STORAGE ---
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_free_rag_data")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 4. QUERY EXECUTION ---
# Load data (Assuming your documents are now inside the 'data' folder)
try:
    documents = SimpleDirectoryReader("data").load_data()
    print(f"✅ Loaded {len(documents)} documents.")
except Exception as e:
    print(f"❌ Data Loading ERROR: {e}")
    # You MUST upload a dummy text file to the 'data' folder to test this.

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("✅ Indexing Complete.")
