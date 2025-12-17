from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
load_dotenv()

os.getenv("GROQ_API_KEY")== os.environ.get("GROQ_API_KEY")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

print("✅ Environment Variables Loaded.")

# Configure LLM for Generation (Groq)
Settings.llm = Groq(model="openai/gpt-oss-120b") 

# Configure Embedding Model for Retrieval (BGE-Small)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
print("✅ LLM (Groq) and Embedding (BGE) Models Configured.")

import chromadb
DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_free_rag_data"

try:
    # A. Initialize Chroma Client to load data from the saved folder
    db = chromadb.PersistentClient(path=DB_PATH) 
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # B. Load the VectorStoreIndex object using the saved Chroma store
    # This does NOT re-embed; it loads the existing vectors and structure.
    index = VectorStoreIndex.from_vector_store(vector_store)

    print(f"✅ Index restored successfully from: {DB_PATH}")

except Exception as e:
    print(f"❌ ERROR: Failed to load index from ChromaDB at {DB_PATH}. Ensure the folder is present and correct.")
    print(e)
    exit()

# --- 3. CONFIGURE ADVANCED QUERY PIPELINE (Sentence Window + Reranking) ---

# A. Reranker (Free, Open-Source Cross-Encoder)
# Filters candidates from 10 down to the best 3.


# B. Sentence Window Post-Processor (The Context Swapper)
# Replaces the small retrieved sentence with its large context window.
post_processor_swap = MetadataReplacementPostProcessor(target_metadata_key="window")

# C. Create Query Engine
# Retrieve 10 candidates -> Rerank to 3 -> Swap to large windows -> Send to Groq.
query_engine = index.as_query_engine(
    similarity_top_k=5, 
)
# --- 4. QUERY EXECUTION ---

query_engine = index.as_query_engine()
while True:
  query = str(input("Enter your query"))
  if query == "exit":
    break
  print(f"\n❓ Querying the Index: '{query}'")

  response = query_engine.query(query)

  print("\n--- LLM Response ---")
  print(str(response))
  print("--------------------")
