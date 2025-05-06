from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

FAISS_INDEX_DIR = "faiss_index" # Directory where FAISS index is stored
OPENAI_API_KEY = "OPENAI_API_KEY"  # Replace with your real key
OPENAI_API_KEY = os.getenv(OPENAI_API_KEY) # Ensure this is set in your environment

# Load index
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Show all documents
docs = db.similarity_search("call", k=5)
for i, doc in enumerate(docs):
    print(f"\n📄 Chunk {i + 1}:")
    print("Text Snippet:", doc.page_content[:150], "...")
    print("Metadata:", doc.metadata)
