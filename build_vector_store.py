import os
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# === Config ===
OPENAI_API_KEY = "YOUR_OPENAI+API_KEY"  # Replace with your real key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)  # Ensure you have the OpenAI API key set in your environment or replace it here
REPORTS_DIR = "reports" # Directory containing HTML reports (change it to your actual directory)
FAISS_INDEX_DIR = "faiss_index" # Directory to save the FAISS index

# === Metadata extraction ===
def extract_metadata(text: str):
    def match(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "call_id": match(r"Call ID[:\-]?\s*(CALL-\d{4}-\d{3})"),
        "date": match(r"Date[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"),
        "rep_id": match(r"Representative(?: ID)?[:\-]?\s*(REP-\d+)"),
        "customer_id": match(r"Customer(?: ID)?[:\-]?\s*(CUST-\d+)"),
        "intent": match(r"Intent[:\-]?\s*(.+?)(?:\n|$)"),
        "customer_sentiment": match(r"Customer Sentiment[:\-]?\s*(\w+)"),
        "rep_sentiment": match(r"Representative Sentiment[:\-]?\s*(\w+)"),
        "keywords": match(r"Keywords[:\-]?\s*(.+?)(?:\n|$)")
    }

# === Load and chunk reports ===
all_documents = []
for fname in os.listdir(REPORTS_DIR):
    if fname.endswith(".html"):
        path = os.path.join(REPORTS_DIR, fname)
        print(f"🟠 Loading HTML: {fname}")
        with open(path, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            full_text = soup.get_text(separator=' ', strip=True)

        metadata = extract_metadata(full_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={**metadata, "source": fname, "chunk": i}
            )
            all_documents.append(doc)

print(f"\n✅ Prepared {len(all_documents)} chunks with enriched metadata.")

# === Embed and save ===
print("🔗 Embedding with OpenAI...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.from_documents(all_documents, embeddings)
db.save_local(FAISS_INDEX_DIR)

print(f"\n✅ FAISS index saved to: {FAISS_INDEX_DIR}")
