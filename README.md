# Smart_Retrievel_Agentic_Rag

**LLM-First. Retrieval-Next. Smarter Document Intelligence.**

SmartRetrieve is a Retrieval-Augmented Generation (RAG) bot architected to **reason before it retrieves**. Unlike traditional RAG pipelines that fetch chunks first, SmartRetrieve lets the LLM attempt an answer using prior knowledge — only retrieving additional context when needed. This leads to **faster, cleaner, and more intelligent responses**.

---

## 🚀 Key Features

- ✅ **LLM-first orchestration** – Retrieval only triggered if the LLM signals low confidence or missing context
- 📄 **HTML & PDF document support** – Embedded and indexed with metadata for precision filtering
- 🔎 **Semantic + structured search** – Query both natural language and structured insights (e.g., KPIs, sentiment, summaries)
- ⚙️ **Real-time embedding refresh** – New documents can be auto-processed and re-indexed
- 🧩 **Modular and agent-ready** – Built to support plug-and-play agents (summarizer, intent classifier, etc.)

---

## 🧠 Why LLM-First?

Most RAG systems blindly retrieve, often overwhelming the LLM with noisy context. SmartRetrieve inverts this by:

1. Letting the LLM **try to answer** using memory
2. Only then pulling relevant documents via **metadata-aware vector search**
3. Combining insights from both sources to respond precisely

---

## 💼 Use Cases

- Customer report analytics
- Internal document Q&A
- KPI aggregation from unstructured text
- LLM-driven dashboards

---

## 🔧 Tech Stack

- 🧠 OpenAI GPT / Hugging Face LLMs
- 🧩 FAISS for semantic search
- 📄 LangChain for orchestration
- 📝 Streamlit (or Flask) interface
- 🗂️ PyMuPDF, BeautifulSoup for HTML/PDF parsing

---

## Example Query Scenarios

- “Summarize all insights for customer X last quarter”

- “What is the average satisfaction score for rep Y?”

- “Show all issues discussed in February by product Z”

---

## 🛠️ Quick Start

```bash
git clone https://github.com/Harshithan07/Smart_Retrieve_Rag.git
cd Smart_Retrieve_Rag
pip install -r requirements.txt
python app.py

