import os
import time
import streamlit as st
from bs4 import BeautifulSoup
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# === Config ===
FAISS_INDEX_DIR = "faiss_index"
OPENAI_API_KEY = ""

# === Load FAISS Vector Store ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# === Streamlit App UI ===
st.set_page_config(page_title="📞 Customer Care Analytics Bot", page_icon="📞")
st.title("📞 Customer Care Analytics Bot")

# === Session State ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# === Display previous messages ===
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat input ===
prompt = st.chat_input("Ask a question about call reports...")

# === Chat logic ===
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    docs = retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No relevant context found."

    history = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in st.session_state.memory.chat_memory.messages
    )

    full_prompt = f"""
You are a helpful assistant answering questions based on customer care reports.

Chat history:
{history}

Relevant reports:
\"\"\"
{context}
\"\"\"

User: {prompt}
Assistant:"""

    raw_response = llm.invoke(full_prompt).content

    with st.chat_message("assistant"):
        animated_text = st.empty()
        displayed = ""
        for char in raw_response:
            displayed += char
            animated_text.markdown(displayed + "▌")
            time.sleep(0.015)
        animated_text.markdown(displayed)

    st.session_state.memory.chat_memory.add_user_message(prompt)
    st.session_state.memory.chat_memory.add_ai_message(raw_response)
    st.session_state.chat_messages.append({"role": "assistant", "content": raw_response})

# === File Upload Below Chat Input ===
with st.expander("📎 Upload Report (HTML or PDF)"):
    uploaded_file = st.file_uploader(" ", label_visibility="collapsed", type=["html", "pdf"])

    def extract_text_from_file(file) -> str:
        if file.name.endswith(".html"):
            soup = BeautifulSoup(file.read(), "html.parser")
            return soup.get_text(separator=' ', strip=True)
        elif file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return ""

    if uploaded_file:
        raw_text = extract_text_from_file(uploaded_file)
        if not raw_text.strip():
            st.error("❌ No readable text found in the uploaded file.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)
            documents = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
            vectorstore.add_documents(documents)
            st.success(f"✅ Indexed {len(documents)} chunks from {uploaded_file.name}")
