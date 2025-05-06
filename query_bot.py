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
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent

# === Config ===
FAISS_INDEX_DIR = "faiss_index" # Directory where FAISS index is stored
OPENAI_API_KEY = "OPENAI_API_KEY"  # Replace with your real key
OPENAI_API_KEY = os.getenv(OPENAI_API_KEY) # Ensure this is set in your environment


# === Load FAISS Vector Store ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# === Streamlit UI ===
st.set_page_config(page_title="VoiceIQ Agentic RAG Bot", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>🤖 VoiceIQ Agentic RAG Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Your smart assistant for analyzing customer care call reports</p>", unsafe_allow_html=True)
st.divider()

# === Session State ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# === Display Chat History ===
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Tool Definition ===
def search_documents(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."

tool = Tool(
    name="SearchReports",
    func=search_documents,
    description="Use this to search and analyze customer care call reports."
)

# === AgentExecutor with Error Recovery ===
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=initialize_agent(
        tools=[tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=st.session_state.memory,
        verbose=True
    ).agent,
    tools=[tool],
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors=True
)

# === Chat Input ===
prompt = st.chat_input("💬 Ask VoiceIQ something about your reports...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Run agent
    result = agent_executor.invoke({"input": prompt})
    final_output = result["output"]
    steps = result.get("intermediate_steps", [])

    # Assistant response
    with st.chat_message("assistant"):
        animated_text = st.empty()
        displayed = ""
        for char in final_output:
            displayed += char
            animated_text.markdown(displayed + "▌")
            time.sleep(0.015)
        animated_text.markdown(displayed)

    # Document metadata
    docs = retriever.invoke(prompt)
    if docs:
        with st.expander("📄 Retrieved Document Metadata", expanded=False):
            for i, doc in enumerate(docs[:3]):
                st.markdown(f"**🔹 Chunk {i+1} Metadata:**")
                for key, value in doc.metadata.items():
                    st.markdown(f"- **{key}**: `{value}`")
                st.markdown("---")

    # Reasoning steps
    if steps:
        with st.expander("🧠 Agent Reasoning", expanded=False):
            for i, step in enumerate(steps):
                st.markdown(f"**Step {i+1}**")
                st.markdown(f"🧠 Thought: {step[0].log.strip()}")
                st.markdown(f"🔍 Observation: {step[1]}")
                st.markdown("---")

    # Memory
    st.session_state.memory.chat_memory.add_user_message(prompt)
    st.session_state.memory.chat_memory.add_ai_message(final_output)
    st.session_state.chat_messages.append({"role": "assistant", "content": final_output})

# === Upload Reports ===
with st.expander("📎 Upload Report (HTML or PDF)", expanded=False):
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
            st.success(f"✅ Indexed {len(documents)} chunks from **{uploaded_file.name}**")
