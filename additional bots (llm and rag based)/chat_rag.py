import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# === Config ===
FAISS_INDEX_DIR = "faiss_index"
OPENAI_API_KEY = ""
# === Load Vector Store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# === Get rep_ids for dropdown
def collect_rep_ids():
    docs = retriever.invoke("call")
    return sorted({d.metadata.get("rep_id") for d in docs if d.metadata.get("rep_id")})

available_rep_ids = collect_rep_ids()

# === Document search by query + optional rep_id
def search_documents(query, rep_id=None):
    docs = retriever.invoke(query)
    if rep_id:
        docs = [doc for doc in docs if doc.metadata.get("rep_id") == rep_id]
    return "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No relevant context found."

# === Streamlit UI setup
st.set_page_config(page_title="Conversational RAG Assistant", page_icon="🤖")
st.title("🧠 Conversational Call Report Assistant")

rep_id = st.selectbox("🔍 Filter by Representative", ["All"] + available_rep_ids)
rep_id_filter = None if rep_id == "All" else rep_id
user_input = st.text_input("💬 Ask a question (supports follow-ups):")

# === Init session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat" not in st.session_state:
    st.session_state.chat = []

# === LLM setup
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# === On query
if user_input:
    context = search_documents(user_input, rep_id_filter)
    memory = st.session_state.memory

    # Create chat history string
    chat_history = "\n".join(
        f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
        for msg in memory.chat_memory.messages
    )

    # Compose prompt with chat + context
    prompt = f"""
You are a helpful assistant analyzing customer care call reports.

Chat so far:
{chat_history}

Relevant reports:
\"\"\"
{context}
\"\"\"

Now answer this:
{user_input}
"""

    # Get response content only (not metadata)
    response = llm.invoke(prompt).content

    # Add to memory + display
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)
    st.session_state.chat.append(("🧑 You", user_input))
    st.session_state.chat.append(("🤖 Assistant", response))

# === Display full chat
for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")
