import os
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# === Config ===
FAISS_INDEX_DIR = "faiss_index"
OPENAI_API_KEY = ""

# === Load FAISS + retriever
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# === LLM setup
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# === Streamlit Page UI ===
st.set_page_config(page_title="📞 Customer Care Analytics Bot", page_icon="📞")
st.title("📞 Customer Care Analytics Bot")

# === Session states: chat memory + UI chat messages
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# === Display chat messages as bubbles
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Input box at bottom
user_input = st.chat_input("Ask a question about call reports...")

# === Chat logic
if user_input:
    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # === Get relevant context from documents
    docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No relevant reports found."

    # === Build prompt with history + context
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

User: {user_input}
Assistant:"""

    # === Get LLM response
    raw_response = llm.invoke(full_prompt).content

    # === Typing animation
    with st.chat_message("assistant"):
        animated_text = st.empty()
        displayed = ""
        for char in raw_response:
            displayed += char
            animated_text.markdown(displayed + "▌")
            time.sleep(0.015)  # typing speed
        animated_text.markdown(displayed)

    # === Store everything
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(raw_response)
    st.session_state.chat_messages.append({"role": "assistant", "content": raw_response})

