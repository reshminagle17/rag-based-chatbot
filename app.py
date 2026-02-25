# app.py
import streamlit as st
from rag_pipeline import build_vectorstore
from graph import run_query
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# Build index on first run
if not os.path.exists("faiss_index"):
    with st.spinner("Building knowledge base..."):
        build_vectorstore()
    st.success("Knowledge base ready!")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Chatbot")
    st.markdown("**Powered by:**")
    st.markdown("- LangGraph Workflow")
    st.markdown("- FAISS Vector Store")
    st.markdown("- GPT-3.5 Turbo")
    st.divider()
    st.markdown("**LangGraph Nodes:**")
    nodes = ["📥 Input Processing","🎯 Intent Routing",
             "🔍 Retrieval","✅ Context Validation",
             "💬 Response Generation","🔄 Fallback","📝 Formatter"]
    for n in nodes:
        st.markdown(f"  {n}")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

# ── Chat History ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 RAG Chatbot with LangGraph")
st.caption("Ask anything grounded in the knowledge base.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through LangGraph workflow..."):
            result = run_query(prompt)

        response = result["formatted_response"]
        st.markdown(response)

        with st.expander("🔍 Workflow Details"):
            st.write(f"**Intent:** {result['intent']}")
            st.write(f"**Confidence:** {result['confidence']}")
            st.write(f"**Retrieved Chunks:** {len(result['context'])}")
            if result["context"]:
                st.write("**Context Used:**")
                for i, c in enumerate(result["context"], 1):
                    st.text_area(f"Chunk {i}", c, height=80, key=f"chunk_{i}_{prompt}")

    st.session_state.messages.append({"role": "assistant", "content": response})