from __future__ import annotations

from pathlib import Path

import streamlit as st

from chatbot import HealthcareChatbot


DATA_PATH = Path(__file__).with_name("healthcare_data.txt")


def load_documents(path: Path) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8")
    docs = []
    for block in [doc.strip() for doc in raw_text.split("\n\n") if doc.strip()]:
        if block.startswith("Topic:"):
            header, _, body = block.partition("|")
            topic = header.replace("Topic:", "").strip()
            text = body.strip()
        else:
            topic = "General"
            text = block
        docs.append({"topic": topic, "text": text})
    return docs


@st.cache_resource
def get_chatbot_instance() -> HealthcareChatbot:
    return HealthcareChatbot(data_path=DATA_PATH)


def configure_chatbot(top_k: int, max_tokens: int, min_score: float) -> HealthcareChatbot:
    chatbot = get_chatbot_instance()
    chatbot.retriever.top_k = top_k
    chatbot.retriever.min_score = min_score
    chatbot.generator.max_tokens = max_tokens
    return chatbot
    if "chatbot" not in st.session_state:
        with st.spinner("Loading FLAN-T5 model..."):
            st.session_state.chatbot = HealthcareChatbot(
                top_k=top_k,
                max_tokens=max_tokens,
                data_path=DATA_PATH,
            )
    else:
        st.session_state.chatbot.retriever.top_k = top_k
        st.session_state.chatbot.generator.max_tokens = max_tokens
    return st.session_state.chatbot


st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="+",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Fraunces:opsz,wght@9..144,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #e8f6f2 0%, #f5f2e9 45%, #fef9f0 100%);
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.0);
}

.title-card {
    background: #102a2f;
    color: #fef9f0;
    padding: 1.5rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(16, 42, 47, 0.25);
}

.title-card h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
}

.title-card p {
    margin: 0;
    color: #d9efe8;
}

.chat-panel {
    background: #ffffff;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 24px rgba(16, 42, 47, 0.08);
}

.badge {
    display: inline-block;
    background: #ffba08;
    color: #2b2d2f;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="title-card">
  <span class="badge">RAG + FLAN-T5</span>
  <span class="badge">Healthcare FAQs</span>
  <h1>Healthcare RAG Assistant</h1>
  <p>Ask about symptoms, prevention, first aid, and wellness. Answers are grounded in the curated knowledge base.</p>
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Configuration")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=2, step=1)
    max_tokens = st.slider("Max output tokens", min_value=40, max_value=200, value=80, step=10)
    min_score = st.slider("Similarity threshold", min_value=0.0, max_value=0.6, value=0.1, step=0.05)

    docs = load_documents(DATA_PATH)
    topics = sorted({doc["topic"] for doc in docs})
    selected_topics = st.multiselect("Topics", options=topics, default=topics)
    show_sources = st.checkbox("Show sources", value=True)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.caption(f"Knowledge base: {len(docs)} paragraphs")

    if st.button("Clear chat"):
        st.session_state.messages = []

    st.subheader("Quick prompts")
    if st.button("Diabetes symptoms"):
        st.session_state.pending_query = "What are diabetes symptoms?"
    if st.button("Blood pressure tips"):
        st.session_state.pending_query = "How to manage blood pressure?"
    if st.button("When to seek emergency care"):
        st.session_state.pending_query = "When should I go to the emergency room?"

tab_chat, tab_experiments = st.tabs(["Chat", "Experiments"])

with tab_chat:
    st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a healthcare question...")
    if "pending_query" in st.session_state:
        prompt = st.session_state.pop("pending_query")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chatbot = configure_chatbot(top_k=top_k, max_tokens=max_tokens, min_score=min_score)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                result = chatbot.chat(
                    prompt,
                    topics=selected_topics,
                    history=st.session_state.messages,
                )
                st.markdown(result["answer"])
                if show_sources and result["sources"]:
                    with st.expander("Sources", expanded=False):
                        for source in result["sources"]:
                            st.markdown(
                                f"**Doc {source['doc_id']}** | {source['topic']} | score={source['score']:.3f}"
                            )
                            if show_chunks:
                                st.caption(source["text"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    st.markdown("</div>", unsafe_allow_html=True)

with tab_experiments:
    st.subheader("Experiment Dashboard")
    st.write("Run a quick grid to compare top_k and max_tokens.")
    exp_query_topk = st.text_input("Query for top_k tests", "What are diabetes symptoms?")
    exp_query_tokens = st.text_input("Query for max_tokens tests", "How to manage blood pressure?")
    if st.button("Run experiments"):
        chatbot = configure_chatbot(top_k=top_k, max_tokens=max_tokens, min_score=min_score)
        rows = []
        for top_k_value in [2, 3, 5]:
            chatbot.retriever.top_k = top_k_value
            result = chatbot.chat(exp_query_topk, topics=selected_topics, history=[])
            rows.append(
                {
                    "top_k": top_k_value,
                    "max_tokens": chatbot.generator.max_tokens,
                    "query": exp_query_topk,
                    "response": result["answer"],
                }
            )
        for max_token_value in [50, 80, 150]:
            chatbot.generator.max_tokens = max_token_value
            result = chatbot.chat(exp_query_tokens, topics=selected_topics, history=[])
            rows.append(
                {
                    "top_k": chatbot.retriever.top_k,
                    "max_tokens": max_token_value,
                    "query": exp_query_tokens,
                    "response": result["answer"],
                }
            )
        st.dataframe(rows, use_container_width=True)
