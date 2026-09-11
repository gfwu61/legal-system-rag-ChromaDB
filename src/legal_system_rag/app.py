from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit.components.v1 import html

from legal_system_rag.config.settings import (
    EMBEDDING_MODEL,
    IGNORE_SSL,
    LLM_ANSWER_MODEL,
    LLM_QUERY_MODEL,
    PERSIST_DIRECTORY,
)
from legal_system_rag.network.client_factory import (
    create_http_client,
    get_proxy_url,
)
from legal_system_rag.rag.agentic_rag import run_agentic_rag_pipeline
from legal_system_rag.rag.chain import build_rag_chain, initialize_global_bm25
import warnings


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================
# EXAMPLE USER QUESTIONS
# ============================================================

HELP_QUESTIONS = {
    "Kündigung": [
        "wie lang ist die Kündigungsfrist für den Mieter?",
        "wie lang ist die Kündigungsfrist für den Vermieter?",
        "wie ist die Kündigungsfrist für den Vermieter in §573 und 573c geregelt?",
        "wie schnell kann ich aus meiner Wohnung als Mieter aus?",
        "Unter welchen Voraussetzungen kann ein Vermieter meinem Mietverhältnis ordentlich kündigen?",
        "Darf mein Vermieter wegen Eigenbedarfs kündigen?",
        "Welche Kündigungsfrist muss ein Vermieter bei einer normalen Kündigung einhalten?",
        "Wie verlängert sich die Kündigungsfrist für den Vermieter nach fünf oder acht Jahren?",
        "Wann kann ein Vermieter eine Wohnung wegen wirtschaftlicher Verwertung kündigen?",
        "Was gilt bei einer Kündigung eines Vermieters in einem Gebäude mit höchstens zwei Wohnungen?",
        "Welche Angaben muss ein Vermieter im Kündigungsschreiben machen?",
    ],
    "Mietzahlung & Betriebskosten": [
        "Bis wann muss ich als Mieter meine Miete bezahlen?",
        "Kann ich eine Forderung gegen den Vermieter mit der Miete verrechnen?",
        "Wann muss der Vermieter über die Betriebskosten abrechnen?",
        "Was passiert, wenn der Vermieter die Betriebskostenabrechnung zu spät erstellt?",
        "Wie werden Betriebskosten auf die Mieter verteilt?",
        "Kann ich als Mieter die Belege der Betriebskostenabrechnung einsehen?",
        "Darf der Vermieter die Miete bei einem neuen Mietvertrag beliebig festlegen?",
        "Wie hoch darf die Miete bei einem angespannten Wohnungsmarkt sein?",
        "Welche Rolle spielt die Vormiete bei der zulässigen Miethöhe?",
        "Welche Informationen muss mir der Vermieter zur Miethöhe geben?",
    ],
}


# ============================================================
# NETWORK CLIENT
# ============================================================

@st.cache_resource
def load_http_client(
    proxy_url: str | None,
):
    """
    Create and cache a synchronous HTTPX client for network requests.
    """
    return create_http_client(
        proxy_url=proxy_url,
        ignore_ssl=IGNORE_SSL,
    )


# ============================================================
# RESOURCE LOADING
# ============================================================

@st.cache_resource
def load_resources(
    _sync_client,
    proxy_url: str | None,
):
    """
    Initialize OpenAI models, ChromaDB vector store, and global BM25 index.
    The global BM25 index is built once in memory to avoid redundant re-indexing.
    """
    del proxy_url

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        http_client=_sync_client,
    )

    llm_query = ChatOpenAI(
        model=LLM_QUERY_MODEL,
        temperature=0.1,
        http_client=_sync_client,
    )

    llm_answer = ChatOpenAI(
        model=LLM_ANSWER_MODEL,
        temperature=0.0,
        http_client=_sync_client,
    )

    persist_path = Path(PERSIST_DIRECTORY)

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store directory '{persist_path}' does not exist."
        )

    if not any(persist_path.iterdir()):
        raise FileNotFoundError(
            f"Vector store directory '{persist_path}' is empty."
        )

    vector_store = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )

    # Initialize the global BM25 index once for the entire database
    global_bm25 = initialize_global_bm25(vector_store)

    return (
        llm_query,
        llm_answer,
        vector_store,
        global_bm25,
    )


# ============================================================
# RAG CHAIN (MODE 1)
# ============================================================

@st.cache_resource
def load_rag_chain(
    _llm_query,
    _llm_answer,
    _vector_store,
    _global_bm25,
    retriever_mode: int,
):
    """
    Build and cache the standard RAG chain (Mode 1), injecting pre-built BM25 index and retriever mode.
    """
    return build_rag_chain(
        _llm_query,
        _llm_answer,
        _vector_store,
        _global_bm25,
        retriever_mode=retriever_mode,
    )


# ============================================================
# RESOURCE INITIALIZATION
# ============================================================

def initialize_resources(retriever_mode: int = 1) -> dict[str, Any]:
    """
    Initialize all system resources including models, vector database, BM25, and RAG chain.
    """
    proxy_url = get_proxy_url()
    sync_client = load_http_client(proxy_url)

    (
        llm_query,
        llm_answer,
        vector_store,
        global_bm25,
    ) = load_resources(
        sync_client,
        proxy_url,
    )

    rag_chain = load_rag_chain(
        llm_query,
        llm_answer,
        vector_store,
        global_bm25,
        retriever_mode=retriever_mode,
    )

    return {
        "rag_chain": rag_chain,
        "llm_query": llm_query,
        "llm_answer": llm_answer,
        "vector_store": vector_store,
        "global_bm25": global_bm25,
    }


# ============================================================
# SOURCE RENDERING
# ============================================================

def render_sources(
    source_docs: list[Any] | None,
) -> None:
    """
    Display retrieved legal documents and metadata inside an expandable UI container.
    """
    if not source_docs:
        return

    with st.expander("📚 Retrieved Legal Texts & Sources"):
        for index, doc in enumerate(source_docs, start=1):
            metadata = getattr(doc, "metadata", {}) or {}
            page_content = getattr(doc, "page_content", "")
            paragraph = metadata.get("paragraph", "Unknown")
            source = metadata.get("source")

            title = f"**Source {index}: Paragraph {paragraph}**"
            if source:
                title += f"  \nFile: `{source}`"

            st.markdown(title)
            st.caption(page_content)


# ============================================================
# CHAT HISTORY
# ============================================================

def render_chat_history() -> None:
    """
    Render all messages saved in the Streamlit session state.
    """
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                render_sources(message.get("docs"))


# ============================================================
# HELP COLUMN
# ============================================================

def render_help_column() -> str | None:
    """
    Display interactive example legal questions grouped by category.
    """
    selected_question = None

    st.subheader("💡 Example Questions")
    st.caption("Click a question to submit it directly to the chatbot.")

    for category, questions in HELP_QUESTIONS.items():
        with st.expander(category, expanded=True):
            for index, question in enumerate(questions):
                button_key = f"help_{category}_{index}"
                if st.button(question, key=button_key, use_container_width=True):
                    selected_question = question

    return selected_question


# ============================================================
# RAG QUESTION PROCESSING
# ============================================================

def process_question(
    resources: dict[str, Any],
    user_input: str,
    rag_mode: int,
) -> None:
    """
    Execute question processing using either Mode 1 (Standard RAG) or Mode 2 (Agentic RAG Loop).
    """
    # 1. Append user prompt to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # 2. Execute RAG pipeline based on selected mode
    try:
        if rag_mode == 1:
            # Mode 1: Standard linear RAG execution chain
            result = resources["rag_chain"].invoke(user_input)
            answer_text = result.get("answer", "No answer could be generated.")
            source_docs = result.get("docs", [])
        else:
            # Mode 2: Agentic evaluation loop with automated query rewriting
            result = run_agentic_rag_pipeline(
                question=user_input,
                llm_query=resources["llm_query"],
                llm_answer=resources["llm_answer"],
                vector_store=resources["vector_store"],
                global_bm25=resources["global_bm25"],
            )
            answer_text = result.get("answer", "No answer could be generated.")
            source_docs = result.get("source_documents", [])

        # 3. Append assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "docs": source_docs,
            }
        )

    except Exception as error:
        error_message = (
            "❌ Error while processing the question: "
            f"{type(error).__name__}: {error}"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "docs": [],
            }
        )


# ============================================================
# AUTO SCROLL
# ============================================================

def scroll_chat_to_bottom() -> None:
    """
    Scroll the chat UI container down to display the latest message automatically.
    """
    html(
        """
        <script>
        function scrollChatToBottom() {
            const doc = window.parent.document;
            const elements = doc.querySelectorAll('[data-testid="stVerticalBlock"]');
            let scrollContainer = null;

            for (const element of elements) {
                const style = window.getComputedStyle(element);
                const isScrollable = element.scrollHeight > element.clientHeight;
                const hasOverflow = style.overflowY === "auto" || style.overflowY === "scroll";

                if (isScrollable && hasOverflow) {
                    scrollContainer = element;
                }
            }

            if (scrollContainer) {
                scrollContainer.scrollTop = scrollContainer.scrollHeight;
            }
        }

        setTimeout(scrollChatToBottom, 100);
        setTimeout(scrollChatToBottom, 300);
        setTimeout(scrollChatToBottom, 600);
        </script>
        """,
        height=0,
    )


# ============================================================
# MAIN APPLICATION ENTRYWAY
# ============================================================

def main() -> None:
    """
    Main execution loop for Streamlit application. Sets up sidebar mode selection,
    renders chat layout, handles user input, and invokes the chosen pipeline mode.
    """
    # --------------------------------------------------------
    # Page Header & Sidebar Setup
    # --------------------------------------------------------
    st.title("⚖️ Legal RAG System")
    st.caption("Legal answers based on the configured Chroma database")

    # --------------------------------------------------------
    # Sidebar: Pipeline & Retriever Configuration
    # --------------------------------------------------------
    st.sidebar.header("⚙️ Pipeline Configuration")
    
    selected_mode_label = st.sidebar.radio(
        "Select RAG Pipeline Mode:",
        options=[
            "Standard RAG (Mode 1)",
            "Self-Corrective RAG (Mode 2)"
        ],
        index=0,
        help="Mode 1 runs a standard single-pass retrieval. Mode 2 executes an agentic loop with self-evaluations and query rewrites."
    )
    
    rag_mode = 1 if "Standard RAG" in selected_mode_label else 2

    # Standardwert festlegen
    retriever_mode = 1 

    # Zeige die Optionen NUR an, wenn Standard RAG (Mode 1) gewählt ist:
    if rag_mode == 1:
        with st.sidebar.container():
            st.sidebar.markdown("---")
            retriever_mode_label = st.sidebar.radio(
                "↳ Select Retriever Mode (for Mode 1):",
                options=[
                    "Mode 1: Normal (Only Reranking)",
                    "Mode 2: With Score (Score Blending)"
                ],
                index=0,
                help="Mode 1 uses plain reranking. Mode 2 blends scores from base retrievers."
            )
            retriever_mode = 1 if "Mode 1: Normal" in retriever_mode_label else 2
        st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Session State & Resource Initialization
    # --------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        # Hier wird der retriever_mode nun korrekt mitgegeben
        resources = initialize_resources(retriever_mode=retriever_mode)
    except FileNotFoundError as error:
        st.warning(f"⚠️ {error}")
        st.info("💡 Please index your documents first.")
        st.stop()
    except Exception as error:
        st.error(f"❌ Initialization failed: {type(error).__name__}: {error}")
        st.stop()

    # --------------------------------------------------------
    # UI Layout setup
    # --------------------------------------------------------
    chat_column, help_column = st.columns(
        [2.2, 1],
        gap="large",
    )

    with chat_column:
        st.subheader("💬 Legal Chatbot")

        user_input = st.chat_input("Ask your legal question in German...")

        chat_area = st.container(
            height=620,
            border=True,
        )

        with chat_area:
            render_chat_history()

    with help_column:
        selected_question = render_help_column()

    # --------------------------------------------------------
    # Input Selection & Execution
    # --------------------------------------------------------
    question = selected_question if selected_question else user_input

    if not question or not question.strip():
        return

    question = question.strip()
    print("???? rag_mode:", rag_mode, "| retriever_mode:", retriever_mode)
    spinner_text = (
        f"Searching database with standard pipeline (Retriever Mode {retriever_mode})..."
        if rag_mode == 1
        else "Executing agentic retrieval loop with query rewrites..."
    )

    with st.spinner(spinner_text):
        process_question(
            resources=resources,
            user_input=question,
            rag_mode=rag_mode,
        )

    # Trigger UI rerun to display question and answer sequentially
    st.rerun()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()