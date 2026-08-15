from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit.components.v1 import html

from legal_system_rag.config.settings import (
    EMBEDDING_MODEL,
    LLM_ANSWER_MODEL,
    LLM_QUERY_MODEL,
    PERSIST_DIRECTORY,
)
from legal_system_rag.network.client_factory import (
    create_http_client,
    get_proxy_url,
)
from legal_system_rag.rag.chain import build_rag_chain


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
def load_http_client(proxy_url: str | None):
    """
    Creates and caches a synchronous HTTPX client.

    The client is reused across Streamlit interactions to avoid
    creating a new HTTP connection for every request.
    """

    return create_http_client(
        proxy_url=proxy_url
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
    Initializes the OpenAI embeddings, LLMs, and ChromaDB.

    The HTTP client is used for OpenAI communication.

    The leading underscore in _sync_client prevents Streamlit
    from using the client itself as a cache key.

    proxy_url is intentionally retained as a cache dependency
    for the network configuration.
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

    persist_path = Path(
        PERSIST_DIRECTORY
    )

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store directory "
            f"'{persist_path}' does not exist."
        )

    if not any(persist_path.iterdir()):
        raise FileNotFoundError(
            f"Vector store directory "
            f"'{persist_path}' is empty."
        )

    vector_store = Chroma(
        persist_directory=str(
            persist_path
        ),
        embedding_function=embeddings,
    )

    return (
        llm_query,
        llm_answer,
        vector_store,
    )


# ============================================================
# RAG CHAIN
# ============================================================

@st.cache_resource
def load_rag_chain(
    _llm_query,
    _llm_answer,
    _vector_store,
):
    """
    Builds and caches the RAG chain.
    """

    return build_rag_chain(
        _llm_query,
        _llm_answer,
        _vector_store,
    )


# ============================================================
# SOURCE RENDERING
# ============================================================

def render_sources(
    source_docs: list[Any] | None,
) -> None:
    """
    Displays the documents retrieved by the RAG system.

    Each source displays its paragraph number, source file,
    and retrieved legal text.
    """

    if not source_docs:
        return

    with st.expander(
        "📚 Retrieved Legal Texts & Sources"
    ):

        for index, doc in enumerate(
            source_docs,
            start=1,
        ):

            metadata = getattr(
                doc,
                "metadata",
                {},
            ) or {}

            page_content = getattr(
                doc,
                "page_content",
                "",
            )

            paragraph = metadata.get(
                "paragraph",
                "Unknown",
            )

            source = metadata.get(
                "source"
            )

            title = (
                f"**Source {index}: "
                f"Paragraph {paragraph}**"
            )

            if source:
                title += (
                    f"  \nFile: `{source}`"
                )

            st.markdown(title)
            st.caption(page_content)


# ============================================================
# CHAT HISTORY
# ============================================================

def render_chat_history() -> None:
    """
    Renders the conversation history stored in Streamlit
    session state.
    """

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(role):

            st.markdown(content)

            if role == "assistant":

                render_sources(
                    message.get("docs")
                )


# ============================================================
# HELP COLUMN
# ============================================================

def render_help_column() -> str | None:
    """
    Displays example legal questions in the help column.

    Returns the selected example question when the user
    clicks one of the example buttons.
    """

    selected_question = None

    # --------------------------------------------------------
    # Title is intentionally outside any fixed-height area.
    # --------------------------------------------------------

    st.subheader(
        "💡 Example Questions"
    )

    st.caption(
        "Click a question to use it directly "
        "in the chatbot."
    )

    for category, questions in HELP_QUESTIONS.items():

        with st.expander(
            category,
            expanded=True,
        ):

            for index, question in enumerate(
                questions
            ):

                button_key = (
                    f"help_{category}_{index}"
                )

                if st.button(
                    question,
                    key=button_key,
                    use_container_width=True,
                ):

                    selected_question = question

    return selected_question


# ============================================================
# RESOURCE INITIALIZATION
# ============================================================

def initialize_resources():
    """
    Initializes the proxy, HTTP client, LLMs, vector store
    and RAG chain.
    """

    proxy_url = get_proxy_url()

    sync_client = load_http_client(
        proxy_url
    )

    (
        llm_query,
        llm_answer,
        vector_store,
    ) = load_resources(
        sync_client,
        proxy_url,
    )

    rag_chain = load_rag_chain(
        llm_query,
        llm_answer,
        vector_store,
    )

    return rag_chain


# ============================================================
# RAG QUESTION PROCESSING
# ============================================================

def process_question(
    rag_chain,
    user_input: str,
) -> None:
    """
    Executes the RAG pipeline for a user question.

    The question is added to the conversation history,
    the RAG chain is invoked, and the generated answer
    together with the retrieved sources is displayed
    and stored.
    """

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Execute RAG pipeline
    # --------------------------------------------------------

    try:

        result = rag_chain.invoke(
            user_input
        )

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "The RAG chain must return "
                "a dictionary."
            )

        answer_text = str(
            result.get(
                "answer",
                "No answer could be generated.",
            )
        )

        source_docs = result.get(
            "docs",
            [],
        )

        # ----------------------------------------------------
        # Store assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "docs": source_docs,
            }
        )

    except Exception as error:

        error_message = (
            "❌ Error while processing "
            "the question: "
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
    Scrolls the chat area to the newest message.

    The script searches for the scrollable Streamlit
    container and moves it to its bottom.
    """

    html(
        """
        <script>

        function scrollChatToBottom() {

            const doc = window.parent.document;

            const elements = doc.querySelectorAll(
                '[data-testid="stVerticalBlock"]'
            );

            let scrollContainer = null;

            for (const element of elements) {

                const style =
                    window.getComputedStyle(element);

                const isScrollable =
                    element.scrollHeight >
                    element.clientHeight;

                const hasOverflow =
                    style.overflowY === "auto" ||
                    style.overflowY === "scroll";

                if (
                    isScrollable &&
                    hasOverflow
                ) {
                    scrollContainer = element;
                }
            }

            if (scrollContainer) {

                scrollContainer.scrollTop =
                    scrollContainer.scrollHeight;
            }
        }


        setTimeout(
            scrollChatToBottom,
            100
        );

        setTimeout(
            scrollChatToBottom,
            300
        );

        setTimeout(
            scrollChatToBottom,
            600
        );

        </script>
        """,
        height=0,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Runs the Streamlit Legal RAG application.

    The function initializes the RAG infrastructure,
    displays the chat history, provides example questions,
    accepts user questions, invokes the RAG chain, and
    displays the generated answer together with the
    retrieved legal sources.
    """

    # --------------------------------------------------------
    # Page header
    # --------------------------------------------------------

    st.title(
        "⚖️ Legal RAG System"
    )

    st.caption(
        "Legal answers based on the configured "
        "Chroma database"
    )

    # --------------------------------------------------------
    # Initialize chat history
    # --------------------------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # --------------------------------------------------------
    # Initialize RAG resources
    # --------------------------------------------------------

    try:

        rag_chain = initialize_resources()

    except FileNotFoundError as error:

        st.warning(
            f"⚠️ {error}"
        )

        st.info(
            "💡 Please index your documents first."
        )

        st.stop()

    except Exception as error:

        st.error(
            "❌ Initialization failed: "
            f"{type(error).__name__}: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # Main layout
    # --------------------------------------------------------

    chat_column, help_column = st.columns(
        [2.2, 1],
        gap="large",
    )

    # ========================================================
    # CHAT COLUMN
    # ========================================================

    with chat_column:

        # ----------------------------------------------------
        # IMPORTANT:
        # Title remains outside the scrollable area.
        # ----------------------------------------------------

        st.subheader(
            "💬 Legal Chatbot"
        )

        # ----------------------------------------------------
        # Question input
        # ----------------------------------------------------

        user_input = st.chat_input(
            "Ask your legal question in German..."
        )

        # ----------------------------------------------------
        # Scrollable chat area
        # ----------------------------------------------------

        chat_area = st.container(
            height=620,
            border=True,
        )

        with chat_area:

            # ------------------------------------------------
            # Conversation history
            # ------------------------------------------------

            render_chat_history()

    # ========================================================
    # HELP COLUMN
    # ========================================================

    with help_column:

        # ----------------------------------------------------
        # Title remains outside any fixed-height area.
        # ----------------------------------------------------

        selected_question = (
            render_help_column()
        )

    # ========================================================
    # DETERMINE QUESTION
    # ========================================================

    question = (
        selected_question
        if selected_question
        else user_input
    )

    if not question:

        return

    if not question.strip():

        return

    question = question.strip()

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    with st.spinner(
        "Searching the database and "
        "generating answer..."
    ):

        process_question(
            rag_chain,
            question,
        )

    # --------------------------------------------------------
    # Rerun so that the new question and answer are rendered
    # together in the correct order.
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()