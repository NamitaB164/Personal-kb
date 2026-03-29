import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.memory_manager import store_message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from services.document_processor import load_and_chunk_documents_with_multiple_strategies
from database.qdrant_service import index_document_with_strategies
from langchain.schema import Document
from core.memory_manager import (
    retrieve_context_relevant_messages,
    get_all_session_messages,
    store_message,
    format_context_messages
)
from core.context_handler import (
    answer_query_with_conversation_context,
    create_context_message
)

# Constants
from database.qdrant_service import index_document_with_strategies, query_qdrant_multi_strategy, hybrid_search
from core.rag_engine import generate_answer
from services.web_scraper import get_scrape_content

def app():
    SESSION_ID = "user_session"
    DOCUMENT_COLLECTION = os.getenv("DOCUMENT_COLLECTION", "document_chunks")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Welcome! I'm your RAG assistant. Upload documents or ask questions.")
        ]
        store_message(SESSION_ID, st.session_state.messages[0].content, "system")

    if "last_retrieved_sources" not in st.session_state:
        st.session_state.last_retrieved_sources = []

    if "use_conversation_memory" not in st.session_state:
        st.session_state.use_conversation_memory = True

    # App layout with columns
    st.title("Conversation-Aware RAG System")

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    # Settings sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_files:
            os.makedirs("data/uploads", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = f"data/uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = load_and_chunk_documents_with_multiple_strategies(file_path)
                    response = index_document_with_strategies(DOCUMENT_COLLECTION, uploaded_file.name, chunks)
                    if response["status"] == "success":
                        st.success(f"Indexed {sum(len(v) for v in chunks.values())} chunks")
                    else:
                        st.error(f"Failed: {response['message']}")

        # Add memory toggle in sidebar
        st.header("Settings")
        st.session_state.use_conversation_memory = st.toggle(
            "Use conversation memory", 
            value=st.session_state.use_conversation_memory,
            help="When enabled, the assistant will use previous conversation context"
        )

    # Add sidebar for URL input
    st.sidebar.header("Upload URL")
    url = st.sidebar.text_input("Enter the URL of the website to scrape")

    # Handle URL input
    if url:
        if st.sidebar.button("Scrape URL"):
            st.sidebar.write(f"Scraping {url}...")
            try:
                scraped_file = asyncio.run(get_scrape_content(url))
                st.sidebar.write(f"Scraped content from {url}")
                chunks = load_and_chunk_documents_with_multiple_strategies(scraped_file)

                response = index_document_with_strategies(DOCUMENT_COLLECTION, url, chunks)
                if response["status"] == "success":
                    st.sidebar.success(f"Indexed {len(chunks)} chunks for {url}")
                else:
                    st.sidebar.error(f"Failed to index {url}: {response['message']}")
            except Exception as e:
                st.sidebar.error(f"Failed to scrape {url}: {str(e)}")

    with st.sidebar:
        st.header("Sources")
        if st.session_state.last_retrieved_sources:
            for i, chunk in enumerate(st.session_state.last_retrieved_sources):
                if "metadata" in chunk:
                    source = chunk["metadata"].get("source", "Unknown Source")
                else:
                    source = chunk.get("source", "Unknown Source")

                score = chunk.get("score", "N/A")
                with st.expander(f"Source {i+1} (Score: {score:.2f}) - {source}"):
                    st.write(chunk.get("text", "No text available"))

    # Main chat area (left column)
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            if isinstance(message, SystemMessage) and "Use the following context" in message.content:
                continue
            with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                st.write(message.content)

        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        if query:
            with st.chat_message("user"):
                st.write(query)

            st.session_state.messages.append(HumanMessage(content=query))
            store_message(SESSION_ID, query, "user")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_query_with_conversation_context(
                        session_id=SESSION_ID,
                        query=query,
                        use_conversation_memory=st.session_state.use_conversation_memory,
                        conversation_weight=0.3,
                        document_weight=0.7
                    )

                    st.session_state.last_retrieved_sources = response.get("sources", [])
                    if response.get("sources"):
                        st.session_state.messages.append(create_context_message(response["sources"]))

                    st.write(response["answer"])
                    st.session_state.messages.append(AIMessage(content=response["answer"]))
                    store_message(SESSION_ID, response["answer"], "assistant")

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = [SystemMessage(content="Welcome! I'm your RAG assistant. Upload documents or ask questions.")]
            st.session_state.last_retrieved_sources = []
            st.rerun()

    # Source panel (right column)
    with col2:
        st.header("Referenced Sources")

        if st.session_state.last_retrieved_sources:
            document_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "document"]
            conversation_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "conversation"]

            if document_sources:
                st.subheader("Document Chunks")
                for i, source in enumerate(document_sources):
                    with st.expander(f"Chunk {i+1} (Score: {source['score']:.2f})"):
                        st.markdown(f"**Strategy:** {source.get('strategy', 'Unknown')}")

                        if source.get("metadata"):
                            meta = source["metadata"]
                            meta_text = ""
                            if meta.get("page"):
                                meta_text += f"**Page:** {meta['page']}"
                            if meta.get("source"):
                                meta_text += f" | **Source:** {meta['source']}"
                            if meta_text:
                                st.markdown(meta_text)

                        st.markdown("---")
                        st.markdown(source["text"])

            if conversation_sources:
                st.subheader("Conversation Context")
                for i, source in enumerate(conversation_sources):
                    with st.expander(f"{source.get('role', 'Message').capitalize()} {i+1}"):
                        st.markdown(source["text"])

            st.info(f"Used {len(document_sources)} document chunks and {len(conversation_sources)} conversation references")
        else:
            st.info("Ask a question to see referenced sources here.")

def login(username, password):
    # Retrieve credentials from environment variables
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    standard_user = os.getenv("USER_USERNAME", "user")
    standard_pass = os.getenv("USER_PASSWORD", "password")
    
    credentials = {
        admin_user: admin_pass,
        standard_user: standard_pass
    }
    return credentials.get(username) == password

def login_page():
    st.title("Knowledgebase Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.get("logged_in"):
    login_page()
else:
    app()
