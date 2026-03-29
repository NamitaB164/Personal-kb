import streamlit as st
import os
import asyncio
import bcrypt
import uuid
import json
from datetime import datetime, timedelta
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
from database.qdrant_service import index_document_with_strategies, query_qdrant_multi_strategy, hybrid_search
from core.rag_engine import generate_answer
from services.web_scraper import get_scrape_content

# User authentication constants
USER_DB_FILE = "user_database.json"
SESSION_DURATION = timedelta(hours=24)

# Initialize user database if it doesn't exist
def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({"users": {}}, f)
        return {"users": {}}
    
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

# Save user database
def save_user_db(db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(db, f)

# Hash password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

# Verify password
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

# Create new user
def create_user(username, password, email):
    db = init_user_db()
    
    if username in db["users"]:
        return False, "Username already exists"
    
    # Check for existing email
    for user_data in db["users"].values():
        if user_data.get("email") == email:
            return False, "Email already registered"
    
    hashed_password = hash_password(password)
    
    db["users"][username] = {
        "password": hashed_password,
        "email": email,
        "created_at": datetime.now().isoformat(),
        "documents": []  # Track user's documents
    }
    
    save_user_db(db)
    return True, "User created successfully"

# Authenticate user
def authenticate_user(username, password):
    db = init_user_db()
    
    if username not in db["users"]:
        return False, "Invalid username or password"
    
    if not verify_password(db["users"][username]["password"], password):
        return False, "Invalid username or password"
    
    # Create session
    session_id = str(uuid.uuid4())
    expiry = (datetime.now() + SESSION_DURATION).isoformat()
    
    if "sessions" not in db:
        db["sessions"] = {}
    
    db["sessions"][session_id] = {
        "username": username,
        "expires": expiry
    }
    
    save_user_db(db)
    return True, session_id

# Validate session
def validate_session(session_id):
    if not session_id:
        return False, None
    
    db = init_user_db()
    
    if "sessions" not in db or session_id not in db["sessions"]:
        return False, None
    
    session = db["sessions"][session_id]
    expiry = datetime.fromisoformat(session["expires"])
    
    if datetime.now() > expiry:
        # Session expired
        del db["sessions"][session_id]
        save_user_db(db)
        return False, None
    
    return True, session["username"]

# Logout user
def logout_user(session_id):
    db = init_user_db()
    
    if "sessions" in db and session_id in db["sessions"]:
        del db["sessions"][session_id]
        save_user_db(db)
        return True
    
    return False

# Initialize session state for authentication
def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

# Login page
def show_login_page():
    st.title("Conversation-Aware RAG System")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", use_container_width=True):
                if username and password:
                    success, result = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.session_id = result
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Choose Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Choose Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Create Account", use_container_width=True):
            if not all([new_username, new_email, new_password, confirm_password]):
                st.warning("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long")
            elif "@" not in new_email or "." not in new_email:
                st.error("Please enter a valid email address")
            else:
                success, message = create_user(new_username, new_password, new_email)
                if success:
                    st.success(message)
                    # Auto-login after signup
                    st.session_state.authenticated = True
                    st.session_state.username = new_username
                    success, session_id = authenticate_user(new_username, new_password)
                    if success:
                        st.session_state.session_id = session_id
                        st.rerun()
                else:
                    st.error(message)

# User-specific session ID
def get_user_session_id():
    return f"{st.session_state.username}_user_session"

# Main application
def show_main_app():
    # Create user-specific document collection
    user_document_collection = f"{st.session_state.username}_documents"
    
    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"Welcome {st.session_state.username}! I'm your RAG assistant. Upload documents or ask questions.")
        ]
        store_message(get_user_session_id(), st.session_state.messages[0].content, "system")

    if "last_retrieved_sources" not in st.session_state:
        st.session_state.last_retrieved_sources = []

    if "use_conversation_memory" not in st.session_state:
        st.session_state.use_conversation_memory = True

    # App layout with columns
    st.title(f"RAG Assistant - Welcome, {st.session_state.username}!")

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    # Settings sidebar
    with st.sidebar:
        # Add logout button
        if st.button("Logout"):
            logout_user(st.session_state.session_id)
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.session_id = None
            st.rerun()
            
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        if uploaded_files:
            # Create user-specific upload directory
            user_upload_dir = f"uploads/{st.session_state.username}"
            os.makedirs(user_upload_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = f"{user_upload_dir}/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = load_and_chunk_documents_with_multiple_strategies(file_path)
                    response = index_document_with_strategies(user_document_collection, uploaded_file.name, chunks)
                    
                    # Track document in user profile
                    db = init_user_db()
                    if uploaded_file.name not in db["users"][st.session_state.username]["documents"]:
                        db["users"][st.session_state.username]["documents"].append(uploaded_file.name)
                        save_user_db(db)
                    
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
                # Create user-specific directory for scraped content
                user_scrape_dir = f"scraped/{st.session_state.username}"
                os.makedirs(user_scrape_dir, exist_ok=True)
                
                scraped_file = asyncio.run(get_scrape_content(url, output_dir=user_scrape_dir))
                st.sidebar.write(f"Scraped content from {url}")
                chunks = load_and_chunk_documents_with_multiple_strategies(scraped_file)
                        
                response = index_document_with_strategies(user_document_collection, url, chunks)
                
                # Track scraped URL in user profile
                db = init_user_db()
                if url not in db["users"][st.session_state.username]["documents"]:
                    db["users"][st.session_state.username]["documents"].append(url)
                    save_user_db(db)
                
                if response["status"] == "success":
                    st.sidebar.success(f"Indexed {sum(len(v) for v in chunks.values())} chunks for {url}")
                else:
                    st.sidebar.error(f"Failed to index {url}: {response['message']}")
            except Exception as e:
                st.sidebar.error(f"Failed to scrape {url}: {str(e)}")

    # Display user documents
    with st.sidebar:
        st.header("Your Documents")
        db = init_user_db()
        user_docs = db["users"][st.session_state.username]["documents"]
        
        if user_docs:
            for doc in user_docs:
                st.write(f"• {doc}")
        else:
            st.write("No documents uploaded yet.")

    with st.sidebar:
        st.header("Sources")
        if st.session_state.last_retrieved_sources:
            for i, chunk in enumerate(st.session_state.last_retrieved_sources):
                # Handle both structures: nested metadata dict or flattened attributes
                if "metadata" in chunk:
                    # Original structure with metadata dict
                    source = chunk["metadata"].get("source", "Unknown Source")
                else:
                    # Flattened structure from MD files
                    source = chunk.get("source", "Unknown Source")
                
                score = chunk.get("score", "N/A")
                with st.expander(f"Source {i+1} (Score: {score:.2f}) - {source}"):
                    st.write(chunk.get("text", "No text available"))

    # Main chat area (left column)
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            if isinstance(message, SystemMessage) and "Use the following context" in message.content:
                # Skip displaying context messages
                continue
            with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                st.write(message.content)

        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        if query:
            with st.chat_message("user"):
                st.write(query)
            
            st.session_state.messages.append(HumanMessage(content=query))
            store_message(get_user_session_id(), query, "user")
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_query_with_conversation_context(
                        session_id=get_user_session_id(),
                        query=query,
                        use_conversation_memory=st.session_state.use_conversation_memory,
                        conversation_weight=0.3,  # Default weights
                        document_weight=0.7,
                        collection_name=user_document_collection  # Use user-specific collection
                    )
                    
                    st.session_state.last_retrieved_sources = response.get("sources", [])
                    if response.get("sources"):
                        st.session_state.messages.append(create_context_message(response["sources"]))
                   
                    st.write(response["answer"])
                    st.session_state.messages.append(AIMessage(content=response["answer"]))
                    store_message(get_user_session_id(), response["answer"], "assistant")

        # Clear chat history button (in main column)
        if st.button("Clear Chat History"):
            st.session_state.messages = [SystemMessage(content=f"Welcome {st.session_state.username}! I'm your RAG assistant. Upload documents or ask questions.")]
            st.session_state.last_retrieved_sources = []
            st.rerun()

    # Source panel (right column)
    with col2:
        st.header("Referenced Sources")
        
        if st.session_state.last_retrieved_sources:
            # Group sources by type
            document_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "document"]
            conversation_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "conversation"]
            
            # Display document sources
            if document_sources:
                st.subheader("Document Chunks")
                for i, source in enumerate(document_sources):
                    with st.expander(f"Chunk {i+1} (Score: {source['score']:.2f})"):
                        st.markdown(f"*Strategy:* {source.get('strategy', 'Unknown')}")
                        
                        # Display metadata if available
                        if source.get("metadata"):
                            meta = source["metadata"]
                            meta_text = ""
                            if meta.get("page"):
                                meta_text += f"*Page:* {meta['page']}"
                            if meta.get("source"):
                                meta_text += f" | *Source:* {meta['source']}"
                            if meta_text:
                                st.markdown(meta_text)
                        
                        # Display chunk text
                        st.markdown("---")
                        st.markdown(source["text"])
            
            # Display conversation sources
            if conversation_sources:
                st.subheader("Conversation Context")
                for i, source in enumerate(conversation_sources):
                    with st.expander(f"{source.get('role', 'Message').capitalize()} {i+1}"):
                        st.markdown(source["text"])
                        
            
            # Display source statistics
            st.info(f"Used {len(document_sources)} document chunks and {len(conversation_sources)} conversation references")
        else:
            st.info("Ask a question to see referenced sources here.")

# Check for existing session on startup
def check_existing_session():
    # Check if we have a stored session cookie
    if "session_id" in st.session_state and st.session_state.session_id:
        valid, username = validate_session(st.session_state.session_id)
        if valid:
            st.session_state.authenticated = True
            st.session_state.username = username
            return True
    
    return False

# Main function
def main():
    # Initialize authentication state
    init_auth_state()
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("scraped", exist_ok=True)
    
    # Check for existing valid session
    session_valid = check_existing_session()
    
    # Show appropriate page based on authentication status
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()
