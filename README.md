# Professional Retrieval-Augmented Generation (RAG) Knowledge Base

## Project Overview

This Research-Grade Retrieval-Augmented Generation (RAG) system provides a robust framework for indexing and querying large volumes of unstructured data. By integrating multi-strategy document chunking, hybrid semantic search (Vector + Fuzzy), and conversation-aware retrieval, the system ensures high-precision information recovery and contextually accurate AI-generated responses.

The architecture is designed for scalability and professional deployment, utilizing Qdrant as a high-performance vector database and DeepSeek as the primary reasoning engine.

---

## Technical Architecture

### Core Components

1.  **Multi-Strategy Indexing:** Documents are processed using multiple chunking sizes (512, 1024, and 2048 tokens) and a rolling window approach to maintain semantic continuity across boundaries.
2.  **Hybrid Retrieval Engine:** Combines Dense Vector Search (Cosine Similarity) with Fuzzy Text Matching to handle both semantic queries and specific keyword/typo-prone searches.
3.  **Conversation-Aware Memory:** A dedicated Qdrant collection tracks chat history, which is independently embedded and queried to provide temporal context to the RAG pipeline.
4.  **Web Intelligence:** Integrated `crawl4ai` asynchronous scraping allows for dynamic knowledge expansion via URLs.

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | Python / LangChain |
| **Vector Database** | Qdrant |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **LLM Inference** | DeepSeek (via OpenAI SDK) |
| **Web Scraping** | Crawl4AI |
| **Authentication** | Bcrypt / Session-based |

---

## System Logic and Workflow

### 1. Document Ingestion and Processing
- **Normalization:** Documents (PDF, DOCX, TXT, MD) are loaded and normalized into plain text.
- **Hierarchical Chunking:** Chunks are generated at varying granulaties to capture both micro-details and macro-context.
- **Vectorization:** Each chunk is converted into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` transformer model.

### 2. Retrieval Pipeline (Hybrid Search)
When a query is received, the system performs a dual-path retrieval:
- **Vector Search:** Identifies chunks with the highest cosine similarity to the query embedding.
- **Fuzzy Search:** Performs Levenshtein-based matching to catch specific terms that might be diluted in vector space.
- **Rank Fusion:** Results are weighted and fused (default 70% Vector, 30% Fuzzy) to produce the final context set.

### 3. Conversation Awareness
- Past dialogue turns are stored in a session-isolated memory collection.
- The system retrieves relevant past exchanges to resolve pronominal references (e.g., "Tell me more about *it*") before generating the final answer.

---

## Directory Structure

```text
├── core/               # Main RAG and logic components
│   ├── auth_service.py # Authentication & Session management
│   ├── context_handler.py # Logic for merging doc & chat context
│   ├── memory_manager.py # Qdrant-based chat history management
│   └── rag_engine.py      # LLM interfacing and document processing
├── database/           # Database abstraction layers
│   └── qdrant_service.py # Qdrant connection and search logic
├── services/           # External service integrations
│   ├── document_processor.py # File parsing and chunking
│   ├── url_loader.py         # URL handling
│   └── web_scraper.py        # Asynchronous web crawling
├── utils/              # Development and debug utilities
├── data/               # Persistent storage for uploads/scrapes
└── app.py              # Application entry point (Streamlit)
```

---

## Setup and Installation

### Prerequisites
- Python 3.9+
- Qdrant Server (Running locally on port 6333 or via Docker)

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure environment variables by copying `.env.example` to `.env` and providing your DeepSeek API Key.
4.  Launch the application:
    ```bash
    streamlit run app.py
    ```

---

## Security and Compliance
- **Credential Management:** No hardcoded secrets. All API keys and credentials must be provided via the `.env` file.
- **Session Isolation:** User data and chat histories are isolated via unique session IDs and collection names.
