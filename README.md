# 🤖 ML & DS RAG Chatbot
### LangGraph + LangChain + FAISS + PostgreSQL + Docker + Jenkins

A production-ready **Retrieval-Augmented Generation (RAG) chatbot** that answers Machine Learning and Data Science questions by searching through a curated collection of PDF books and guides. Built with LangGraph for conversation management, FAISS for fast vector search, and PostgreSQL for persistent chat memory.

---

## 📌 What This Project Does

Instead of relying on general AI knowledge, this chatbot:

1. Loads a collection of **ML/DS PDF documents** from a local folder
2. Splits them into small searchable chunks and stores them in a **FAISS vector index**
3. When you ask a question, it **searches the PDFs** for the most relevant content
4. Sends that content to a **Groq LLM** to generate a clean, accurate answer
5. **Remembers your full conversation history** using LangGraph + PostgreSQL
6. Lets you **switch between past conversations** from the sidebar

---

## 🏗️ Architecture Overview

```
User Question
      ↓
Question Rewriting  (using chat history for context)
      ↓
FAISS Vector Search  (top 5 relevant chunks retrieved)
      ↓
LLM Answer Generation  (Groq — strictly from retrieved context)
      ↓
Answer displayed in Streamlit UI
      ↓
Conversation saved to PostgreSQL via LangGraph Checkpointing
```

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (`openai/gpt-oss-120b`) | Generates final answers |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Converts text to vectors |
| **Vector Store** | FAISS (Facebook AI) | Fast similarity search on PDF chunks |
| **RAG Framework** | LangChain | PDF loading, text splitting, chain building |
| **Conversation Graph** | LangGraph | Manages multi-turn conversation flow as a graph |
| **Memory / Persistence** | PostgreSQL + LangGraph PostgresSaver | Saves chat history permanently |
| **UI** | Streamlit | Web-based chat interface |
| **Containerization** | Docker + Docker Compose | Easy deployment anywhere |
| **CI/CD** | Jenkins | Automated testing and deployment pipeline |

---

## 📂 Project Structure

```
LANGGRAPH RAG/
│
├── config.py               # LLM, embeddings setup and folder path config
├── rag_chatbot.py          # Core RAG pipeline (load → split → embed → retrieve → answer)
├── backend.py              # LangGraph graph, PostgreSQL checkpointing, chatbot logic
├── frontend.py             # Streamlit UI (chat interface, sidebar, history management)
│
├── data/                   # PDF documents (ML/DS books and guides)
│
├── faiss_index/            # Auto-generated FAISS vector index (saved to disk)
│   └── data_index/
│       ├── index.faiss
│       └── index.pkl
│
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose service configuration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, DB credentials)
└── .gitignore              # Git ignored files
```

---

## ⚙️ How It Works — Step by Step

### Step 1 — Document Loading
All PDF files inside the `data/` folder are loaded using LangChain's `DirectoryLoader` with `PyPDFLoader`. Every page of every PDF is read into memory.

### Step 2 — Text Splitting
The raw text is split into chunks of **800 characters** with an overlap of **150 characters** between consecutive chunks. The overlap ensures no important context is lost at the boundary of two chunks.

### Step 3 — Embedding & Indexing
Each chunk is converted into a **vector (a list of numbers)** using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model. All vectors are stored in a **FAISS index** and saved to disk. On future runs, the existing index is loaded directly — no re-processing needed.

### Step 4 — Question Rewriting
When you ask a follow-up question like *"Can you explain that again?"*, the chatbot first **rewrites it into a standalone question** like *"Can you explain what Gradient Descent is again?"* using the full chat history. This makes the search more accurate.

### Step 5 — Retrieval
The rewritten question is converted to a vector and compared against all stored chunk vectors in FAISS. The **top 5 most similar chunks** are retrieved.

### Step 6 — Answer Generation
The 5 retrieved chunks are passed as context to the Groq LLM, along with your question and chat history. The LLM generates a clean, formatted answer **strictly based on the retrieved context** — no outside knowledge is used.

### Step 7 — Persistent Memory
The full conversation is saved to **PostgreSQL** via LangGraph's `PostgresSaver`. Each conversation has a unique `thread_id`. You can close the app, reopen it, and all past conversations are restored automatically.

---

## 🚀 Getting Started (Local Setup)

### Prerequisites
- Python 3.11+
- PostgreSQL database (local or cloud — e.g. Supabase, Neon, Aiven)
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Docker (optional, for containerized setup)

### 1. Clone the Repository
```bash
git clone https://github.com/bikram2github/11.ML-DS-Rag-App.git
cd "LANGGRAPH RAG"
```

### 2. Create a `.env` File
```env
GROQ_API_KEY=your_groq_api_key_here

PG_DB=your_database_name
PG_USER=your_database_user
PG_PASSWORD=your_database_password
PG_HOST=your_database_host
PG_PORT=5432
```

### 3. Install Dependencies
```bash
# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

### 4. Add Your PDF Documents
Place your PDF files inside the `data/` folder. The app will automatically load and index them on the first run.

### 5. Run the App
```bash
streamlit run frontend.py
```

Open your browser at **http://localhost:8501**

---

## 🐳 Running with Docker

```bash
# Build and start the app
docker-compose up --build

# Run in background
docker-compose up -d

# Stop the app
docker-compose down
```

The app will be available at **http://localhost:8501**

> ⚠️ Make sure your `.env` file is present before running Docker. The `docker-compose.yml` picks it up automatically via `env_file`.

---

## 🔁 CI/CD Pipeline — Jenkins

The `Jenkinsfile` defines a fully automated pipeline with these stages:

| Stage | What It Does |
|---|---|
| **Checkout** | Pulls latest code from GitHub (`main` branch) |
| **Create Virtual Environment** | Sets up a Python venv (skips if already exists) |
| **Verify Python Version** | Confirms the correct Python version is active |
| **Install Dependencies** | Installs PyTorch (CPU) + all `requirements.txt` packages |
| **Run Pytest** | Runs all unit tests — pipeline fails if any test fails |
| **Stop Existing Containers** | Brings down any running Docker containers |
| **Build Docker Image** | Builds a fresh Docker image from the latest code |
| **Start Application** | Starts the app in detached Docker mode |

On **success** → app is live and all tests passed  
On **failure** → pipeline stops and reports the failed stage

---

## 🧪 Running Tests

```bash
pytest -v
```

| Test File | What It Tests |
|---|---|
| `test_config.py` | LLM and embeddings load correctly |
| `test_db.py` | PostgreSQL connection and saving works |
| `test_basics.py` | Core RAG functions (load, split, embed, retrieve) |
| `test_frontend.py` | Streamlit UI behavior |

---

## ✨ Key Features

- ✅ **Answers only from your PDFs** — no hallucinations from outside knowledge
- ✅ **Persistent memory** — chat history saved in PostgreSQL, survives restarts
- ✅ **Multi-conversation support** — start new chats, switch between old ones from sidebar
- ✅ **Smart question rewriting** — handles follow-ups like "explain that again" naturally
- ✅ **FAISS index caching** — index built once and reused on all future runs
- ✅ **Clean formatted answers** — numbered bullet points, grammar correction, proper English
- ✅ **Dockerized** — runs anywhere with a single command
- ✅ **CI/CD ready** — automated testing and deployment via Jenkins

---

## 📄 PDF Documents Included

| Document | Topic |
|---|---|
| 50 Most Important AI ML Questions | AI/ML Concepts & Interview Prep |
| Building Machine Learning Systems with Python | Practical ML Engineering |
| Fresher Data Scientist Interview Guide | Career & Interview Prep |
| Generative AI: A Beginner's Guide | GenAI Concepts |
| NLP with Transformers (Hugging Face) | NLP & Deep Learning |
| Top 80 ML Interview Questions | ML Interview Prep |
| Top 80 Scikit-Learn Interview Questions | Sklearn Interview Prep |
| Top 30 NLP Interview Questions | NLP Interview Prep |

---

## 🔐 Environment Variables Reference

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLM access |
| `PG_DB` | PostgreSQL database name |
| `PG_USER` | PostgreSQL username |
| `PG_PASSWORD` | PostgreSQL password |
| `PG_HOST` | PostgreSQL host (e.g. `localhost` or cloud URL) |
| `PG_PORT` | PostgreSQL port (default: `5432`) |

---

## 👨‍💻 Author

**Bikram Maity**  
GitHub: [@bikram2github](https://github.com/bikram2github)

