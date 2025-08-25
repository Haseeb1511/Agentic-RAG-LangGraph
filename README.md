# 📚 Agentic RAG with LangGraph

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph** and **LangChain**, featuring:

* Persistent memory with SQLite checkpointer
* Modular, node-based design
* Google Generative AI embeddings
* FastAPI backend + Streamlit frontend
* Docker Compose for deployment

---

## 🎨 UI Preview

![Streamlit UI](ui_images/6.png)

## 📊 Workflow Diagram

![Graph Workflow](graph.png)

---

## 🚀 Major Features

1. **Chat persistence** → via LangGraph SQLite checkpointer
2. **Memory support** (recall past chats)
3. **PDF upload & Q\&A**
4. **Predefined vector stores** (Dermatology, Psychology, Legal, etc.)
5. **Agentic system** with modular nodes
6. **FastAPI + Streamlit integration**
7. **Docker Compose** for easy deployment

---

## ✅ Development Checklist

* [x] Agentic vector store
* [x] Memory
* [x] Node-based OOP design
* [x] Model OOP (Configurable)
* [x] Docker
* [x] FastAPI backend
* [ ] Streaming responses
* [ ] User-selectable models
* [ ] Logging & exception handling

📌 *SQLite checkpointer = your diary → keeps track of all past conversations.*

---

## 📂 Project Structure

```bash
backend/app.py                 # FastAPI backend
frontend/streamlit             # Streamlit UI
src/
 ├── agent/
 │    ├── model_loader.py      # Model loading utilities
 │    ├── agentic_workflow.py  # Graph builder logic
 ├── all_nodes/nodes.py        # LangGraph nodes (loader, splitter, retriever, agent)
 ├── exception/                # Custom error handling
 ├── logger/                   # Logging utilities
 ├── prompt_library/           # Prompts for the agent
requirements.txt               # Dependencies
chat_hist/chat.db              # SQLite checkpoint store
graph.png                      # Workflow diagram
notebook/
 ├── notebook_with_conversationbuffermemory.ipynb
 ├── notebook_with_createvectorstore.ipynb
 ├── notebook_with_load_vectorstore.ipynb
 └── notebook_with_streaming.ipynb
```

---

## ⚡ Installation

```bash
# Clone repo
git clone https://github.com/Haseeb1511/Agentic-RAG-LangGraph.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1. Start Backend (FastAPI)

```bash
uvicorn backend.app:app --reload
```

### 2. Start Frontend (Streamlit)

```bash
streamlit run streamlit_app.py
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

* Streamlit: [http://localhost:8501](http://localhost:8501)
* FastAPI: [http://localhost:8000](http://localhost:8000)

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

