# Adaptive RAG - Agentic AI Chatbot

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-AgenticAI-orange)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLM-red)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

🔗 **Repository:** [github.com/Tejaswi75/Adaptive-RAG](https://github.com/Tejaswi75/Adaptive-RAG)

---

## 📋 Overview

Adaptive RAG is an Agentic AI-powered [Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) system that intelligently routes user queries through document retrieval, general reasoning, or web search.

The system supports:

- PDF and TXT document uploads
- Semantic document search using [FAISS](https://github.com/facebookresearch/faiss)
- Conversational memory using [MongoDB](https://www.mongodb.com)
- [Groq](https://groq.com)-powered LLM inference
- [LangGraph](https://langchain-ai.github.io/langgraph/) workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com) backend
- [Streamlit](https://streamlit.io) frontend

Users can upload documents and ask natural language questions about their content.

---

## 🚀 Features

### 🧠 Intelligent Query Routing

The application automatically classifies user queries into:

- Index Queries (Document-based)
- General Queries (LLM knowledge)
- Search Queries (Web Search via [Tavily](https://tavily.com))

---

### 📚 Document Intelligence

- PDF Upload Support
- TXT Upload Support
- Semantic Search
- Chunking & Embeddings ([Sentence Transformers](https://www.sbert.net))
- Context-Aware Responses

---

### 🤖 Agentic AI Workflow

Built using [LangGraph](https://langchain-ai.github.io/langgraph/):

- Query Analysis
- Retrieval
- Relevance Grading
- Query Rewriting
- Response Generation
- Web Search Fallback

---

### 💾 Memory Management

- MongoDB Chat History
- Session-Based Conversations
- Persistent User Context

---

### 🎨 User Interface

Built using [Streamlit](https://streamlit.io):

- Chat Interface
- Document Upload
- Session Management
- Real-Time Responses

---

## 🏗️ Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI Backend
 │
 ▼
LangGraph Workflow
 │
 ├── Query Classifier
 ├── Retriever
 ├── Relevance Grader
 ├── Query Rewriter
 ├── Generator
 └── Web Search
 │
 ▼
Groq LLM
 │
 ▼
Response
```

---

## 📂 Project Structure

```text
Adaptive-RAG/
│
├── src/
│   ├── api/
│   ├── config/
│   ├── core/
│   ├── db/
│   ├── llms/
│   ├── memory/
│   ├── models/
│   ├── rag/
│   └── tools/
│
├── streamlit_app/
│   ├── pages/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .env
```

Quick links: [`src/`](src) · [`streamlit_app/`](streamlit_app) · [`requirements.txt`](requirements.txt)

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | [Streamlit](https://streamlit.io) |
| Workflow | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | [Groq](https://console.groq.com/docs/models) (Llama 3.3 70B) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Database | [MongoDB](https://www.mongodb.com) |
| Search | [Tavily](https://tavily.com) |
| Embeddings | [Sentence Transformers](https://www.sbert.net) |
| Language | [Python](https://www.python.org) |

---

## ⚙️ Installation

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [MongoDB Community Edition](https://www.mongodb.com/docs/manual/installation/)
- [Groq API key](https://console.groq.com/keys)
- [Tavily API key](https://app.tavily.com)

### Clone Repository

```bash
git clone https://github.com/Tejaswi75/Adaptive-RAG.git

cd Adaptive-RAG
```

### Create Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key

MONGO_URI=mongodb://localhost:27017

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
| `MONGO_URI` | Local MongoDB, or [MongoDB Atlas](https://www.mongodb.com/atlas) |

---

## 🗄️ Start MongoDB

**macOS** ([Homebrew](https://brew.sh)):

```bash
brew services start mongodb-community
```

**Windows:** MongoDB runs as a Windows service after [installation](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/). Start it with:

```powershell
net start MongoDB
```

**Linux:**

```bash
sudo systemctl start mongod
```

Verify with [mongosh](https://www.mongodb.com/docs/mongodb-shell/install/):

```bash
mongosh
```

---

## ▶️ Run Backend

```bash
uvicorn src.main:app --reload
```

- Backend: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ▶️ Run Frontend

```bash
streamlit run streamlit_app/home.py
```

- Frontend: [http://localhost:8501](http://localhost:8501)

---

## 📤 Upload Documents

Supported formats:

- PDF
- TXT

After upload, ask questions like:

```text
What is my name?

What is my education?

What skills do I have?

Tell me about my projects.

Summarize my resume.
```

---

## 🔌 API Endpoints

Full interactive docs are available at [/docs](http://127.0.0.1:8000/docs) once the backend is running.

### Query Endpoint

```http
POST /rag/query
```

Request:

```json
{
  "query": "What skills do I have?",
  "session_id": "user123"
}
```

---

### Upload Document

```http
POST /rag/documents/upload
```

Headers:

```http
X-Description: Resume of Tejaswi Sonal
```

Form Data:

```text
file = resume.pdf
```

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/rag/documents/upload \
  -H "X-Description: Resume of Tejaswi Sonal" \
  -F "file=@resume.pdf"
```

---

## 📈 Project Status

✅ Agentic RAG Workflow

✅ FastAPI Backend

✅ Streamlit Frontend

✅ MongoDB Chat Memory

✅ FAISS Semantic Search

✅ Groq Integration

✅ Document Upload

🚀 Deployment Ready

---

## 🔒 Security

Never push:

```text
.env
venv/
__pycache__/
description.txt
```

Add to [`.gitignore`](.gitignore):

```gitignore
venv/
.env
__pycache__/
*.pyc
.DS_Store
description.txt
```

---

## 👨‍💻 Author

### Tejaswi Sonal

B.Tech Computer Science

Passionate about:

- AI Engineering
- Agentic AI
- RAG Systems
- Full Stack Development
- Backend Engineering

- GitHub: [github.com/tejaswi75](https://github.com/tejaswi75)
- LinkedIn: [linkedin.com/in/tejaswisonal](https://linkedin.com/in/tejaswisonal)

---

## ⭐ Future Improvements

- Multi-document Retrieval
- Hybrid Search
- Authentication
- Docker Deployment
- Cloud Hosting
- Advanced Memory Management
- Evaluation Dashboard

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).