# Adaptive RAG - Agentic AI Chatbot

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![LangGraph](https://img.shields.io/badge/LangGraph-AgenticAI-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)
![Groq](https://img.shields.io/badge/Groq-LLM-red)

---

## 📋 Overview

Adaptive RAG is an Agentic AI-powered Retrieval-Augmented Generation (RAG) system that intelligently routes user queries through document retrieval, general reasoning, or web search.

The system supports:

- PDF and TXT document uploads
- Semantic document search using FAISS
- Conversational memory using MongoDB
- Groq-powered LLM inference
- LangGraph workflow orchestration
- FastAPI backend
- Streamlit frontend

Users can upload documents and ask natural language questions about their content.

---

## 🚀 Features

### 🧠 Intelligent Query Routing

The application automatically classifies user queries into:

- Index Queries (Document-based)
- General Queries (LLM knowledge)
- Search Queries (Web Search)

---

### 📚 Document Intelligence

- PDF Upload Support
- TXT Upload Support
- Semantic Search
- Chunking & Embeddings
- Context-Aware Responses

---

### 🤖 Agentic AI Workflow

Built using LangGraph:

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

Built using Streamlit:

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
Adaptive-Rag/
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

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Workflow | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Vector Store | FAISS |
| Database | MongoDB |
| Search | Tavily |
| Embeddings | Sentence Transformers |
| Language | Python |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Adaptive-Rag.git

cd Adaptive-Rag
```

### Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
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

---

## 🗄️ Start MongoDB

```bash
brew services start mongodb-community
```

Verify:

```bash
mongosh
```

---

## ▶️ Run Backend

```bash
uvicorn src.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Frontend

```bash
streamlit run streamlit_app/home.py
```

Frontend:

```text
http://localhost:8501
```

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

Add to `.gitignore`:

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

GitHub:
https://github.com/your-github-username

LinkedIn:
https://linkedin.com/in/your-linkedin-profile

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

This project is licensed under the MIT License.