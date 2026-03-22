Insurance RAG Assistant (FastAPI + FAISS)

A Retrieval-Augmented Generation (RAG) system built to answer questions from insurance data using semantic search and LLMs.

Project Overview

This project started as a notebook-based experiment and was upgraded into a FastAPI application, simulating a real-world AI system.

It retrieves relevant data using vector search and generates structured answers using an LLM, with additional post-processing to improve reliability.

System Architecture

User Query
→ Embedding (Sentence Transformers)
→ FAISS Vector Search
→ Context Cleaning & Deduplication
→ LLM Generation (FLAN-T5)
→ Post-processing (Rule-based filtering)
→ Final Structured Answer

⚙️ Key Features
Semantic search using Sentence Transformers
FAISS vector database for fast retrieval
LLM-based response generation
Context cleaning and deduplication
Rule-based post-processing for reliable output
FastAPI backend for real-time interaction
API Usage

Run the server:

uvicorn app.main:app --reload

Open Swagger UI:
http://127.0.0.1:8000/docs

Endpoint

POST /ask

Example Request
{
  "question": "What occupations are associated with higher claim amounts?"
}
Example Response
{
  "answer": "- Occupation: Engineer"
}
⚠️ Limitations
RAG is not suitable for numerical analytics (e.g., averages, max values)
Results depend heavily on data quality
Uses lightweight local models for inference

🧠 Key Learnings
LLM outputs require control and validation
Data formatting significantly impacts RAG performance
RAG is strong for semantic understanding but weak for aggregation
Post-processing improves reliability in real-world systems

🚀 Future Improvements
Docker containerization
Cloud deployment
Monitoring and logging
Evaluation metrics
Hybrid system (RAG + SQL analytics)

 Tech Stack
Python
FastAPI
FAISS
Sentence Transformers
Hugging Face Transformers

Author
Pranav A Kumar
