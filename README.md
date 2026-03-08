# insurance-rag-assistant


This project implements a **Retrieval-Augmented Generation (RAG)** system to answer insurance-related questions using claim records as the knowledge base.

The system retrieves relevant information using **vector similarity search (FAISS)** and generates answers using a **Large Language Model (LLM)**.

# Project Objective

The goal of this project is to demonstrate how **RAG pipelines can combine document retrieval with language models** to produce accurate and context-aware responses.
This approach is widely used in **enterprise AI assistants and knowledge retrieval systems**.

---

# RAG Workflow

Insurance Claim Records  
↓  
Convert Structured Data to Text  
↓  
Chunk Documents  
↓  
Generate Embeddings  
↓  
Store Embeddings in FAISS Vector Index  
↓  
Retrieve Relevant Context  
↓  
LLM Generates Final Answer  

---

# Project Structure

### Data Preparation

`row_to_text.ipynb`

Converts structured insurance claim records into text format suitable for NLP processing.

---

### Document Chunking

`chunk_record.ipynb`

Splits long documents into smaller chunks to improve semantic search performance.

---

### Embedding Generation

`generate_embeddings.ipynb`

Generates vector embeddings using **Sentence Transformers**.

---

### Vector Search with FAISS

`FAISS_index_and_retreival.ipynb`

Creates a FAISS vector index and retrieves the most relevant document chunks for a given query.

---

### RAG Pipeline

`rag_with_llm.ipynb`

Implements the final **Retrieval-Augmented Generation pipeline** combining vector retrieval with an LLM to answer questions.

---

# Dataset

`insurance_claims_records.txt`

Sample insurance claim records used as the knowledge base for the RAG system.

---

# Stored Embeddings

`insurance_embedding.npy`

Precomputed embeddings used to build the FAISS vector index.

---

# Technologies Used

Python  
FAISS (Facebook AI Similarity Search)  
Sentence Transformers  
Transformers  
NumPy  
Jupyter Notebook  

---

# Example Query

Example question that can be asked to the system:
"What are the most common reasons for insurance claims?"


The system retrieves the most relevant claim records and generates a contextual response.

---

# Future Improvements

Convert notebook pipeline into reusable Python modules
Add FastAPI backend for real-time question answering
Add Streamlit UI for interactive queries
Deploy the RAG system as a web service

---

# Author
Pranav Anil  
AI / Machine Learning Enthusiast  
Transitioning from Mainframe Production Support to AI Engineering
Pranav Anil  
AI / Machine Learning Enthusiast  
Transitioning from Mainframe Production Support to AI Engineering
