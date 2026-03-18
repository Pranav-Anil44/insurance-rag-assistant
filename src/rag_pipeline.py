from sentence_transformers import SentenceTransformer
from src.retrieval import search_index
from transformers import pipeline


# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM text generation pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")


def ask_question(question, index, chunks):
    """
    Run the full RAG pipeline and generate an answer
    """

    # Convert question to embedding
    query_embedding = embedding_model.encode([question])

    # Search vector database
    indices = search_index(index, query_embedding)

    # Retrieve relevant chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Combine retrieved chunks into context
    context = " ".join(retrieved_chunks)

    # Build prompt for LLM
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response
    result = generator(prompt, max_length=200, num_return_sequences=1)

    answer = result[0]["generated_text"]

    return answer