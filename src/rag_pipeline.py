from sentence_transformers import SentenceTransformer
from src.retrieval import search_index
from transformers import pipeline
import pickle
import faiss
import re

def clean_context(text):
    # remove numbers (age, income, ids)
    text = re.sub(r'\d+\.?\d*', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# DEFINE MODELS HERE (GLOBAL)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# LOAD INDEX + CHUNKS
index = faiss.read_index("data/faiss_index/index.faiss")

with open("data/faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def ask_rag(question: str):

    query_embedding = embedding_model.encode([question]).astype("float32")

    # Step 1: Use top_k = 3
    indices = search_index(index, query_embedding, top_k=3)

    # Safe retrieval
    retrieved_chunks = [
        chunks[i] for i in indices[0]
        if i != -1 and i < len(chunks)
    ]

    # No data
    if len(retrieved_chunks) == 0:
        return {
            "answer": "No relevant data found for this question.",
            "sources": []
        }

    # Light safety check (updated)
    if len(retrieved_chunks) < 2:
        return {
            "answer": "I don't have enough relevant data to answer this question.",
            "sources": []
        }

    # Step 2: Clean + deduplicate
    cleaned_chunks = [clean_context(chunk) for chunk in retrieved_chunks]
    unique_chunks = list(set(cleaned_chunks))

    # Final context (very controlled)
    context = "\n".join([
        f"- {chunk.split('.')[0]}"
        for chunk in unique_chunks[:3]
    ])


    prompt = f"""
You are a data analyst.

Analyze the dataset and summarize patterns among high-income customers.

STRICT RULES:
- DO NOT copy or repeat any sentence from the context
- DO NOT include specific numbers (ages, income values)
- ONLY describe general patterns


Context:
{context}

Question:
{question}

Answer:
""".strip()

    result = generator(
    prompt,
    max_new_tokens=120,
    min_length=40,
    do_sample=False,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    num_return_sequences=1
    )

    # Clean final answer
    answer = result[0]["generated_text"].strip()

    # Remove repeated lines
    lines = list(dict.fromkeys(answer.split("\n")))

    # Keep only non-empty lines
    cleaned_lines = [line for line in lines if line.strip()]

    # Limit to 3 lines (structured output)
    # Extract keywords (simple clean output)
    text = " ".join(cleaned_lines).lower()

    if "engineer" in text:
        occupation = "Engineer"
    elif "doctor" in text:
        occupation = "Doctor"
    elif "teacher" in text:
        occupation = "Teacher"
    else:
        occupation = "Various roles"

    final_answer = f"- Occupation: {occupation}"

    return {
        "answer": final_answer,
        "sources": retrieved_chunks[:3]
    }