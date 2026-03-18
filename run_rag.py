from src.preprocessing import load_data, convert_record_to_text
from src.chunking import chunk_documents
from src.embeddings import generate_embeddings
from src.retrieval import build_faiss_index
from src.rag_pipeline import ask_question
from src.utils import save_embeddings, load_embeddings


print("Loading data...")
records = load_data("data/insurance_claims_records.txt")

print("Converting records to documents...")
documents = convert_record_to_text(records)

print("Chunking documents...")
chunks = chunk_documents(documents)


# Load or generate embeddings
embeddings = load_embeddings()

if embeddings is None:
    print("Generating embeddings (first run)...")
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings)
else:
    print("Loaded cached embeddings.")


# Check consistency
if len(embeddings) != len(chunks):
    print("Embedding mismatch detected. Regenerating...")
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings)


# Build index
index = build_faiss_index(embeddings)

print("RAG system ready!")


while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer = ask_question(question, index, chunks)

    print("\nAnswer:\n")
    print(answer)