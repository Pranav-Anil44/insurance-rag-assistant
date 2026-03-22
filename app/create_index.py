from src.chunking import chunk_documents
from src.embeddings import generate_embeddings
from src.retrieval import build_faiss_index, save_faiss_index

# Step 1: Load your real data
with open("data/insurance_claims_records.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Convert to documents (list)
documents = [text]

# Step 3: Chunk the documents
chunks = chunk_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Step 4: Generate embeddings
embeddings = generate_embeddings(chunks)

# Convert to float32 (VERY IMPORTANT for FAISS)
embeddings = embeddings.astype("float32")

# Step 5: Build FAISS index
index = build_faiss_index(embeddings)

# Step 6: Save index + chunks
save_faiss_index(index, chunks)

print("✅ REAL FAISS index created successfully!")