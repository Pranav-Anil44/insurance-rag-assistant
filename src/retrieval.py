import faiss
import numpy as np
import os
import pickle

def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def save_faiss_index(index, chunks):
    """
    Save FAISS index and chunks
    """

    # Create foldeR
    os.makedirs("data/faiss_index", exist_ok=True)

    # Save index
    faiss.write_index(index, "data/faiss_index/index.faiss")

    # Save chunks
    with open("data/faiss_index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def search_index(index, query_embedding, top_k=3):
    """
    Search the FAISS index for similar embeddings
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices