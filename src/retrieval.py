import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings
    """

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_index(index, query_embedding, top_k=3):
    """
    Search the FAISS index for similar embeddings
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices