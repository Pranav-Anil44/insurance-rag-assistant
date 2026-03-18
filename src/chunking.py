def chunk_documents(documents, chunk_size=200, overlap=50):
    """
    Split documents into smaller chunks for embedding
    """

    chunks = []

    for doc in documents:
        start = 0
        
        while start < len(doc):
            
            chunk = doc[start:start + chunk_size]
            chunks.append(chunk)
            start += chunk_size - overlap

    return chunks

