import os
#This module helps you interact with the file system.
import pickle
#pickle is used to serialize and deserialize Python objects.

 def save_embeddings(embeddings,file_path="embeddings.pkl"):
     with open(file_path, "wb") as f:
         pickle.dump(embeddings,f)

 def load_embeddings(file_path="embeddings.pkl"):
    """
    Load embeddings if they exist
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None
