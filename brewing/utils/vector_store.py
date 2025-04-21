from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os



def load_faiss_vector_store():
    """
    Generate text from the dataset by combining origin, desc_1, roast, and agtron.
    Create embeddings → Build a FAISS vector store.
    """
    cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")

    embedding_model = HuggingFaceEmbeddings(
        model_name=cache_dir+"/all-MiniLM-L6-v2",
        model_kwargs={
            "local_files_only":True,
            },
        )
    
   
    
    # FAISS vector store
    return FAISS.load_local(
        "faiss_store",
        embedding_model,
        allow_dangerous_deserialization=True
        )
