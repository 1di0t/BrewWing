from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import os

hugginhgface_token = os.getenv("HUGGINGFACE_API_KEY")

def create_vector_store_from_coffee_df(coffee_df: pd.DataFrame):
    """
    Generate text from the dataset by combining origin, desc_1, roast, and agtron.
    Create embeddings → Build a FAISS vector store.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "local_files_only":True,
            },
        )
    
    texts = []
    metadatas = []
    for _, row in coffee_df.iterrows():
        # text content for vector store
        text_content = (
            f"Origin: {row['origin']}\n"
            f"Roast: {row['roast']}\n"
            f"Agtron: {row['agtron']}\n"
            f"Description1: {row['desc_1']}\n"
        )
        texts.append(text_content)
        
        # metadata
        metadatas.append({
            "origin": row["origin"],
            "roast": row["roast"],
            "agtron": row["agtron"],
            "desc_1": row["desc_1"],
        })
    
    # FAISS vector store
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    return vectorstore
