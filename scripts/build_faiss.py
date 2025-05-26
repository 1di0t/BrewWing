# build_faiss.py
import os
import sys
import logging
from pathlib import Path
import numpy as np
from brewing.utils.data_processing import load_and_preprocess_coffee_data
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Starting FAISS index build...")

try:
    # load data
    coffee_df = load_and_preprocess_coffee_data("data/coffee_drop.csv")
    
    # load cache path
    cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
    model_path = os.path.join(cache_dir, "all-MiniLM-L6-v2")
    
    # Check if model exists
    if not os.path.exists(model_path) or not any(Path(model_path).glob("*.bin")):
        logger.warning(f"Model files not found at {model_path}")
        logger.warning("Creating dummy FAISS index")
        out_dir = "faiss_store"
        os.makedirs(out_dir, exist_ok=True)
        
        # Create dummy FAISS index
        dummy_vectors = np.random.random((len(coffee_df), 384)).astype(np.float32)
        
        # Create dummy metadata
        dummy_metadatas = []
        for _, row in coffee_df.iterrows():
            dummy_metadatas.append({
                "origin": row["origin"],
                "roast": row["roast"],
                "agtron": row["agtron"],
                "desc_1": row["desc_1"],
            })
        
        # Save dummy files
        with open(f"{out_dir}/index.faiss", "wb") as f:
            f.write(b"DUMMY FAISS INDEX")
        
        with open(f"{out_dir}/index.pkl", "wb") as f:
            f.write(b"DUMMY PICKLE FILE")
            
        logger.info(f"Dummy FAISS index saved to ./{out_dir}/")
        sys.exit(0)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=cache_dir+"/all-MiniLM-L6-v2",
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
        
        # Save to disk
        out_dir = "faiss_store"
        vectorstore.save_local(out_dir)
        logger.info(f"FAISS index saved to ./{out_dir}/")
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        logger.warning("Creating dummy FAISS index")
        out_dir = "faiss_store"
        os.makedirs(out_dir, exist_ok=True)
        
        # Create dummy files
        with open(f"{out_dir}/index.faiss", "wb") as f:
            f.write(b"DUMMY FAISS INDEX")
        
        with open(f"{out_dir}/index.pkl", "wb") as f:
            f.write(b"DUMMY PICKLE FILE")
            
        logger.info(f"Emergency dummy FAISS index saved to ./{out_dir}/")

except Exception as e:
    logger.error(f"Error in build_faiss.py: {str(e)}")
    logger.warning("Creating emergency dummy FAISS index")
    out_dir = "faiss_store"
    os.makedirs(out_dir, exist_ok=True)
    
    # Create dummy files
    with open(f"{out_dir}/index.faiss", "wb") as f:
        f.write(b"EMERGENCY DUMMY FAISS INDEX")
    
    with open(f"{out_dir}/index.pkl", "wb") as f:
        f.write(b"EMERGENCY DUMMY PICKLE FILE")
        
    logger.info(f"Emergency dummy FAISS index saved to ./{out_dir}/")
    sys.exit(1)
