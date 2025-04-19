# build_faiss.py
import os
from brewing.utils.data_processing import load_and_preprocess_coffee_data
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load data
coffee_df = load_and_preprocess_coffee_data("data/coffee_drop.csv")

# load cache path
cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")

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

# 5) 디스크에 저장 (폴더 하나로)
out_dir = "faiss_store"
vectorstore.save_local(out_dir)
print(f"FAISS index saved to ./{out_dir}/")
