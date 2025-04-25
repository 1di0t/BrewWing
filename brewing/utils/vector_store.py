from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging
import traceback

# 로깅 설정
logger = logging.getLogger(__name__)

def load_faiss_vector_store():
    """
    Generate text from the dataset by combining origin, desc_1, roast, and agtron.
    Create embeddings → Build a FAISS vector store.
    """
    try:
        cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
        logger.info(f"Using cache directory: {cache_dir}")
        
        # 임베딩 모델 로드
        logger.info(f"Loading embedding model from {cache_dir}/all-MiniLM-L6-v2")
        
        # 모델 디렉토리 확인
        model_path = f"{cache_dir}/all-MiniLM-L6-v2"
        if os.path.exists(model_path):
            logger.info(f"Model directory exists at {model_path}")
            logger.info(f"Model directory contents: {os.listdir(model_path)}")
        else:
            logger.error(f"Model directory does not exist at {model_path}")
            # 대체 경로 시도
            alternative_paths = [
                "/app/huggingface_cache/all-MiniLM-L6-v2",
                "./huggingface_cache/all-MiniLM-L6-v2",
                "all-MiniLM-L6-v2"
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found model at alternative path: {alt_path}")
                    model_path = alt_path
                    break
            else:
                logger.error("Model not found in any location")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                "local_files_only":True,
            },
        )
        logger.info("Embedding model loaded successfully")
        
        # FAISS 벡터 스토어 로드
        possible_paths = [
            "faiss_store",
            "/app/brewing/faiss_store",
            "/app/faiss_store",
            "./faiss_store",
            "../faiss_store"
        ]
        
        faiss_store = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"Attempting to load FAISS index from {path}")
                    logger.info(f"Path contents: {os.listdir(path)}")
                    faiss_store = FAISS.load_local(
                        path,
                        embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"FAISS index loaded successfully from {path}")
                    break
            except Exception as e:
                logger.error(f"Failed to load FAISS from {path}: {str(e)}")
                continue
        
        if faiss_store is None:
            logger.error("Failed to load FAISS index from any path")
            raise FileNotFoundError("FAISS index not found in any of the expected locations")
        
        return faiss_store
    except Exception as e:
        logger.error(f"Error in load_faiss_vector_store: {str(e)}")
        logger.error(traceback.format_exc())
        raise
