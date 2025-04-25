import os
import logging
import traceback
from dotenv import load_dotenv
from django.conf import settings

from utils.data_processing import load_and_preprocess_coffee_data
from utils.vector_store import load_faiss_vector_store
from utils.llama_loader import load_llama_llm
from utils.coffee_chain import create_coffee_retrieval_qa_chain
from utils.text import extract_origin_text, translate_with_linebreaks

import numpy as np
np.zeros(1) 

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수 로드 (Hugging Face API 키 등)
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")

logger.info(f"Hugging Face token exists: {huggingface_token is not None}")
logger.info(f"Using cache directory: {cache_dir}")

coffee_qa_chain = None

def initialize_coffee_chain():
    """
    Initialize the Coffee QA Chain when the server starts.
    """
    global coffee_qa_chain

    try:
        logger.info("Initializing coffee QA chain...")
        
        DATA_FILE_PATH = os.path.join(settings.BASE_DIR, 'data', 'coffee_drop.csv')
        logger.info(f"Loading data from: {DATA_FILE_PATH}")
        
        # 데이터 파일 존재 여부 확인
        if os.path.exists(DATA_FILE_PATH):
            logger.info(f"Data file exists at {DATA_FILE_PATH}")
        else:
            logger.error(f"Data file does not exist at {DATA_FILE_PATH}")
            # 현재 디렉토리 목록 확인
            try:
                base_dir_contents = os.listdir(settings.BASE_DIR)
                logger.info(f"Base directory contents: {base_dir_contents}")
                
                if 'data' in base_dir_contents:
                    data_dir_contents = os.listdir(os.path.join(settings.BASE_DIR, 'data'))
                    logger.info(f"Data directory contents: {data_dir_contents}")
            except Exception as e:
                logger.error(f"Failed to list directory contents: {str(e)}")

        # 데이터 전처리
        logger.info("Preprocessing coffee data...")
        coffee_df = load_and_preprocess_coffee_data(DATA_FILE_PATH)
        logger.info(f"Data loaded and preprocessed: {len(coffee_df)} records")

        # 벡터 스토어 생성
        logger.info("Loading FAISS vector store...")
        vectorstore = load_faiss_vector_store()
        logger.info("FAISS vector store loaded successfully")

        # LLM 로드 (Hugging Face 모델)
        model_path = os.path.join(cache_dir, "Llama-3.2-1B")
        logger.info(f"Loading LLM from: {model_path}")
        llm = load_llama_llm(model_path, token=huggingface_token)
        logger.info("LLM loaded successfully")

        # 체인 생성
        logger.info("Creating QA chain...")
        coffee_qa_chain = create_coffee_retrieval_qa_chain(llm, vectorstore)
        logger.info("Coffee QA chain initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing coffee QA chain: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def recommend_coffee(query: str) -> dict:
    """
    Process user query and return the coffee recommendation.
    
    Args:
        query (str): User's input query.
    
    Returns:
        dict: Recommendation result.
    """
    global coffee_qa_chain

    logger.info(f"Processing query: {query}")

    if coffee_qa_chain is None:
        logger.error("Coffee QA Chain is not initialized")
        raise ValueError("Coffee QA Chain is not initialized. Call initialize_coffee_chain() first.")

    try:
        # 체인 실행 (질문 처리)
        logger.info("Invoking QA chain...")
        answer = await coffee_qa_chain.invoke({"query": query})
        logger.info("QA chain responded successfully")
        
        # 응답 후처리
        logger.info("Processing answer...")
        answer['result'] = await extract_origin_text(answer['result'])
        answer['result'] = await translate_with_linebreaks(answer['result'])
        logger.info("Answer processed successfully")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error during chain invocation: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 응답 형식 유지
        return {
            "answer": {
                "result": f"죄송합니다. 추천 과정에서 오류가 발생했습니다: {str(e)}"
            }
        }
