# E:\self\brewWing\brewing\coffee_recommender\services.py

import os
import logging
import traceback
from django.conf import settings

from utils.data_processing import load_and_preprocess_coffee_data
from utils.vector_store import load_faiss_vector_store
from utils.direct_rag import DirectRAG

# 로깅 설정
logger = logging.getLogger(__name__)

direct_rag = None  # DirectRAG 인스턴스 저장 변수
is_initialized = False

def initialize_coffee_chain():
    """
    벡터 스토어 기반 검색 시스템을 초기화합니다.
    지연 초기화 패턴을 사용합니다 - 첫 요청 시에만 초기화됩니다.
    """
    global direct_rag, is_initialized

    try:
        logger.info("Initializing vector search system...")
        
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
        
        # DirectRAG 생성
        logger.info("Creating vector search system...")
        direct_rag = DirectRAG(vectorstore, max_docs=4)
        logger.info("Vector search system initialized successfully")
        is_initialized = True
        
    except Exception as e:
        logger.error(f"Error initializing vector search system: {str(e)}")
        logger.error(traceback.format_exc())
        is_initialized = False

def recommend_coffee(query: str) -> dict:
    """
    사용자 쿼리를 처리하고 커피 추천 결과를 반환합니다.
    
    Args:
        query (str): 사용자 입력 쿼리
    
    Returns:
        dict: 검색 결과
    """
    global direct_rag, is_initialized

    logger.info(f"Processing query: {query}")

    # 초기화되지 않았으면 초기화
    if not is_initialized:
        logger.info("Vector search system not initialized. Initializing now...")
        initialize_coffee_chain()
        
    # 다시 확인
    if direct_rag is None:
        logger.error("Vector search system failed to initialize")
        return {
            "result": "죄송합니다. 시스템 초기화에 실패했습니다. 잠시 후 다시 시도해주세요.",
            "_debug": {
                "error": "Initialization failed"
            }
        }
    
    try:
        # 쿼리 처리
        result = direct_rag.process_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "result": "죄송합니다. 쿼리 처리 중 오류가 발생했습니다.",
            "_debug": {
                "error": str(e)
            }
        }