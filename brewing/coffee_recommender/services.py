import os
import logging
import traceback
from dotenv import load_dotenv
from django.conf import settings

from utils.data_processing import load_and_preprocess_coffee_data
from utils.vector_store import load_faiss_vector_store
from utils.llama_loader import load_llama_llm
# from utils.coffee_chain import create_coffee_retrieval_qa_chain  # 기존 코드 주석 처리
from utils.direct_rag import DirectRAG  # 새로운 DirectRAG 클래스 임포트
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

# coffee_qa_chain = None  # 기존 코드 주석 처리
direct_rag = None  # 새로운 DirectRAG 인스턴스 저장 변수

def initialize_coffee_chain():
    """
    Initialize the Coffee QA Chain when the server starts.
    """
    # global coffee_qa_chain  # 기존 코드 주석 처리
    global direct_rag  # 새로운 DirectRAG 인스턴스 참조

    try:
        logger.info("Initializing DirectRAG system...")
        
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

        # 기존 체인 생성 코드 주석 처리
        # logger.info("Creating QA chain...")
        # coffee_qa_chain = create_coffee_retrieval_qa_chain(llm, vectorstore)
        # logger.info("Coffee QA chain initialized successfully")
        
        # 새로운 DirectRAG 생성
        logger.info("Creating DirectRAG system...")
        direct_rag = DirectRAG(vectorstore, llm, max_docs=4)
        logger.info("DirectRAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing DirectRAG system: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def recommend_coffee(query: str) -> dict:
    """
    Process user query and return the coffee recommendation.
    
    Args:
        query (str): User's input query.
    
    Returns:
        dict: Recommendation result.
    """
    # global coffee_qa_chain  # 기존 코드 주석 처리
    global direct_rag  # 새로운 DirectRAG 인스턴스 참조

    logger.info(f"Processing query: {query}")

    # if coffee_qa_chain is None:  # 기존 코드 주석 처리
    if direct_rag is None:
        logger.error("DirectRAG system is not initialized")
        raise ValueError("DirectRAG system is not initialized. Call initialize_coffee_chain() first.")

    try:
        # 기존 체인 실행 코드 주석 처리
        # logger.info("Invoking QA chain...")
        # logger.info(f"Input query: {query}")
        # answer = coffee_qa_chain.invoke({"query": query})
        
        # 새로운 DirectRAG 실행
        logger.info("Invoking DirectRAG system...")
        answer = direct_rag.process_query(query)
        logger.info("DirectRAG system responded successfully")
        
        # 원본 응답 전체 로깅 (디버깅용)
        if isinstance(answer, dict) and 'result' in answer:
            # 로그 파일에 전체 원본 응답 기록
            log_dir = os.path.join(settings.BASE_DIR, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "raw_response.log")
            
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n----------\nQuery: {query}\n----------\n")
                log_file.write(f"{answer['result']}\n")
                log_file.write("==========\n")
            
            # 일반 로그에는 일부만 출력
            raw_response = answer['result']
            print("\n======== RAW RESPONSE ========")
            print(f"Query: {query}")
            print("------------------------------")
            print(raw_response)
            print("==============================\n")
            
            logger.info(f"Raw response length: {len(raw_response)}")
            logger.info(f"Raw response preview: {raw_response[:500]}...")
            
            # 디버그 정보가 있으면 로깅
            if '_debug' in answer:
                logger.info(f"Debug info: {answer['_debug']}")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error during DirectRAG invocation: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 응답 형식 유지
        return {
            "answer": {
                "result": f"죄송합니다. 추천 과정에서 오류가 발생했습니다: {str(e)}"
            }
        }
