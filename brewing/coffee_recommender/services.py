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

def recommend_coffee(query: str) -> dict:
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
        logger.info(f"Input query: {query}")
        
        # 원본 응답 저장
        try:
            answer = coffee_qa_chain.invoke({"query": query})
            logger.info("QA chain responded successfully")
            
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
                
                # 응답 구조 분석
                line_count = raw_response.count('\n')
                dash_count = raw_response.count('-')
                bean_count = raw_response.lower().count('coffee bean')
                logger.info(f"Response structure - Lines: {line_count}, Dashes: {dash_count}, 'coffee bean' mentions: {bean_count}")
            else:
                logger.warning(f"Unexpected answer format: {type(answer)}")
        except Exception as e:
            logger.error(f"Error invoking QA chain: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # 응답 후처리
        logger.info("Processing answer...")
        logger.info(f"Raw answer type: {type(answer)}")
        
        # 응답 형식 확인
        if isinstance(answer, dict) and 'result' in answer:
            try:
                # 질문 및 응답 기록
                logger.info(f"Query: {query}")
                logger.info(f"Raw answer preview: {answer['result'][:100]}...")
                
                # 추천 텍스트 추출
                extracted_text = extract_origin_text(answer['result'])
                if extracted_text:
                    logger.info(f"Extracted text preview: {extracted_text[:100]}...")
                    logger.info(f"Extracted text length: {len(extracted_text)}")
                    
                    # 번역 전 추출된 텍스트 출력
                    print("\n======== EXTRACTED TEXT ========")
                    print(extracted_text[:1000])
                    if len(extracted_text) > 1000:
                        print("... (truncated)")
                    print("===============================\n")
                    
                    answer['result'] = extracted_text
                else:
                    logger.warning("Text extraction returned empty string. Using original result.")
                
                # 번역 시도
                try:
                    translated = translate_with_linebreaks(answer['result'])
                    logger.info(f"Translation successful, length: {len(translated)}")
                    logger.info(f"Translated preview: {translated[:100]}...")
                    
                    # 번역된 최종 결과도 출력
                    print("\n======== FINAL TRANSLATED RESULT ========")
                    print(translated[:1000])
                    if len(translated) > 1000:
                        print("... (truncated)")
                    print("======================================\n")
                    
                    answer['result'] = translated
                except Exception as translate_error:
                    logger.error(f"Translation error: {str(translate_error)}")
                    logger.error("Keeping original text")
                
                logger.info("Answer processing completed")
            except Exception as process_error:
                logger.error(f"Error in answer processing: {str(process_error)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"Unexpected answer format: {type(answer)}")
            answer = {"result": str(answer)}
        
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
