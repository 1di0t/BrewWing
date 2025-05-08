# E:\self\brewWing\brewing\utils\direct_rag.py

import logging
from typing import List, Dict, Any
import time
import traceback
import concurrent.futures

from utils.huggingface_api import HuggingFaceAPI
from utils.coffee_recommendation import create_prompt, extract_answer, DEFAULT_RECOMMENDATIONS

# 로깅 설정
logger = logging.getLogger(__name__)

class DirectRAG:
    """
    RAG 시스템에서 문서 검색만 로컬에서 수행하고 
    LLM 추론은 Hugging Face Inference API를 사용하는 하이브리드 접근법
    """
    
    def __init__(self, vectorstore, max_docs=4):
        self.vectorstore = vectorstore  # 기존 FAISS 벡터 스토어
        self.max_docs = max_docs  # 검색할 문서 수
        self.hf_api = HuggingFaceAPI()  # Hugging Face API 인스턴스
        logger.info(f"DirectRAG initialized with max_docs={max_docs} using Hugging Face API")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        쿼리에 관련된 문서를 검색합니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            검색된 문서 리스트
        """
        try:
            logger.info(f"Retrieving documents for query: {query}")
            # 기존 vectorstore의 retriever 활용
            docs = self.vectorstore.similarity_search(query, k=self.max_docs)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # 검색된 문서의 내용 반환
            results = []
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1} preview: {doc.page_content[:100]}...")
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            처리된 결과 딕셔너리
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # 시간 측정 시작
            start_time = time.time()
            
            # 기본 응답 준비
            default_response = {
                "result": f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}",
                "_debug": {
                    "query": query,
                    "error": "Fallback response"
                }
            }
            
            # 1. 관련 문서 검색
            retrieved_docs = self.retrieve(query)
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
            
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return default_response
            
            # 2. 프롬프트 생성
            prompt = create_prompt(query, retrieved_docs)
            prompt_time = time.time() - start_time - retrieval_time
            logger.info(f"Prompt creation completed in {prompt_time:.2f} seconds")
            
            # 3. Hugging Face API를 통한 답변 생성
            try:
                generation_start = time.time()
                logger.info("Generating answer with Hugging Face API...")
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.hf_api.generate_text, prompt)
                    try:
                        # 30초 타임아웃 설정
                        raw_response = future.result(timeout=30)
                        generation_time = time.time() - generation_start
                        logger.info(f"Answer generation completed in {generation_time:.2f} seconds")
                    except concurrent.futures.TimeoutError:
                        logger.warning("API call timed out after 30 seconds")
                        raw_response = ""
                        generation_time = time.time() - generation_start
                        logger.warning(f"Empty response after {generation_time:.2f} seconds timeout")
                
                # 응답이 비어있거나 너무 짧으면 기본 응답 사용
                if not raw_response or len(raw_response.strip()) < 30:
                    logger.warning(f"API response too short or empty: '{raw_response}'")
                    raw_response = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                
            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(traceback.format_exc())
                raw_response = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                generation_time = time.time() - generation_start
            
            # 4. 답변 추출
            final_answer = extract_answer(raw_response)
            extraction_time = time.time() - start_time - retrieval_time - prompt_time - generation_time
            logger.info(f"Answer extraction completed in {extraction_time:.2f} seconds")
            
            # 5. 결과 반환
            total_time = time.time() - start_time
            logger.info(f"Total processing completed in {total_time:.2f} seconds")
            
            return {
                "result": final_answer,
                "_debug": {
                    "query": query,
                    "docs_count": len(retrieved_docs),
                    "prompt_length": len(prompt),
                    "raw_response_length": len(raw_response),
                    "times": {
                        "retrieval": f"{retrieval_time:.2f}s",
                        "prompt": f"{prompt_time:.2f}s",
                        "generation": f"{generation_time:.2f}s",
                        "extraction": f"{extraction_time:.2f}s",
                        "total": f"{total_time:.2f}s"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return default_response