# E:\self\brewWing\brewing\utils\direct_rag.py

import logging
from typing import List, Dict, Any
import time
import traceback
import concurrent.futures

from utils.huggingface_api import HuggingFaceAPI
from utils.coffee_recommendation import create_prompt, extract_answer, DEFAULT_RECOMMENDATIONS
from utils.direct_rag_fallback import process_vector_results

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
                # API 토큰 유효성 확인
                use_api = True
                if hasattr(self.hf_api, 'headers') and 'Authorization' in self.hf_api.headers:
                    api_token = self.hf_api.headers['Authorization'].replace('Bearer ', '')
                    if not api_token or api_token == "YOUR_HF_API_TOKEN":
                        use_api = False
                        logger.warning("Invalid Hugging Face API token. Using direct vector search fallback.")
                else:
                    use_api = False
                    logger.warning("No Hugging Face API headers found. Using direct vector search fallback.")
                
                generation_start = time.time()
                
                if use_api:
                    # API 호출 시도
                    logger.info("Generating answer with Hugging Face API...")
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self.hf_api.generate_text, prompt)
                        logger.info("Waiting for API response...")
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
                    
                    # 응답이 비어있거나 너무 짧으면 대체 방법 사용
                    if not raw_response or len(raw_response.strip()) < 30:
                        logger.warning(f"API response too short or empty: '{raw_response}'")
                        # 벡터 검색 결과 사용
                        raw_response = process_vector_results(retrieved_docs, query)
                        logger.info("Using vector search fallback instead of API response")
                else:
                    # API 토큰이 없으면 벡터 검색 결과를 직접 처리
                    logger.info("Using vector search results directly (no valid API token)")
                    raw_response = process_vector_results(retrieved_docs, query)
                    generation_time = time.time() - generation_start
                    logger.info(f"Direct vector result processing completed in {generation_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(traceback.format_exc())
                # 오류 발생 시 벡터 검색 결과 사용
                try:
                    logger.info("Trying vector search fallback after API error")
                    raw_response = process_vector_results(retrieved_docs, query)
                    logger.info("Using vector search fallback after API error")
                    generation_time = time.time() - generation_start
                except Exception as e2:
                    logger.error(f"Error in fallback processing: {str(e2)}")
                    raw_response = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}\n\n**참고**: 오류가 발생하여 기본 추천을 제공합니다."
                    generation_time = time.time() - generation_start
            logger.debug(f"Final response preview: {raw_response[:200]}...")
            # 4. 답변 추출
            final_answer = extract_answer(raw_response)
            extraction_time = time.time() - start_time - retrieval_time - prompt_time - generation_time
            logger.info(f"Answer extraction completed in {extraction_time:.2f} seconds")
            
            # 5. 결과 반환 - 원본 문서 정보 추가
            total_time = time.time() - start_time
            logger.info(f"Total processing completed in {total_time:.2f} seconds")
            
            # 원본 문서 정보 추가
            original_docs = []
            for doc in retrieved_docs:
                original_docs.append({
                    "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": doc["metadata"]
                })
            
            return {
                "result": final_answer,
                "_debug": {
                    "query": query,
                    "docs_count": len(retrieved_docs),
                    "docs": original_docs,  # 원본 문서 정보 추가
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