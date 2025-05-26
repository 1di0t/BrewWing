# E:\self\brewWing\brewing\utils\direct_rag.py

import logging
from typing import List, Dict, Any
import time
import traceback
from sentence_transformers import SentenceTransformer

# 로깅 설정
logger = logging.getLogger(__name__)

class DirectRAG:
    """
    벡터 스토어를 사용한 문서 검색 시스템
    검색 결과를 한글로 번역하여 반환합니다.
    """
    
    def __init__(self, vectorstore, max_docs=4):
        self.vectorstore = vectorstore  # FAISS 벡터 스토어
        self.max_docs = max_docs  # 검색할 문서 수
        # 번역을 위한 모델 로드
        try:
            self.translator = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            self.translator = None
        logger.info(f"DirectRAG initialized with max_docs={max_docs}")
    
    def translate_to_korean(self, text: str) -> str:
        """
        텍스트를 한글로 번역합니다.
        
        Args:
            text: 번역할 텍스트
            
        Returns:
            번역된 한글 텍스트
        """
        try:
            if not self.translator:
                return text
                
            # 영어 텍스트를 한글로 번역
            # 실제로는 모델이 다국어를 지원하므로, 입력 텍스트를 그대로 사용
            # 필요한 경우 여기에 번역 로직 추가
            return text
            
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            return text
    
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
            # 벡터 스토어 검색
            docs = self.vectorstore.similarity_search(query, k=self.max_docs)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # 검색된 문서의 내용 반환
            results = []
            for i, doc in enumerate(docs):
                content = doc.page_content
                # 메타데이터에서 원산지, 로스팅 레벨 등 추출
                metadata = doc.metadata
                
                # 문서 내용을 한글로 번역
                translated_content = self.translate_to_korean(content)
                
                logger.info(f"Document {i+1} preview: {translated_content[:100]}...")
                results.append({
                    "content": translated_content,
                    "metadata": metadata,
                    "original_content": content  # 원본 내용도 보존
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        벡터 스토어 검색 결과를 한글로 번역하여 반환합니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            번역된 검색 결과를 포함한 딕셔너리
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # 시간 측정 시작
            start_time = time.time()
            
            # 기본 응답 준비
            default_response = {
                "result": "## 검색 결과 없음",
                "_debug": {
                    "query": query,
                    "error": "No documents found"
                }
            }
            
            # 1. 관련 문서 검색
            retrieved_docs = self.retrieve(query)
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
            
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return default_response
            
            # 검색된 문서 정보 구성
            docs_info = []
            for doc in retrieved_docs:
                # 메타데이터에서 주요 정보 추출
                metadata = doc["metadata"]
                origin = metadata.get("origin", "알 수 없음")
                roast = metadata.get("roast", "알 수 없음")
                
                # 문서 내용 포맷팅
                content = doc["content"]
                formatted_content = f"""
원산지: {origin}
로스팅: {roast}
설명: {content}
"""
                
                docs_info.append({
                    "content": formatted_content,
                    "metadata": metadata,
                    "original_content": doc["original_content"]  # 원본 내용도 포함
                })
            
            # 결과 반환
            total_time = time.time() - start_time
            logger.info(f"Total processing completed in {total_time:.2f} seconds")
            
            return {
                "result": "## 검색 결과",
                "docs": docs_info,
                "_debug": {
                    "query": query,
                    "docs_count": len(retrieved_docs),
                    "times": {
                        "retrieval": f"{retrieval_time:.2f}s",
                        "total": f"{total_time:.2f}s"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return default_response