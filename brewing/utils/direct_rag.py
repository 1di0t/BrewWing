# E:\self\brewWing\brewing\utils\direct_rag.py

import logging
from typing import List, Dict, Any
import time
import traceback

# 로깅 설정
logger = logging.getLogger(__name__)

class DirectRAG:
    """
    벡터 스토어를 사용한 문서 검색 시스템
    """
    
    def __init__(self, vectorstore, max_docs=4):
        self.vectorstore = vectorstore  # FAISS 벡터 스토어
        self.max_docs = max_docs  # 검색할 문서 수
        logger.info(f"DirectRAG initialized with max_docs={max_docs}")
    
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
        벡터 스토어 검색 결과만 반환합니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            검색된 문서 정보를 포함한 결과 딕셔너리
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
                docs_info.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"]
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