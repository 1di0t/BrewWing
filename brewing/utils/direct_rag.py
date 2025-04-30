import logging
import numpy as np
import torch
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DirectRAG:
    """
    LangChain을 사용하지 않고 직접 구현한 RAG 시스템
    """
    
    def __init__(self, vectorstore, llm, max_docs=4):
        self.vectorstore = vectorstore  # 기존 FAISS 벡터 스토어
        self.llm = llm  # 기존 LLM (HuggingFacePipeline)
        self.max_docs = max_docs  # 검색할 문서 수
        logger.info(f"DirectRAG initialized with max_docs={max_docs}")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        쿼리에 관련된 문서를 검색합니다.
        """
        try:
            logger.info(f"Retrieving documents for query: {query}")
            # 기존 vectorstore의 retriever 활용
            docs = self.vectorstore.similarity_search(query, k=self.max_docs)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # 검색된 문서의 내용 반환
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
    
    def create_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        검색된 문서와 쿼리를 바탕으로 프롬프트를 생성합니다.
        """
        # 문서 내용을 하나의 문자열로 결합
        context = ""
        for doc in docs:
            context += doc["content"] + "\n\n"
        
        # 간결하고 명확한 프롬프트 생성
        prompt = f"""아래 커피 정보를 바탕으로 질문에 답해주세요:

{context}

질문: {query}

답변:
## 커피 추천

1. **[원산지] [커피이름]**
   - **맛 프로필**: 
   - **로스팅**: 
   - **특징**: 

2. **[원산지] [커피이름]**
   - **맛 프로필**: 
   - **로스팅**: 
   - **특징**: 

3. **[원산지] [커피이름]**
   - **맛 프로필**: 
   - **로스팅**: 
   - **특징**: 
"""
        
        logger.info(f"Created prompt with length: {len(prompt)}")
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        LLM을 사용하여 답변을 생성합니다.
        """
        try:
            logger.info("Generating answer...")
            # LLM이 HuggingFacePipeline 타입인 경우
            if hasattr(self.llm, 'pipeline'):
                # 직접 파이프라인 사용
                raw_response = self.llm.pipeline(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    return_full_text=False  # 프롬프트 반복 방지
                )
                
                # 결과 추출
                if isinstance(raw_response, list) and len(raw_response) > 0:
                    if "generated_text" in raw_response[0]:
                        return raw_response[0]["generated_text"]
                    else:
                        return str(raw_response[0])
                else:
                    return str(raw_response)
            else:
                # 일반 LangChain LLM 인터페이스 사용
                response = self.llm.invoke(prompt)
                return response
                
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def extract_answer(self, full_response: str) -> str:
        """
        생성된 전체 응답에서 실제 답변 부분만 추출합니다.
        """
        logger.info("Extracting answer from full response")
        
        # 답변 시작 부분 찾기
        start_markers = [
            "## 커피 추천",
            "커피 추천",
            "1. **[",
            "1.**"
        ]
        
        for marker in start_markers:
            if marker in full_response:
                # 마커부터 끝까지 추출
                answer = full_response[full_response.find(marker):]
                logger.info(f"Found answer starting with marker: {marker}")
                return answer
        
        # 마커를 찾지 못한 경우 전체 응답 반환
        logger.warning("Could not find answer markers, returning full response")
        return full_response
    
    def process_query(self, query: str) -> Dict[str, str]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # 1. 관련 문서 검색
            retrieved_docs = self.retrieve(query)
            if not retrieved_docs:
                return {"result": "관련 커피 정보를 찾을 수 없습니다."}
            
            # 2. 프롬프트 생성
            prompt = self.create_prompt(query, retrieved_docs)
            
            # 3. 답변 생성
            raw_response = self.generate_answer(prompt)
            logger.info(f"Raw response length: {len(raw_response)}")
            
            # 4. 답변 추출
            final_answer = self.extract_answer(raw_response)
            
            # 5. 결과 반환
            return {
                "result": final_answer,
                "_debug": {
                    "query": query,
                    "docs_count": len(retrieved_docs),
                    "prompt_length": len(prompt),
                    "raw_response_length": len(raw_response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"result": f"처리 중 오류가 발생했습니다: {str(e)}"}
