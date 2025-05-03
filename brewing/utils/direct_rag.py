import logging
import numpy as np
from typing import List, Dict, Any
import time
import traceback

logger = logging.getLogger(__name__)

# 기본 커피 추천 템플릿
DEFAULT_RECOMMENDATIONS = """
1. **[케냐] 키암부**
   - **맛 프로필**: 강한 산미, 시트러스와 베리류 노트
   - **로스팅**: 라이트-미디엄
   - **특징**: 상큼한 과일향과 선명한 산미

2. **[에티오피아] 예가체프**
   - **맛 프로필**: 화사한 산미, 꽃과 베리 향미
   - **로스팅**: 라이트
   - **특징**: 복합적인 향과 상쾌한 산미

3. **[르완다] 키부**
   - **맛 프로필**: 신선한 산미, 레드베리 향
   - **로스팅**: 라이트-미디엄
   - **특징**: 달콤한 단맛과 균형잡힌 산미
"""

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
    
    def create_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        검색된 문서와 쿼리를 바탕으로 프롬프트를 생성합니다.
        짧고 간결한 프롬프트를 생성하여 모델이 빠르게 응답할 수 있도록 합니다.
        """
        try:
            # 매우 간결한 컨텍스트 생성 (빠른 처리를 위해)
            context = ""
            for idx, doc in enumerate(docs[:2]):  # 처리 시간 단축을 위해 2개 문서만 사용
                content = doc["content"].strip()
                # 200자로 제한하여 처리 속도 향상
                context += f"커피 {idx+1}: {content[:200]}...\n\n"
            
            # 속도 최적화를 위한 매우 짧고 명확한 프롬프트
            prompt = f"""커피 정보를 바탕으로 산미가 강한 커피 3가지를 추천해주세요.

{context}

질문: {query}

아래 형식으로 정확히 산미가 강한 커피 3가지만 추천해주세요:

## 커피 추천

1. **[원산지] [이름]**
   - **맛 프로필**: (산미 관련 특징)
   - **로스팅**: (로스팅 정보)
   - **특징**: (간략한 특징)

2. **[원산지] [이름]**
   ...

3. **[원산지] [이름]**
   ...
"""
            
            logger.info(f"Created prompt with length: {len(prompt)}")
            return prompt
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return f"산미가 강한 커피 3가지를 추천해주세요. 질문: {query}"
    
    def generate_answer(self, prompt: str) -> str:
        """
        LLM을 사용하여 답변을 생성합니다.
        더 빠른 응답을 위해 토큰 수를 제한하고 temperature를 낮춥니다.
        """
        try:
            logger.info("Generating answer...")
            start_time = time.time()
            
            # 직접 파이프라인 최적화 설정
            if hasattr(self.llm, 'pipeline'):
                logger.info("Using optimized direct pipeline")
                # 속도 향상을 위한 최적화 설정
                pipe_result = self.llm.pipeline(
                    prompt,
                    max_new_tokens=200,      # 토큰 수 제한으로 속도 향상
                    temperature=0.2,         # 낮은 temperature로 일관된 응답
                    top_p=0.8,
                    return_full_text=False,  # 프롬프트 반복 방지
                    repetition_penalty=1.2,  # 불필요한 반복 방지
                    num_beams=1,             # 빔 서치 비활성화로 속도 향상
                    do_sample=False          # 샘플링 비활성화로 결정적 응답 생성
                )
                
                # 파이프라인 결과 추출
                if isinstance(pipe_result, list) and len(pipe_result) > 0:
                    if "generated_text" in pipe_result[0]:
                        raw_text = pipe_result[0]["generated_text"]
                    else:
                        raw_text = str(pipe_result[0])
                else:
                    raw_text = str(pipe_result)
            else:
                # LangChain 인터페이스 사용
                response = self.llm.invoke(prompt)
                
                if isinstance(response, str):
                    raw_text = response
                elif hasattr(response, 'content'):
                    raw_text = response.content
                else:
                    raw_text = str(response)
            
            # 응답이 충분한지 확인
            if len(raw_text.strip()) < 50:
                # 응답이 너무 짧으면 기본 추천 사용
                logger.warning(f"Response too short ({len(raw_text)}), using default recommendation")
                raw_text = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
            
            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f} seconds")
            logger.info(f"Raw response preview: {raw_text[:100]}...")
            
            return raw_text
                
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생 시 기본 추천 제공
            return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
    
    def extract_answer(self, full_response: str) -> str:
        """
        생성된 전체 응답에서 실제 답변 부분만 추출합니다.
        """
        try:
            logger.info(f"Extracting answer from response of length {len(full_response)}")
            
            # 응답이 너무 짧은 경우 기본 추천 사용
            if len(full_response.strip()) < 50:
                logger.warning("Response too short, using default recommendation")
                return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
            
            # "## 커피 추천" 마커 찾기
            if "## 커피 추천" in full_response:
                answer = full_response[full_response.find("## 커피 추천"):]
                logger.info(f"Found marker '## 커피 추천', extracted response")
                
                # 추출된 응답이 실제 추천을 포함하는지 확인
                if "**[" not in answer or len(answer.strip()) < 100:
                    logger.warning("Extracted response incomplete, using default recommendation")
                    return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                
                return answer
            
            # 다른 형태의 마커 찾기
            if "커피 추천" in full_response:
                # 마커 위치 찾기
                start_idx = full_response.find("커피 추천")
                answer = "## " + full_response[start_idx:]
                logger.info(f"Found alternative marker, extracted response")
                
                if "**[" not in answer or len(answer.strip()) < 100:
                    logger.warning("Alternative marker response incomplete, using default")
                    return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                
                return answer
            
            # 어떤 마커도 없는 경우 기본 추천 사용
            logger.warning("No markers found, using default recommendation")
            return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
            
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생 시 기본 추천 제공
            return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
    
    def process_query(self, query: str) -> Dict[str, str]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        타임아웃을 120초로 늘려 충분한 처리 시간을 제공합니다.
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
            prompt = self.create_prompt(query, retrieved_docs)
            prompt_time = time.time() - start_time - retrieval_time
            logger.info(f"Prompt creation completed in {prompt_time:.2f} seconds")
            
            # 3. 답변 생성 (늘어난 타임아웃 적용)
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                generation_start = time.time()
                logger.info("Generating answer with extended timeout...")
                
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.generate_answer, prompt)
                    try:
                        # 타임아웃을 120초로 늘림 (충분한 생성 시간 제공)
                        raw_response = future.result(timeout=120)
                        generation_time = time.time() - generation_start
                        logger.info(f"Answer generation completed in {generation_time:.2f} seconds")
                    except TimeoutError:
                        logger.warning("Answer generation timed out after 120 seconds")
                        raw_response = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                        generation_time = time.time() - generation_start
                        logger.warning(f"Using default response after {generation_time:.2f} seconds timeout")
            except Exception as e:
                logger.error(f"Error during answer generation with timeout: {str(e)}")
                logger.error(traceback.format_exc())
                raw_response = f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
                generation_time = 0
            
            # 4. 답변 추출
            final_answer = self.extract_answer(raw_response)
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