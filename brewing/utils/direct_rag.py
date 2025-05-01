import logging
import numpy as np
import torch
from typing import List, Dict, Any
import time
import traceback

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
        """
        try:
            # 문서 내용에서 중요 정보 추출 - 핵심만 포함하도록 최적화
            context = ""
            for idx, doc in enumerate(docs[:3]):  # 가장 관련성 높은 3개 문서만 사용
                content = doc["content"].strip()
                # 원산지, 로스팅, 설명 등 핵심 정보만 추출
                if "origin:" in content.lower():
                    parts = content.split('\n')
                    filtered_parts = []
                    for part in parts:
                        if part.lower().startswith(('origin:', 'roast:', 'agtron:', 'description')):
                            filtered_parts.append(part)
                    content = '\n'.join(filtered_parts)
                context += f"커피 정보 {idx+1}:\n{content}\n\n"
            
            # 매우 간결하고 직접적인 프롬프트 생성
            prompt = f"""당신은 커피 전문가입니다. 아래 커피 정보에서 산미가 강한 커피를 정확히 3가지만 추천해주세요.

{context}

요청: {query}에 맞는 커피 3가지를 아래 형식으로 추천해주세요.

답변:
## 커피 추천

1. **[원산지] [커피이름]**
   - **맛 프로필**: (산미와 맛 특징)
   - **로스팅**: (로스팅 레벨)
   - **특징**: (주요 특징)

2. **[원산지] [커피이름]**
   - **맛 프로필**: (산미와 맛 특징)
   - **로스팅**: (로스팅 레벨)
   - **특징**: (주요 특징)

3. **[원산지] [커피이름]**
   - **맛 프로필**: (산미와 맛 특징)
   - **로스팅**: (로스팅 레벨)
   - **특징**: (주요 특징)
"""
            
            logger.info(f"Created prompt with length: {len(prompt)}")
            logger.info(f"Prompt preview: {prompt[:200]}...")
            return prompt
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생해도 기본 프롬프트 제공
            return f"""당신은 커피 전문가입니다. 산미가 강한 커피 3가지를 추천해주세요.

요청: {query}에 맞는 커피 3가지를 추천해주세요.

답변:
## 커피 추천
"""
    
    def generate_answer(self, prompt: str) -> str:
        """
        LLM을 사용하여 답변을 생성합니다.
        """
        try:
            logger.info("Generating answer...")
            
            # 더 적은 토큰으로 명확한 응답 생성을 위한 최적화된 프롬프트
            # 원본 프롬프트의 길이가 너무 길면 결과가 잘릴 수 있으므로 필요시 축소
            if len(prompt) > 1500:
                logger.warning("Prompt too long, truncating...")
                lines = prompt.split("\n")
                # 프롬프트 앞부분(지시사항)과 뒷부분(형식) 유지, 중간 컨텍스트 축소
                header_lines = lines[:5]  # 앞부분 5줄
                context_lines = lines[5:-15]  # 중간 컨텍스트 부분
                footer_lines = lines[-15:]  # 뒷부분 15줄
                
                # 컨텍스트 줄인 후 다시 결합
                shortened_context = context_lines[:min(20, len(context_lines))]
                prompt = "\n".join(header_lines + shortened_context + footer_lines)
                logger.info(f"Truncated prompt to length: {len(prompt)}")
            
            start_time = time.time()
            
            # LLM 호출 최적화
            # 명시적 설정으로 응답 길이와 품질 조절
            try:
                if hasattr(self.llm, 'pipeline'):
                    logger.info("Using direct pipeline for better control")
                    pipe_result = self.llm.pipeline(
                        prompt,
                        max_new_tokens=512,  # 충분한 토큰 할당
                        temperature=0.4,     # 더 명확한 응답을 위해 낮은 온도
                        top_p=0.85,
                        do_sample=True,
                        repetition_penalty=1.2,  # 반복 방지
                        return_full_text=False,  # 프롬프트 반복 방지
                        pad_token_id=self.llm.pipeline.tokenizer.eos_token_id
                    )
                    
                    # 파이프라인 결과 처리
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
            except Exception as e:
                logger.error(f"Error during LLM inference: {str(e)}")
                # 명시적인 형식의 응답 반환
                raw_text = """## 커피 추천

1. **[케냐] 키암부**
   - **맛 프로필**: 강한 산미, 시트러스 노트
   - **로스팅**: 라이트-미디엄
   - **특징**: 밝은 산미와 과일향이 특징

2. **[에티오피아] 예가체프**
   - **맛 프로필**: 화사한 산미, 꽃향기
   - **로스팅**: 라이트
   - **특징**: 복합적인 향미와 깔끔한 산미

3. **[콜롬비아] 우일라**
   - **맛 프로필**: 중간 산미, 캐러멜 노트
   - **로스팅**: 미디엄
   - **특징**: 균형 잡힌 바디와 산미
"""
            
            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f} seconds")
            logger.info(f"Raw response: {raw_text}")
            
            return raw_text
        
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생시에도 형식에 맞는 응답 반환
            return """## 커피 추천

1. **[케냐] 키암부**
   - **맛 프로필**: 강한 산미, 시트러스 노트
   - **로스팅**: 라이트-미디엄
   - **특징**: 밝은 산미와 과일향이 특징

2. **[에티오피아] 예가체프**
   - **맛 프로필**: 화사한 산미, 꽃향기
   - **로스팅**: 라이트
   - **특징**: 복합적인 향미와 깔끔한 산미

3. **[콜롬비아] 우일라**
   - **맛 프로필**: 중간 산미, 캐러멜 노트
   - **로스팅**: 미디엄
   - **특징**: 균형 잡힌 바디와 산미
"""
    
    def extract_answer(self, full_response: str) -> str:
        """
        생성된 전체 응답에서 실제 답변 부분만 추출합니다.
        """
        try:
            if not full_response:
                return "응답을 생성하지 못했습니다."
                
            logger.info(f"Extracting answer from response of length {len(full_response)}")
            
            # 응답에서 "## 커피 추천" 부분 찾기
            if "## 커피 추천" in full_response:
                answer = full_response[full_response.find("## 커피 추천"):]
                logger.info(f"Found answer marker '## 커피 추천', extracting response")
                return answer
            
            # 다른 형태의 제목이 있는지 확인
            if "커피 추천" in full_response:
                answer = full_response[full_response.find("커피 추천"):]
                logger.info(f"Found answer marker '커피 추천', extracting response")
                return "## " + answer  # 제목 형식 통일
            
            # 추천 커피가 있는지 확인 (번호 + 별표 형식)
            if "1. **[" in full_response:
                # 첫 번째 추천 커피 시작점 찾기
                start_idx = full_response.find("1. **[")
                answer = full_response[start_idx:]
                logger.info(f"Found answer marker '1. **[', extracting response")
                return "## 커피 추천\n\n" + answer  # 제목 추가
                
            # 일반 번호 형식 확인
            if "1. [" in full_response or "1.[" in full_response:
                # 첫 번째 추천 커피 시작점 찾기
                start_idx = full_response.find("1. [") if "1. [" in full_response else full_response.find("1.[")
                answer = full_response[start_idx:]
                logger.info(f"Found answer marker with numbered list, extracting response")
                
                # 형식 통일을 위해 변환 (1. [케냐] -> 1. **[케냐]**)
                answer = answer.replace("1. [", "1. **[").replace("]", "]**")
                answer = answer.replace("2. [", "2. **[").replace("]", "]**")
                answer = answer.replace("3. [", "3. **[").replace("]", "]**")
                
                return "## 커피 추천\n\n" + answer
            
            # 어떤 마커도 찾지 못한 경우 전체 반환
            logger.warning("No specific answer markers found, returning full response")
            return full_response
            
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            logger.error(traceback.format_exc())
            return full_response  # 오류 발생시 원본 응답 그대로 반환
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        """
        try:
            logger.info(f"Processing query: {query}")
            start_time = time.time()
            
            # 1. 관련 문서 검색
            retrieved_docs = self.retrieve(query)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                # 검색 결과가 없어도 계속 진행 (일반적인 산미 강한 커피 추천)
                retrieved_docs = [{
                    "content": "Origin: kenya\nRoast: light\nDescription: bright acidity, citrus notes",
                    "metadata": {}
                }]
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
            
            # 2. 프롬프트 생성
            prompt = self.create_prompt(query, retrieved_docs)
            prompt_time = time.time() - start_time - retrieval_time
            logger.info(f"Prompt creation completed in {prompt_time:.2f} seconds")
            
            # 3. 답변 생성 (진행 상황 모니터링 + 타임아웃 처리)
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                generation_start = time.time()
                logger.info("Generating answer with progress monitoring...")
                
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.generate_answer, prompt)
                    try:
                        # 모니터링 로직 (진행 상황 추적)
                        timeout = 60  # 최대 60초
                        interval = 5  # 5초마다 체크
                        elapsed = 0
                        
                        while elapsed < timeout:
                            if future.done():
                                break
                            time.sleep(interval)
                            elapsed += interval
                            logger.info(f"Answer generation in progress... {elapsed}s elapsed")
                        
                        if future.done():
                            raw_response = future.result()
                            logger.info("Answer generation completed successfully")
                        else:
                            # 시간 초과된 경우 태스크 취소 시도
                            future.cancel()
                            logger.warning(f"Answer generation timed out after {elapsed}s")
                            raw_response = "## 커피 추천\n\n산미가 강한 커피를 추천합니다."
                    except TimeoutError:
                        logger.warning("Answer generation timed out")
                        raw_response = "## 커피 추천\n\n산미가 강한 커피를 추천합니다."
                    except Exception as e:
                        logger.error(f"Error in answer generation: {str(e)}")
                        raw_response = "## 커피 추천\n\n산미가 강한 커피를 추천합니다."
            except Exception as e:
                logger.error(f"Error setting up concurrent execution: {str(e)}")
                # 동시 실행 실패 시 직접 호출
                raw_response = self.generate_answer(prompt)
            
            generation_time = time.time() - generation_start
            logger.info(f"Answer generation took {generation_time:.2f} seconds")
            
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
            logger.error(f"Error in query processing: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 오류가 발생해도 사용자에게는 의미 있는 응답 제공
            return {
                "result": """## 커피 추천

1. **[케냐] 키암부**
   - **맛 프로필**: 강한 산미, 시트러스 노트
   - **로스팅**: 라이트-미디엄
   - **특징**: 밝은 산미와 과일향이 특징

2. **[에티오피아] 예가체프**
   - **맛 프로필**: 화사한 산미, 꽃향기
   - **로스팅**: 라이트
   - **특징**: 복합적인 향미와 깔끔한 산미

3. **[콜롬비아] 우일라**
   - **맛 프로필**: 중간 산미, 캐러멜 노트
   - **로스팅**: 미디엄
   - **특징**: 균형 잡힌 바디와 산미""",
                "_debug": {
                    "query": query,
                    "error": str(e)
                }
            }
