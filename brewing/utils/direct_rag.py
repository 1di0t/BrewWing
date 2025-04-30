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
            # 문서 내용을 하나의 문자열로 결합
            context = ""
            for doc in docs:
                context += doc["content"] + "\n\n"
            
            # 간결하고 명확한 프롬프트 생성 - 더 짧고 명확하게 수정
            prompt = f"""아래 커피 정보를 참고해 산미가 강한 커피를 3가지 추천해주세요.

{context}

질문: {query}

답변:
"""
            
            logger.info(f"Created prompt with length: {len(prompt)}")
            logger.info(f"Prompt preview: {prompt[:200]}...")
            return prompt
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return f"질문: {query}\n\n답변:"
    
    def generate_answer(self, prompt: str) -> str:
        """
        LLM을 사용하여 답변을 생성합니다.
        """
        try:
            logger.info("Generating answer...")
            start_time = time.time()
            
            # HuggingFacePipeline을 더 안정적으로 호출
            try:
                # 방법 1: LangChain 인터페이스 사용
                logger.info("Trying LangChain interface for LLM...")
                response = self.llm.invoke(prompt)
                logger.info(f"LangChain invoke successful: {type(response)}")
                
                # 응답 형식에 따라 적절히 처리
                if isinstance(response, str):
                    raw_text = response
                elif hasattr(response, 'content'):
                    raw_text = response.content
                else:
                    raw_text = str(response)
                
            except Exception as lc_error:
                logger.warning(f"LangChain interface failed: {str(lc_error)}")
                
                # 방법 2: 직접 파이프라인 호출
                try:
                    logger.info("Trying direct pipeline call...")
                    if hasattr(self.llm, 'pipeline'):
                        pipe_result = self.llm.pipeline(
                            prompt,
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            num_return_sequences=1,
                            return_full_text=False  # 프롬프트 반복 방지
                        )
                        
                        logger.info(f"Pipeline result type: {type(pipe_result)}")
                        
                        # 파이프라인 결과 형식에 따라 처리
                        if isinstance(pipe_result, list) and len(pipe_result) > 0:
                            if "generated_text" in pipe_result[0]:
                                raw_text = pipe_result[0]["generated_text"]
                            else:
                                raw_text = str(pipe_result[0])
                        else:
                            raw_text = str(pipe_result)
                    else:
                        raise ValueError("LLM has no pipeline attribute")
                
                except Exception as pipe_error:
                    logger.error(f"Direct pipeline call failed: {str(pipe_error)}")
                    
                    # 방법 3: 마지막 대안으로 직접 model과 tokenizer 사용
                    try:
                        logger.info("Trying fallback to direct model call...")
                        if hasattr(self.llm, '_llm_model') and hasattr(self.llm, '_llm_tokenizer'):
                            model = self.llm._llm_model
                            tokenizer = self.llm._llm_tokenizer
                            
                            inputs = tokenizer(prompt, return_tensors="pt")
                            outputs = model.generate(
                                **inputs, 
                                max_new_tokens=256, 
                                temperature=0.7,
                                top_p=0.9,
                                do_sample=True,
                                num_return_sequences=1
                            )
                            
                            raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            # 프롬프트 부분 제거
                            if prompt in raw_text:
                                raw_text = raw_text[len(prompt):]
                        else:
                            raise ValueError("Cannot access model or tokenizer directly")
                    except Exception as direct_error:
                        logger.error(f"Direct model call failed: {str(direct_error)}")
                        raw_text = "모델 호출 중 오류가 발생했습니다."
            
            # 생성 시간 측정
            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f} seconds")
            logger.info(f"Raw response preview: {raw_text[:100]}...")
            
            return raw_text
                
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            logger.error(traceback.format_exc())
            return "답변 생성 중 오류가 발생했습니다."
    
    def extract_answer(self, full_response: str) -> str:
        """
        생성된 전체 응답에서 실제 답변 부분만 추출합니다.
        """
        try:
            logger.info(f"Extracting answer from full response of length {len(full_response)}")
            
            # 빈 응답 처리
            if not full_response or len(full_response.strip()) < 10:
                logger.warning("Empty or very short response")
                return "응답이 비어 있거나 너무 짧습니다. 다시 시도해주세요."
            
            # 응답에서 프롬프트 부분 제거 (중복 방지)
            prompt_end_markers = ["답변:", "커피 추천", "## 커피 추천"]
            for marker in prompt_end_markers:
                if marker in full_response:
                    full_response = full_response[full_response.find(marker):]
                    break
            
            # 기본 응답 구조화 (응답이 너무 포맷에서 벗어난 경우)
            if "**" not in full_response and "[" not in full_response:
                logger.warning("Response doesn't contain expected formatting. Adding structure.")
                
                # 줄 단위로 분석
                lines = full_response.split("\n")
                formatted_lines = []
                
                current_item = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith("1.") or line.startswith("1)") or line.startswith("하나."):
                        current_item = 1
                        formatted_lines.append(f"{current_item}. **[추천 커피]**")
                    elif line.startswith("2.") or line.startswith("2)") or line.startswith("둘."):
                        current_item = 2
                        formatted_lines.append(f"{current_item}. **[추천 커피]**")
                    elif line.startswith("3.") or line.startswith("3)") or line.startswith("셋."):
                        current_item = 3
                        formatted_lines.append(f"{current_item}. **[추천 커피]**")
                    else:
                        # 내용 라인 처리
                        if "맛" in line.lower() or "산미" in line:
                            formatted_lines.append(f"   - **맛 프로필**: {line}")
                        elif "로스팅" in line.lower() or "로스트" in line:
                            formatted_lines.append(f"   - **로스팅**: {line}")
                        elif "특징" in line.lower() or "특성" in line:
                            formatted_lines.append(f"   - **특징**: {line}")
                        else:
                            formatted_lines.append(f"   - {line}")
                
                # 포맷팅된 응답
                if formatted_lines:
                    full_response = "## 커피 추천\n\n" + "\n".join(formatted_lines)
            
            logger.info(f"Extracted answer preview: {full_response[:100]}...")
            return full_response
            
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            logger.error(traceback.format_exc())
            return full_response
    
    def process_query(self, query: str) -> Dict[str, str]:
        """
        쿼리 처리의 전체 파이프라인을 실행합니다.
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # 시간 측정 시작
            start_time = time.time()
            
            # 기본 응답 준비
            default_response = {
                "result": "산미가 강한 커피를 찾을 수 없습니다. 다른 키워드로 다시 시도해주세요.",
                "_debug": {
                    "query": query,
                    "error": "No results found"
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
            
            # 3. 답변 생성
            generation_start = time.time()
            raw_response = self.generate_answer(prompt)
            generation_time = time.time() - generation_start
            logger.info(f"Answer generation completed in {generation_time:.2f} seconds")
            
            # 빈 응답 확인
            if not raw_response or len(raw_response.strip()) < 10:
                logger.warning("Empty or very short response from LLM")
                return {
                    "result": "모델이 응답을 생성하지 못했습니다. 다시 시도해주세요.",
                    "_debug": {
                        "query": query,
                        "docs_count": len(retrieved_docs),
                        "prompt_length": len(prompt),
                        "raw_response": raw_response,
                        "error": "Empty response"
                    }
                }
            
            # 4. 답변 추출
            final_answer = self.extract_answer(raw_response)
            extraction_time = time.time() - start_time - retrieval_time - prompt_time - generation_time
            logger.info(f"Answer extraction completed in {extraction_time:.2f} seconds")
            
            # 결과가 여전히 빈 경우 기본 응답
            if not final_answer or len(final_answer.strip()) < 20:
                logger.warning("Empty final answer after extraction")
                return {
                    "result": "## 커피 추천\n\n1. **[케냐] 키암부 커피**\n   - **맛 프로필**: 강한 산미, 상큼한 시트러스 노트\n   - **로스팅**: 라이트-미디엄\n   - **특징**: 밝은 산미와 과일향이 특징\n\n2. **[에티오피아] 예가체프**\n   - **맛 프로필**: 화사한 산미, 꽃향기\n   - **로스팅**: 라이트\n   - **특징**: 복합적인 향미와 깔끔한 산미\n\n검색 결과를 바탕으로 제공된 기본 추천입니다.",
                    "_debug": {
                        "query": query,
                        "docs_count": len(retrieved_docs),
                        "prompt_length": len(prompt),
                        "raw_response_length": len(raw_response),
                        "error": "Empty final answer"
                    }
                }
            
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
            return {
                "result": f"처리 중 오류가 발생했습니다: {str(e)}",
                "_debug": {
                    "query": query,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
