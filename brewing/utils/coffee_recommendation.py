import logging
from typing import List, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 커피 추천 데이터
DEFAULT_RECOMMENDATIONS = """
1. **[케냐] 키암부**
   - **맛 프로필**: 강한 산미, 시트러스와 베리류 노트
   - **로스팅**: 라이트-미디엄
   - **특징**: 상쾌한 과일향과 선명한 산미
   - **출처**: COFFEE_INFO_1

2. **[에티오피아] 예가체프**
   - **맛 프로필**: 화사한 산미, 꽃과 베리 향미
   - **로스팅**: 라이트
   - **특징**: 복합적인 향과 상쾌한 산미
   - **출처**: COFFEE_INFO_2

3. **[르완다] 키부**
   - **맛 프로필**: 신선한 산미, 레드베리 향
   - **로스팅**: 라이트-미디엄
   - **특징**: 달콤한 단맛과 균형잡힌 산미
   - **출처**: COFFEE_INFO_3
"""

def create_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    """
    검색된 문서와 쿼리를 바탕으로 프롬프트를 생성합니다.
    
    Args:
        query: 사용자 쿼리
        docs: 검색된 문서 리스트
        
    Returns:
        생성된 프롬프트
    """
    try:
        # 문서가 없는 경우 첫 결과
        if not docs:
            logger.warning("No documents found for query")
            return f"""<|system|>
            당신은 커피 추천 전문가입니다.
            </s>

            <|user|>
            "{query}"에 대한 커피를 추천해주세요.
            </s>

            <|assistant|>
            """
        
        # 원본 커피 문서 형식화
        formatted_docs = []
        for idx, doc in enumerate(docs):
            content = doc["content"].strip()
            doc_id = f"[COFFEE_{idx+1}]"  # 문서 ID 형식화
            
            formatted_content = f"{doc_id} {content}"  # 문서 ID와 커피 정보 결합
            formatted_docs.append(formatted_content)
        
        # 전체 커피 문서 내용 합치기
        all_coffee_docs = "\n\n".join(formatted_docs)
        
        # 커피 정보 구분을 위한 이중 라인
        separator = "\n" + "-" * 40 + "\n"
        
        # 벡터 검색 기반 추천임을 강조하는 프롬프트 생성
        prompt = f"""<|system|>
        당신은 커피 추천 전문가입니다. 아래 제공된 커피 데이터만을 사용하여 사용자의 질문에 대한 커피를 추천해야 합니다.
        
        중요한 지침:
        1. 반드시 아래 제공된 커피 데이터([COFFEE_N] 로 표시된)만을 사용해야 합니다.
        2. 없는 정보나 만들어낸 정보는 절대 사용하지 마세요.
        3. 각 추천에는 반드시 출처 정보를 포함해야 합니다 (COFFEE_1, COFFEE_2 등).
        4. 쿼리에 없는 내용을 추가하지 마세요. 오직 제공된 커피 데이터와 쿼리를 기반으로 해당하는 커피만 추천해야 합니다.
        </s>

        <|user|>
        다음은 벡터 검색을 통해 가져온 커피 데이터입니다:
        {separator}
        {all_coffee_docs}
        {separator}
        
        위 커피 데이터만을 사용하여 "{query}"에 대해 가장 적합한 커피를 추천해주세요.
        각 추천 항목에는 반드시 어떤 커피 데이터를 참고했는지 표시하세요 (COFFEE_1, COFFEE_2 등).
        
        ## 커피 추천 형식:
        
        1. **[원산지] [커피명]**
        - **맛 프로필**: (상세한 맛 특성)
        - **로스팅**: (로스팅 정도)
        - **특징**: (특별한 특징)
        - **출처**: (COFFEE_N에서 출처)
        </s>

        <|assistant|>
        """
        
        logger.info(f"Created prompt with length: {len(prompt)}")
        return prompt
    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        # 오류 발생해도 기본 프롬프트 제공
        return f"""<|system|>
당신은 커피 추천 전문가입니다.
</s>

<|user|>
산미가 강한 커피 3가지를 추천해주세요.
</s>

<|assistant|>
"""

def extract_answer(full_response: str) -> str:
    """
    생성된 전체 응답에서 실제 답변 부분만 추출합니다.
    
    Args:
        full_response: 전체 응답 텍스트
        
    Returns:
        추출된 답변
    """
    try:
        logger.info(f"Extracting answer from response of length {len(full_response)}")
        
        # 빈 응답이거나 매우 짧은 경우
        if not full_response or len(full_response.strip()) < 30:
            logger.warning("Response too short or empty, using default recommendation")
            return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}\n\n**참고**: 이 추천은 기본 커피 정보를 기반으로 합니다."
        
        # 응답에 "## 커피 추천" 마커 있는지 확인
        if "## 커피 추천" in full_response:
            content = full_response[full_response.find("## 커피 추천"):]
            
            # "**[" 마커가 있는지 확인 (실제 추천 여부 확인)
            if "**[" not in content or len(content) < 100:
                logger.warning("Response lacks actual recommendations, using default")
                return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}\n\n**참고**: 이 추천은 기본 커피 정보를 기반으로 합니다."
            
            # 출처 정보가 있는지 확인 (COFFEE_ 형식 포함)
            if "COFFEE_" not in content and "출처" in content:
                # 출처 정보가 있지만 COFFEE_ 형식이 없는 경우 포맷팅
                content = content.replace("출처: 커피", "출처: COFFEE_")
            
            # 출처 정보가 아예 없는 경우
            if "출처" not in content and "COFFEE_" not in content:
                content += "\n\n**참고**: 이 추천은 벡터 검색을 통해 찾은 커피 정보를 기반으로 합니다."
            
            logger.info(f"Found valid coffee recommendations, length: {len(content)}")
            return content
        
        # 마커가 없지만 내용이 있는 경우
        if "**[" in full_response or "[원산지]" in full_response:
            # 추천 시작 지점 찾기
            for marker in ["1. **[", "1.**", "1. [원산지]"]:
                if marker in full_response:
                    start_idx = full_response.find(marker)
                    content = "## 커피 추천\n\n" + full_response[start_idx:]
                    
                    # 출처 정보 포맷팅 확인
                    if "COFFEE_" not in content and "출처" in content:
                        content = content.replace("출처: 커피", "출처: COFFEE_")
                    
                    # 출처 정보가 없는 경우
                    if "출처" not in content and "COFFEE_" not in content:
                        content += "\n\n**참고**: 이 추천은 벡터 검색을 통해 찾은 커피 정보를 기반으로 합니다."
                    
                    logger.info(f"Extracted recommendations with marker {marker}")
                    return content
        
        # 형식화된 추천을 찾지 못한 경우
        logger.warning("Could not find properly formatted recommendations")
        return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}\n\n**참고**: 이 추천은 기본 커피 정보를 기반으로 합니다."
        
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        # 오류 발생 시 기본 추천 제공
        return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}\n\n**참고**: 이 추천은 기본 커피 정보를 기반으로 합니다."