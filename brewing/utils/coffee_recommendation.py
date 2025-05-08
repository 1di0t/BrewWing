import logging
from typing import List, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 커피 추천 데이터
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
        # 문서 내용에서 중요 정보 추출 (최대 3개 문서만 사용)
        coffee_info = []
        for idx, doc in enumerate(docs[:3]):
            content = doc["content"].strip()
            
            # 정보 추출
            origin = ""
            roast = ""
            description = ""
            
            lines = content.split('\n')
            for line in lines:
                line = line.lower().strip()
                if line.startswith('origin:'):
                    origin = line[7:].strip()
                elif line.startswith('roast:'):
                    roast = line[6:].strip()
                elif line.startswith('description'):
                    description = line[line.find(':')+1:].strip()
            
            coffee_info.append({
                "origin": origin,
                "roast": roast,
                "description": description[:150]  # 길이 제한
            })
        
        # 추출한 정보를 기반으로 간결한 프롬프트 작성
        context = ""
        for idx, coffee in enumerate(coffee_info):
            context += f"커피 {idx+1}:\n"
            context += f"- 원산지: {coffee['origin']}\n"
            context += f"- 로스팅: {coffee['roast']}\n"
            context += f"- 설명: {coffee['description']}\n\n"
        
        # Llama 모델용 프롬프트 템플릿
        prompt = f"""<|system|>
당신은 커피 추천 전문가입니다. 아래 커피 정보를 바탕으로 산미가 강한 커피 3가지를 추천해주세요.
</s>

<|user|>
다음 커피 정보를 참고해서 산미가 강한 커피 3가지만 추천해주세요:

{context}

요청: {query}

아래 형식으로 정확히 3가지 커피를 추천해주세요:
## 커피 추천

1. **[원산지] [이름]**
   - **맛 프로필**: (산미 관련 특징)
   - **로스팅**: (로스팅 정보)
   - **특징**: (간략한 특징)

2. **[원산지] [이름]**
   ...

3. **[원산지] [이름]**
   ...
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
            return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
        
        # 응답에 "## 커피 추천" 마커 있는지 확인
        if "## 커피 추천" in full_response:
            content = full_response[full_response.find("## 커피 추천"):]
            
            # "**[" 마커가 있는지 확인 (실제 추천 여부 확인)
            if "**[" not in content or len(content) < 100:
                logger.warning("Response lacks actual recommendations, using default")
                return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
            
            logger.info(f"Found valid coffee recommendations, length: {len(content)}")
            return content
        
        # 마커가 없지만 내용이 있는 경우
        if "**[" in full_response or "[원산지]" in full_response:
            # 추천 시작 지점 찾기
            for marker in ["1. **[", "1.**", "1. [원산지]"]:
                if marker in full_response:
                    start_idx = full_response.find(marker)
                    content = "## 커피 추천\n\n" + full_response[start_idx:]
                    logger.info(f"Extracted recommendations with marker {marker}")
                    return content
        
        # 형식화된 추천을 찾지 못한 경우
        logger.warning("Could not find properly formatted recommendations")
        return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"
        
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        # 오류 발생 시 기본 추천 제공
        return f"## 커피 추천\n{DEFAULT_RECOMMENDATIONS}"