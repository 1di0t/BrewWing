# E:\self\brewWing\brewing\utils\direct_rag_fallback.py

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def process_vector_results(retrieved_docs: List[Dict[str, Any]], query: str) -> str:
    """
    벡터 검색 결과를 직접 처리하여 커피 추천 텍스트를 생성합니다.
    Hugging Face API가 작동하지 않을 때 대체 방법으로 사용됩니다.
    
    Args:
        retrieved_docs: 벡터 검색으로 가져온 문서 리스트
        query: 사용자 쿼리
        
    Returns:
        생성된 추천 텍스트
    """
    try:
        logger.info(f"Fallback: Processing vector results directly for query: {query}")
        
        # 검색된 문서에서 커피 정보 추출
        coffee_results = []
        
        # 최대 3개 문서만 처리
        for idx, doc in enumerate(retrieved_docs[:3]):
            content = doc["content"]
            lines = content.split('\n')
            
            coffee_info = {
                "source": f"COFFEE_{idx+1}", 
                "origin": "", 
                "roast": "", 
                "flavor": "",
                "agtron": "",
                "description": ""
            }
            
            # 기본 정보 추출
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('origin:'):
                    coffee_info["origin"] = line[7:].strip().title()
                elif line_lower.startswith('roast:'):
                    coffee_info["roast"] = line[6:].strip().title()
                elif line_lower.startswith('agtron:'):
                    coffee_info["agtron"] = line[7:].strip()
                elif line_lower.startswith('description'):
                    description_parts = line.split(':', 1)
                    if len(description_parts) > 1:
                        coffee_info["description"] += description_parts[1].strip() + " "
                # 추가 내용 통합
                elif "description" in coffee_info and coffee_info["description"] and not line_lower.startswith(('origin:', 'roast:', 'agtron:')):
                    coffee_info["description"] += line.strip() + " "
            
            # 맛 프로필 추출
            if "description" in coffee_info and coffee_info["description"]:
                desc = coffee_info["description"].lower()
                
                # 산미 키워드 추출
                acidity_keywords = ["acidity", "bright", "citrus", "lemon", "lime", "orange", "tangy", 
                                   "tart", "sour", "vibrant", "fruity", "berry", "apple", "grape"]
                
                found_keywords = []
                for keyword in acidity_keywords:
                    if keyword in desc:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    coffee_info["flavor"] = f"산미 특성: {', '.join(found_keywords)}"
                else:
                    coffee_info["flavor"] = "상세 맛 정보 없음"
            
            coffee_results.append(coffee_info)
        
        # 검색된 커피 정보로 답변 구성
        results_text = "## 커피 추천\n\n"
        
        for idx, coffee in enumerate(coffee_results):
            origin = coffee["origin"] if coffee["origin"] else "Unknown"
            
            # 산미 관련 텍스트 강조
            flavor_text = coffee['flavor'] if coffee['flavor'] else '상세 정보 없음'
            if "산미" in query.lower() and "산미" not in flavor_text.lower():
                if "citrus" in flavor_text.lower() or "tart" in flavor_text.lower() or "bright" in flavor_text.lower():
                    flavor_text += " (산미가 있는 커피입니다)"
            
            results_text += f"{idx+1}. **[{origin}]**\n"
            results_text += f"- **맛 프로필**: {flavor_text}\n"
            results_text += f"- **로스팅**: {coffee['roast'] if coffee['roast'] else '상세 정보 없음'}\n"
            
            # 특징 부분 구성
            if coffee["agtron"]:
                results_text += f"- **특징**: {coffee['description'][:100] if coffee['description'] else '벡터 검색으로 찾은 커피'} (Agtron: {coffee['agtron']})\n"
            else:
                results_text += f"- **특징**: {coffee['description'][:100] if coffee['description'] else '벡터 검색으로 찾은 커피'}\n"
                
            results_text += f"- **출처**: {coffee['source']}\n\n"
        
        results_text += "**참고**: 이 추천은 벡터 검색 결과를 직접 가공한 것입니다. 더 정확한 응답을 위해 Hugging Face API 토큰을 설정해주세요."
        
        return results_text
        
    except Exception as e:
        logger.error(f"Error in fallback processing: {str(e)}")
        return f"## 커피 추천\n\n벡터 검색 결과를 처리하는 중 오류가 발생했습니다: {str(e)}"