import traceback
import logging
import os

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.permissions import AllowAny
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie

from .services import initialize_coffee_chain, recommend_coffee, is_initialized, direct_rag
from django.middleware.csrf import get_token

logger = logging.getLogger(__name__)

# 지연 초기화(Lazy Initialization)를 위해 서버 시작 시 초기화하지 않음
# try:
#     logger.info("Initializing coffee chain on server startup...")
#     initialize_coffee_chain()
#     logger.info("Coffee chain initialized successfully")
# except Exception as e:
#     logger.error(f"Failed to initialize coffee chain: {str(e)}")
#     logger.error(traceback.format_exc())

# 대신 Health Check 엔드포인트에서만 초기화 상태 확인
logger.info("Coffee chain will be initialized on first request (lazy initialization)")

@ensure_csrf_cookie
def get_csrf_token(request):
    logger.info("CSRF token requested")
    return JsonResponse({'csrfToken': get_token(request)})

@method_decorator(ensure_csrf_cookie, name='dispatch')
class CoffeeRecommendationView(APIView):
    authentication_classes = []  # disable authentication for this view
    permission_classes = [AllowAny] 

    def post(self, request):
        """
        POST 요청 처리: 사용자 질문 기반 커피 추천
        """
        logger.info("Received POST request to /api/recommend/")
        
        try:
            # 요청 데이터 로깅
            logger.info(f"Request data: {request.data}")
            user_query = request.data.get("query", "")
            logger.info(f"User query: {user_query}")
            
            if not user_query:
                logger.warning("Empty query received")
                return Response(
                    {"error": "Query parameter is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # sysytem information logging (optional)
            try:
                import platform
                import psutil
                system_info = {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "memory": psutil.virtual_memory()._asdict(),
                }
                logger.info(f"System info: {system_info}")
            except Exception as e:
                logger.warning(f"Failed to get system info: {str(e)}")

            # 서비스 계층 호출
            logger.info("Calling recommend_coffee service...")
            recommendation = recommend_coffee(user_query)
            logger.info("Recommendation service returned successfully")
            for key, value in recommendation.items():
                logger.info(f"{key}: {value}")
            
            # 응답 정보 로깅 (민감 정보 제외)
            result_preview = recommendation.get("answer", {}).get("result", "")[:100]
            logger.info(f"Result preview: {result_preview}...")
            
            return Response({
                "response": {
                    "query": user_query,
                    "result": recommendation.get("answer", {}).get("result")
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # 상세한 오류 로깅
            logger.error(f"Error in recommend API: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 디버그 환경에서만 상세 오류 정보 반환
            debug_mode = os.environ.get("DEBUG", "FALSE").upper() == "TRUE"
            error_details = {
                "error": f"서버 오류: {str(e)}",
                "type": type(e).__name__
            }
            
            if debug_mode:
                error_details["traceback"] = traceback.format_exc()
                
            return Response(
                error_details,
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get(self, request):
        """CSRF token request endpoint"""
        logger.info("CSRF token requested via GET /api/recommend/")
        return Response(
            {"csrfToken": request.META.get("CSRF_COOKIE")},
            status=status.HTTP_200_OK
        )

# 건강 체크 엔드포인트 추가
def health_check(request):
    """서버 상태 및 모델 로드 상태 확인"""
    logger.info("Health check requested")
    
    try:
        # 시스템 정보 수집
        import platform
        import psutil
        
        # 기본 시스템 정보
        system_info = {
            "status": "healthy",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "cpu_count": psutil.cpu_count(),
        }
        
        # 모델 상태 확인 - 지연 초기화 사용으로 현재 상태만 보고
        model_status = {
            "is_initialized": is_initialized,
            "direct_rag_available": direct_rag is not None,
        }
        
        # 지연 초기화 구현으로 실제 초기화를 시도하지 않음
        
        # 모델 디렉토리 확인
        cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
        model_dirs = []
        if os.path.exists(cache_dir):
            model_dirs = os.listdir(cache_dir)
        
        # FAISS 인덱스 확인
        faiss_paths = [
            "/app/brewing/faiss_store",
            "/app/faiss_store",
            "faiss_store"
        ]
        faiss_info = {}
        for path in faiss_paths:
            if os.path.exists(path):
                faiss_info[path] = os.listdir(path)
        
        return JsonResponse({
            "status": "healthy",
            "system_info": system_info,
            "model_status": model_status,
            "model_directories": model_dirs,
            "faiss_indices": faiss_info
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__
        }, status=500)