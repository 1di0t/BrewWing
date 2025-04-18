import traceback

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.permissions import AllowAny
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie

from .services import initialize_coffee_chain, recommend_coffee
from django.middleware.csrf import get_token

initialize_coffee_chain()

@ensure_csrf_cookie
async def get_csrf_token(request):
    return JsonResponse({'csrfToken': get_token(request)})

@method_decorator(ensure_csrf_cookie, name='dispatch')
class CoffeeRecommendationView(APIView):
    authentication_classes = []  # 인증 비활성화 (CSRF만 검증)
    permission_classes = [AllowAny] 

    async def post(self, request):
        """
        POST 요청 처리: 사용자 질문 기반 커피 추천
        """
        try:
            user_query = request.data.get("query", "")
            
            if not user_query:
                return Response(
                    {"error": "Query parameter is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 서비스 계층 호출
            recommendation = await recommend_coffee(user_query)
            
            return Response({
                "response": {
                    "query": user_query,
                    "result": recommendation.get("answer", {}).get("result")
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(traceback.format_exc())
            return Response(
                {"error": f"서버 오류: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def get(self, request):
        """CSRF 토큰 발급 엔드포인트"""
        return Response(
            {"csrfToken": request.META.get("CSRF_COOKIE")},
            status=status.HTTP_200_OK
        )