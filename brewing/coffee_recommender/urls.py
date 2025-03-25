from django.urls import path
from .views import get_csrf_token, CoffeeRecommendationView

urlpatterns = [
    path('csrf/', get_csrf_token, name='csrf_token'),  # CSRF 토큰 발급
    path('api/recommend/', CoffeeRecommendationView.as_view(), name='recommend'), # API 엔드포인트
]