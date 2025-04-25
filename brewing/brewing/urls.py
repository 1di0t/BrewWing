"""
URL configuration for brewing project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from django.contrib import admin
from django.views.generic.base import RedirectView

# health check 뷰 추가
from coffee_recommender.views import health_check, get_csrf_token

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # health check 엔드포인트 추가
    path('api/health/', health_check, name='health_check'),
    
    # CSRF 토큰 발급 엔드포인트
    path('api/get-csrf-token/', get_csrf_token, name='get_csrf_token'),
    
    # 커피 추천 앱 API 라우트
    path('api/', include('coffee_recommender.urls')),
    
    # 리액트 앱으로 모든 요청 리다이렉트 (API 요청 제외)
    path('', RedirectView.as_view(url='/static/index.html'), name='index'),
]

# 디버그 정보 미들웨어 추가
from django.conf import settings

if settings.DEBUG:
    import debug_toolbar
    urlpatterns += [
        path('__debug__/', include(debug_toolbar.urls)),
    ]