# Base image
FROM python:3.10-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED 1

# 작업 디렉터리 설정
WORKDIR /brewing

# 의존성 설치
COPY brewing/requirements.txt /brewing/
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY brewing /brewing/

# 포트 노출
EXPOSE 8000

# 서버 실행 명령어
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
