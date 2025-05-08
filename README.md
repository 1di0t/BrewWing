# **커피 원두 추천 페이지**

사용자의 선호도에 맞는 커피 원두를 추천해주는 간단한 챗봇 기반 웹 페이지입니다.​

## **프로젝트 목적**

이 프로젝트의 주요 목표는 Hugging Face 라이브러리, LangChain, RAG(Retrieval-Augmented Generation) 기술을 활용하여 사용자 입력에 기반한 추천 시스템을 제공하는 것입니다.​

## **주요 기능**

- LLaMA와 Hugging Face 라이브러리를 활용한 자연어 처리​
- LangChain을 통한 대화형 인터랙션 관리​
- RAG(Retrieval-Augmented Generation)를 활용한 추천 시스템 강화​
- 사용자가 쉽게 상호작용할 수 있는 챗봇 인터페이스 제공​

## **기술 스택**

<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/django-%23092E20.svg?&style=for-the-badge&logo=django&logoColor=white" /> <img src="https://img.shields.io/badge/react-%2361DAFB.svg?&style=for-the-badge&logo=react&logoColor=black" />

- **백엔드**: Python, Django
- **프론트엔드**: React
- **자연어 처리**: Hugging Face Transformers, LLaMA
- **대화 관리**: LangChain​
- **추천 시스템**: RAG(Retrieval-Augmented Generation)

## **설치 및 실행 방법**

### 백엔드 설치 및 실행
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 패키지 설치
pip install -r requirements.txt

# 모델 다운로드
python download_model.py

# 서버 실행
cd brewing
python manage.py runserver
```

### 프론트엔드 설치 및 실행
```bash
# 프론트엔드 디렉토리로 이동
cd brewing/react-app

# 패키지 설치
npm install

# 개발 서버 실행
npm start
```

## **프로젝트 구조**

```
brewWing/
├── .git/                  # Git 저장소
├── brewing/               # 메인 Django 프로젝트 
│   ├── brewing/           # Django 설정
│   ├── coffee_recommender/ # 커피 추천 앱
│   ├── data/              # 데이터 파일
│   ├── react-app/         # React 프론트엔드
│   ├── utils/             # 유틸리티 함수
│   └── manage.py          # Django 관리 스크립트
├── venv/                  # 가상환경
├── .env                   # 환경 변수
├── Dockerfile.backend     # 백엔드 Docker 설정
├── requirements.txt       # Python 패키지 목록
└── README.md              # 프로젝트 설명
```

## **Docker 배포**

```bash
# Docker 이미지 빌드 및 실행
docker-compose up --build
```

## **참여자**

이 프로젝트는 개인 프로젝트로 제작되었습니다.

## **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

# **Coffee Bean Recommendation Page**

A simple chatbot-based web page that recommends coffee beans based on user preferences.

## **Project Purpose**

The main goal of this project is to provide a recommendation system based on user input using Hugging Face libraries, LangChain, and RAG (Retrieval-Augmented Generation) technology.

## **Key Features**

- Natural language processing using LLaMA and Hugging Face libraries
- Conversational interaction management through LangChain
- Enhanced recommendation system using RAG (Retrieval-Augmented Generation)
- User-friendly chatbot interface for easy interaction

## **Technology Stack**

<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/django-%23092E20.svg?&style=for-the-badge&logo=django&logoColor=white" /> <img src="https://img.shields.io/badge/react-%2361DAFB.svg?&style=for-the-badge&logo=react&logoColor=black" />

- **Backend**: Python, Django
- **Frontend**: React
- **Natural Language Processing**: Hugging Face Transformers, LLaMA
- **Conversation Management**: LangChain
- **Recommendation System**: RAG (Retrieval-Augmented Generation)

## **Installation and Execution**

### Backend Installation and Execution
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download model
python download_model.py

# Run server
cd brewing
python manage.py runserver
```

### Frontend Installation and Execution
```bash
# Navigate to frontend directory
cd brewing/react-app

# Install packages
npm install

# Run development server
npm start
```

## **Project Structure**

```
brewWing/
├── .git/                  # Git repository
├── brewing/               # Main Django project
│   ├── brewing/           # Django settings
│   ├── coffee_recommender/ # Coffee recommendation app
│   ├── data/              # Data files
│   ├── react-app/         # React frontend
│   ├── utils/             # Utility functions
│   └── manage.py          # Django management script
├── venv/                  # Virtual environment
├── .env                   # Environment variables
├── Dockerfile.backend     # Backend Docker configuration
├── requirements.txt       # Python package list
└── README.md              # Project description
```

## **Docker Deployment**

```bash
# Build and run Docker images
docker-compose up --build
```

## **Contributors**

This project was created as a personal project.

## **License**

This project is distributed under the MIT license.
