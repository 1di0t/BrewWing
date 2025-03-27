# **Coffee Bean Recommendation Page**

사용자의 선호도에 맞는 커피 원두를 추천해주는 간단한 챗봇 기반 웹 페이지입니다.​

## **목적**

이 프로젝트의 주요 목표는 Hugging Face 라이브러리, LangChain, RAG(Retrieval-Augmented Generation) 기술을 활용하여 사용자 입력에 기반한 추천 시스템을 제공하는 것입니다. ​

## **주요 기능**

- LLaMA와 Hugging Face 라이브러리를 활용한 자연어 처리​

- LangChain을 통한 대화형 인터랙션 관리​

- RAG(Retrieval-Augmented Generation)를 활용한 추천 시스템 강화​

- 사용자가 쉽게 상호작용할 수 있는 챗봇 인터페이스 제공​

## **기술 스택**


<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/fastapi-%23009688.svg?&style=for-the-badge&logo=fastapi&logoColor=white" /> <img src="https://img.shields.io/badge/react-%2361DAFB.svg?&style=for-the-badge&logo=react&logoColor=black" />

#### **자연어 처리: Hugging Face Transformers, LLaMA**


#### **대화 관리: LangChain​**


#### **추천 시스템: RAG**



## **실행 방법**

1.  brewing 루트 디렉터리에 .env 파일 생성

`#env file  `<br>
`HUGGINGFACE_API_KEY=허깅페이스키`
- **필요한 모델 키:**
  - meta-llama/Llama-2-7b-hf
  - facebook/nllb-200-distilled-600M, 
  - sentence-transformers/all-MiniLM-L6-v2

---
#### 도커 사용시
`brewing `경로에서 아래 명령어를 실행합니다.

`docker-compose up --build`

---
#### 명령어 실행방법
`brewing/brewing` 경로에서 아래 명령어를 실행합니다.

`python manage.py runserver`

`brewing/frontend` 경로에서 아래 명령어를 실행합니다.

`npm start`

---
`cpu를 기반으로 실행됩니다.`<br>
`cpu 성능에 따라 답변 시간이 달라질 수 있습니다.`

<br><br><br><br><br>

# Coffee Bean Recommendation Page
This is a simple chatbot-based web page that recommends coffee beans tailored to the user's preferences.

## Purpose
The main goal of this project is to provide a recommendation system based on user input by utilizing the Hugging Face library, LangChain, and RAG (Retrieval-Augmented Generation) technologies.

## Key Features
- Natural language processing using LLaMA and the Hugging Face library.

- Chat-based interaction management with LangChain.

- Enhanced recommendation system using RAG (Retrieval-Augmented Generation).

- A chatbot interface that allows users to interact easily.

## Tech Stack
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/fastapi-%23009688.svg?&style=for-the-badge&logo=fastapi&logoColor=white" /> <img src="https://img.shields.io/badge/react-%2361DAFB.svg?&style=for-the-badge&logo=react&logoColor=black" />

- Natural Language Processing: Hugging Face Transformers, LLaMA
- Conversation Management: LangChain
- Recommendation System: RAG

## How to Run
1. Create a .env file in the root directory of brewing:

`# .env file`

`HUGGINGFACE_API_KEY=your_huggingface_key`

- Required Model Keys:

  - meta-llama/Llama-2-7b-hf

  - facebook/nllb-200-distilled-600M

  - sentence-transformers/all-MiniLM-L6-v2

### Using Docker
From the brewing directory, run the following command:

`docker-compose up --build`
### Running via Commands
From the brewing/brewing directory, run:

`python manage.py runserver`

From the brewing/frontend directory, run:

`npm start`

### Execution Environment:
`The application runs on a CPU.`<br>
`Response times may vary depending on the CPU performance.`
