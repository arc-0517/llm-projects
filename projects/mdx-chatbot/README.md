# MDX 문서 기반 챗봇

MDX 형식의 문서를 처리하여 질의응답 챗봇을 구현한 프로젝트입니다.

## 기능

- MDX 파일 처리 및 텍스트 추출
- 텍스트 청크 분할 및 벡터화
- RAG(Retrieval-Augmented Generation) 시스템 구현
- 대화형 챗봇 인터페이스

## 사용 방법

1. 필요한 라이브러리 설치: `pip install -r requirements.txt`
2. MDX 파일 처리: `python process_mdx.py`
3. 벡터 데이터베이스 생성: `python create_vectordb.py`
4. 챗봇 실행: `python rag_chatbot.py`
EOL