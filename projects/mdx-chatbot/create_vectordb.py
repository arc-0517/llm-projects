"""
벡터 데이터베이스 생성 스크립트
"""

import os
import sys
import pickle

# 공통 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from common.utils.file_utils import ensure_directory

def create_vector_database(chunks, use_openai=False):
    """벡터 데이터베이스 생성"""
    # 텍스트와 메타데이터 분리
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    if use_openai:
        # OpenAI 임베딩 사용
        try:
            from langchain_openai import OpenAIEmbeddings
            from dotenv import load_dotenv
            load_dotenv()
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            print("OpenAI 임베딩 모델을 사용합니다.")
        except (ImportError, Exception) as e:
            print(f"OpenAI 임베딩을 사용할 수 없습니다: {e}")
            print("대신 Ollama 임베딩을 사용합니다.")
            use_openai = False
    
    if not use_openai:
        # Ollama 임베딩 사용
        try:
            from langchain_ollama import OllamaEmbeddings
            
            # 사용 가능한 Ollama 모델 확인
            print("사용 가능한 Ollama 모델 확인:")
            os.system("ollama list")
            
            model_name = input("\n사용할 모델 이름을 입력하세요(기본값: gemma): ")
            if not model_name:
                model_name = "gemma"
                
            embeddings = OllamaEmbeddings(model=model_name)
            print(f"Ollama {model_name} 임베딩 모델을 사용합니다.")
        except (ImportError, Exception) as e:
            print(f"Ollama 임베딩을 사용할 수 없습니다: {e}")
            print("임베딩 설정에 실패했습니다.")
            return None
    
    # FAISS 벡터 스토어 생성
    try:
        from langchain_community.vectorstores import FAISS
        
        print(f"임베딩 생성 중... (총 {len(texts)}개 청크)")
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        
        # 저장 디렉토리 생성
        vector_db_dir = os.path.join(os.path.dirname(__file__), "vector_db")
        ensure_directory(vector_db_dir)
        
        # 벡터 스토어를 디스크에 저장
        vector_store.save_local(os.path.join(vector_db_dir, "faiss_index"))
        print(f"FAISS 벡터 데이터베이스가 생성되었습니다.")
        
        return vector_store
    
    except Exception as e:
        print(f"벡터 스토어 생성 중 오류 발생: {e}")
        return None

def main():
    # 처리된 청크 파일 경로
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks_file = os.path.join(data_dir, "processed_chunks.pkl")
    
    if not os.path.exists(chunks_file):
        print(f"오류: 처리된 청크 파일 '{chunks_file}'이 없습니다.")
        print("먼저 process_mdx.py를 실행하여 MDX 파일을 처리하세요.")
        return
    
    # 처리된 청크 로드
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"청크 {len(chunks)}개 로드 완료")
    
    # OpenAI API 사용 여부 확인
    use_openai = input("OpenAI API를 사용하시겠습니까? (y/n, 기본값: n): ").lower() == 'y'
    
    # 벡터 데이터베이스 생성
    vector_store = create_vector_database(chunks, use_openai=use_openai)
    
    if vector_store:
        # 검색 테스트
        test_query = input("\n테스트 검색어를 입력하세요: ")
        results = vector_store.similarity_search(test_query, k=2)
        
        print("\n검색 결과:")
        for i, doc in enumerate(results):
            print(f"\n결과 {i+1}:")
            print(f"내용: {doc.page_content[:150]}...")
            print(f"메타데이터: {doc.metadata}")

if __name__ == "__main__":
    main()