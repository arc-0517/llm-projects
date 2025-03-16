# vector_database_fixed.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from mdx_processor import process_mdx_files, split_documents

def create_vector_database(chunks, embedding_model_name="gemma3", use_faiss=True):
    """
    청크를 벡터 데이터베이스에 저장합니다.
    
    Args:
        chunks: 청크 목록
        embedding_model_name: Ollama에서 사용할 임베딩 모델 이름
        use_faiss: FAISS 사용 여부 (False면 Chroma 사용)
    
    Returns:
        vector_store: 생성된 벡터 스토어 객체
    """
    # 텍스트와 메타데이터 분리
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Ollama 임베딩 모델 설정 - 디버깅 코드 추가
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    
    # 임베딩 테스트 (Ollama 서버가 실행 중인지, 모델이 올바른지 확인)
    print("임베딩 테스트 수행 중...")
    test_embedding = embeddings.embed_query("테스트 임베딩")
    if test_embedding:
        print(f"임베딩 작동 확인 (벡터 크기: {len(test_embedding)})")
    else:
        print("오류: 임베딩을 생성할 수 없습니다. Ollama 서버가 실행 중이고 모델이 올바른지 확인하세요.")
        return None
    
    print(f"임베딩 생성 중... (총 {len(texts)}개 청크)")
    
    # 벡터 데이터베이스 생성
    if use_faiss:
        # FAISS 벡터 스토어 생성
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        
        # 저장 디렉토리 생성
        os.makedirs("vector_db", exist_ok=True)
        
        # 벡터 스토어를 디스크에 저장
        vector_store.save_local("vector_db/faiss_index")
        print("FAISS 벡터 데이터베이스가 'vector_db/faiss_index'에 저장되었습니다.")
    else:
        # Chroma 벡터 스토어 생성
        vector_store = Chroma.from_texts(
            texts=texts, 
            embedding=embeddings, 
            metadatas=metadatas,
            persist_directory="vector_db/chroma"
        )
        
        # 벡터 스토어를 디스크에 저장
        vector_store.persist()
        print("Chroma 벡터 데이터베이스가 'vector_db/chroma'에 저장되었습니다.")
    
    return vector_store

def load_vector_database(embedding_model_name="gemma3", use_faiss=True):
    """저장된 벡터 데이터베이스를 로드합니다."""
    
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    
    # 임베딩 테스트
    test_embedding = embeddings.embed_query("테스트 임베딩")
    if not test_embedding:
        print("오류: 임베딩을 생성할 수 없습니다. Ollama 서버가 실행 중이고 모델이 올바른지 확인하세요.")
        return None
    
    if use_faiss:
        if os.path.exists("vector_db/faiss_index"):
            vector_store = FAISS.load_local("vector_db/faiss_index", embeddings)
            print("FAISS 벡터 데이터베이스를 로드했습니다.")
            return vector_store
    else:
        if os.path.exists("vector_db/chroma"):
            vector_store = Chroma(persist_directory="vector_db/chroma", embedding_function=embeddings)
            print("Chroma 벡터 데이터베이스를 로드했습니다.")
            return vector_store
    
    print("저장된 벡터 데이터베이스를 찾을 수 없습니다.")
    return None

if __name__ == "__main__":
    # MDX 파일 디렉토리
    mdx_dir = './mdx_files/try-mellerikat'
    
    # 벡터 데이터베이스 생성 여부
    create_new_db = True
    
    # Ollama 서버 실행 확인 안내
    print("중요: 코드를 실행하기 전에 Ollama 서버가 실행되고 있는지 확인하세요.")
    print("Ollama 서버가 실행되지 않았다면, 별도의 터미널에서 'ollama serve' 명령을 실행하세요.")
    
    # 사용 가능한 모델 출력
    print("\n사용 가능한 Ollama 모델 확인:")
    os.system("ollama list")
    
    # 모델 이름 선택
    model_name = input("\n사용할 임베딩 모델 이름을 입력하세요(기본값: gemma): ")
    if not model_name:
        model_name = "gemma"
    
    if create_new_db:
        # MDX 파일 처리
        documents = process_mdx_files(mdx_dir)
        print(f"처리된 문서 수: {len(documents)}")
        
        # 문서를 청크로 분할
        chunks = split_documents(documents)
        print(f"생성된 청크 수: {len(chunks)}")
        
        # 벡터 데이터베이스 생성
        vector_store = create_vector_database(chunks, embedding_model_name=model_name, use_faiss=True)
    else:
        # 기존 벡터 데이터베이스 로드
        vector_store = load_vector_database(embedding_model_name=model_name, use_faiss=True)
    
    # 임베딩 기능 테스트
    if vector_store:
        test_query = "mellerikat 체험 방법은 어떻게 되나요?"
        results = vector_store.similarity_search(test_query, k=2)
        
        print("\n검색 테스트:")
        print(f"쿼리: {test_query}")
        print(f"결과 수: {len(results)}")
        
        for i, doc in enumerate(results):
            print(f"\n결과 {i+1}:")
            print(f"내용: {doc.page_content[:150]}...")
            print(f"메타데이터: {doc.metadata}")