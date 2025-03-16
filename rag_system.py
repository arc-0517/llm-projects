# rag_system.py - 수정된 버전

import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def create_rag_chatbot(model_name="llama3:latest", verbose=True):
    """Ollama 모델을 사용한 RAG 챗봇 생성"""
    
    # 임베딩 모델 설정
    embeddings = OllamaEmbeddings(model=model_name)
    
    # 벡터 스토어 로드
    if os.path.exists("vector_db/faiss_index"):
        vector_store = FAISS.load_local("vector_db/faiss_index", 
                                        embeddings,
                                        allow_dangerous_deserialization=True)
        print("FAISS 벡터 데이터베이스를 로드했습니다.")
    else:
        print("오류: 벡터 데이터베이스를 찾을 수 없습니다.")
        return None
    
    # 검색기 설정
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 검색할 문서 청크 수
    )
    
    # LLM 설정
    llm = Ollama(
        model=model_name,
        temperature=0.2,
    )
    
    # 프롬프트 템플릿 설정
    qa_prompt_template = """
    당신은 mellerikat 제품 문서에 기반한 지식을 갖고 있는 질의응답 챗봇입니다.
    다음 문맥 정보가 주어집니다:
    
    {context}
    
    이 정보를 바탕으로 아래 질문에 정확하고 도움이 되는 답변을 제공하세요.
    반드시 한국어(한글)로 답변해주세요.
    정보가 없는 경우에는 정직하게 모른다고 답변하고, 추측하지 마세요.
    
    질문: {question}
    
    답변:
    """
    
    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question"]
    )
    
    # 중요 수정: 메모리는 사용하지 않는 방식으로 변경
    # 대화형 검색 체인 생성 (메모리 없이)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,  # 메모리 비활성화
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=verbose
    )
    
    return qa_chain

def chat_with_bot(qa_chain):
    """챗봇과 대화하는 간단한 CLI 인터페이스"""
    
    print("\n=== mellerikat 메뉴얼 챗봇 ===")
    print("질문을 입력하세요. 종료하려면 'exit' 또는 'quit'를 입력하세요.\n")
    
    # 대화 기록을 수동으로 관리
    chat_history = []
    
    while True:
        user_input = input("\n질문: ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break
        
        # 챗봇에 질문하고 응답 받기
        try:
            # invoke 메서드 사용 (최신 방식)
            result = qa_chain.invoke({
                "question": user_input,
                "chat_history": chat_history  # 수동으로 대화 기록 전달
            })
            
            # 응답 출력
            answer = result["answer"]
            sources = result.get("source_documents", [])
            
            print(f"\n답변: {answer}")
            
            # 대화 기록 업데이트
            chat_history.append((user_input, answer))
            
            # 출처 정보 출력
            if sources:
                print("\n참고 문서:")
                for i, doc in enumerate(sources):
                    source = doc.metadata.get('source', '알 수 없는 출처')
                    print(f"  {i+1}. {source}")
        
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해 주세요.")

# 메인 코드는 동일하게 유지
if __name__ == "__main__":
    # 사용 가능한 Ollama 모델 출력
    print("사용 가능한 Ollama 모델 확인:")
    os.system("ollama list")
    
    # 모델 이름 선택
    model_name = input("\n사용할 모델 이름을 입력하세요(기본값: gemma): ")
    if not model_name:
        model_name = "gemma"
    
    # RAG 챗봇 생성
    qa_chain = create_rag_chatbot(model_name=model_name)
    
    if qa_chain:
        # 챗봇과 대화
        chat_with_bot(qa_chain)
    else:
        print("챗봇을 초기화할 수 없습니다.")