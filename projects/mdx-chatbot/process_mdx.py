"""
MDX 파일을 처리하는 스크립트
"""

import os
import sys

# 공통 모듈 경로 추가 (상대 경로 임포트)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from common.data_processors.mdx_processor import process_mdx_files, split_documents
from common.utils.file_utils import ensure_directory

def main():
    # MDX 파일 디렉토리 설정
    mdx_dir = input("MDX 파일이 있는 디렉토리 경로를 입력하세요: ")
    
    if not os.path.exists(mdx_dir):
        print(f"오류: 디렉토리 '{mdx_dir}'가 존재하지 않습니다.")
        return
    
    # 결과 저장 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    ensure_directory(output_dir)
    
    # MDX 파일 처리
    print(f"MDX 파일 처리 중...")
    documents = process_mdx_files(mdx_dir)
    print(f"처리된 문서 수: {len(documents)}")
    
    # 문서를 청크로 분할
    print(f"문서를 청크로 분할 중...")
    chunks = split_documents(documents)
    print(f"생성된 청크 수: {len(chunks)}")
    
    # 처리 결과 출력
    if chunks:
        print("\n첫 번째 청크 내용 샘플:")
        print(f"메타데이터: {chunks[0]['metadata']}")
        print(f"내용: {chunks[0]['content'][:150]}...")
    
    # 결과를 중간 파일로 저장
    import pickle
    output_file = os.path.join(output_dir, "processed_chunks.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"\n처리된 청크가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()