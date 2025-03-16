# content_focused_mdx_processor.py

import os
import re
import frontmatter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_content(mdx_content):
    """MDX 파일에서 실제 사용자에게 보이는 텍스트 콘텐츠만 추출"""
    
    # 스타일 태그 내부 CSS 코드 제거
    mdx_content = re.sub(r'<style>.*?</style>', '', mdx_content, flags=re.DOTALL)
    
    # import 문 제거
    mdx_content = re.sub(r'import.*?;', '', mdx_content, flags=re.DOTALL)
    
    # export default 컴포넌트 함수 블록 시작 부분 제거
    mdx_content = re.sub(r'export\s+default\s+function.*?{', '', mdx_content, flags=re.DOTALL | re.MULTILINE, count=1)
    
    # 컴포넌트 선언 부분 제거
    mdx_content = re.sub(r'export\s+const\s+\w+\s*=\s*\(\)\s*=>\s*{', '', mdx_content, flags=re.DOTALL)
    
    # 마지막 닫는 중괄호 제거
    mdx_content = re.sub(r'};\s*$', '', mdx_content, flags=re.DOTALL)
    
    # HTML/JSX 태그를 제거하되 그 내용은 보존
    def remove_tags_keep_content(match):
        # 태그 내부 텍스트 보존
        inner_content = match.group(1) if match.group(1) else ''
        return ' ' + inner_content + ' '
    
    # 복잡한 HTML/JSX 태그 제거하면서 내부 텍스트 보존 (여러 줄 태그)
    mdx_content = re.sub(r'<[^>]*>(.*?)</[^>]*>', remove_tags_keep_content, mdx_content, flags=re.DOTALL)
    
    # 단일 태그 제거
    mdx_content = re.sub(r'<[^>]*?/>', '', mdx_content)
    
    # 남은 HTML/JSX 태그들 제거
    mdx_content = re.sub(r'<[^>]*>', '', mdx_content)
    
    # 마크다운 헤더, 리스트 등 보존
    
    # 여러 줄 공백 정리
    mdx_content = re.sub(r'\n\s*\n', '\n\n', mdx_content)
    
    # 남은 중괄호 정리
    mdx_content = re.sub(r'[{}]', '', mdx_content)
    
    # 특수 JSX 문법 정리
    mdx_content = re.sub(r'{\s*\/\*.*?\*\/\s*}', '', mdx_content, flags=re.DOTALL)  # 주석 제거
    
    # 연속된 공백 제거
    mdx_content = re.sub(r'\s+', ' ', mdx_content)
    
    # 문장 단위로 정리
    lines = []
    for line in mdx_content.split('\n'):
        line = line.strip()
        if line and not line.startswith('//') and not line.startswith('/*'):
            lines.append(line)
    
    return '\n'.join(lines)

def process_mdx_files(directory):
    documents = []
    
    # 디렉토리 내의 모든 MDX 파일 처리
    for filename in os.listdir(directory):
        if filename.endswith('.mdx'):
            file_path = os.path.join(directory, filename)
            
            # MDX 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as file:
                # frontmatter와 콘텐츠 분리
                post = frontmatter.load(file)
                
                # 메타데이터와 콘텐츠 추출
                metadata = dict(post.metadata) if post.metadata else {}
                raw_content = post.content
                
                # 실제 텍스트 콘텐츠만 추출
                cleaned_content = extract_text_content(raw_content)
                
                # 메타데이터에 파일 이름 추가 (출처 추적용)
                metadata['source'] = filename
                metadata['file_path'] = file_path
                
                # 실제 내용이 있는 경우만 추가
                if cleaned_content.strip():
                    documents.append({
                        'content': cleaned_content,
                        'metadata': metadata
                    })
    
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """텍스트를 검색에 효율적인 크기의 청크로 분할"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
    )
    
    chunks = []
    for doc in documents:
        if not doc['content'].strip():  # 빈 콘텐츠 건너뛰기
            continue
            
        texts = text_splitter.split_text(doc['content'])
        
        for i, text in enumerate(texts):
            # 내용이 의미 있는 경우만 청크로 추가
            if len(text.strip()) > 50:  # 최소 50자 이상인 경우만 의미 있는 컨텐츠로 간주
                # 각 청크에 원본 메타데이터 복사 및 청크 관련 메타 추가
                chunk_metadata = doc['metadata'].copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['chunk_of'] = len(texts)
                
                chunks.append({
                    'content': text,
                    'metadata': chunk_metadata
                })
    
    return chunks

if __name__ == "__main__":
    mdx_dir = './mdx_files/try-mellerikat'
    
    # MDX 파일 처리
    documents = process_mdx_files(mdx_dir)
    print(f"처리된 문서 수: {len(documents)}")
    
    # 문서를 청크로 분할
    chunks = split_documents(documents)
    print(f"생성된 청크 수: {len(chunks)}")
    
    # 청크 확인
    if chunks:
        print("\n첫 번째 청크 내용 샘플:")
        print(f"메타데이터: {chunks[0]['metadata']}")
        print(f"내용: {chunks[0]['content']}")