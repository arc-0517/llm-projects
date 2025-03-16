"""
파일 처리를 위한 유틸리티 함수
"""

import os
from typing import List, Dict, Any, Optional

def ensure_directory(path: str) -> str:
    """디렉토리가 존재하는지 확인하고, 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def list_files(directory: str, extension: str = None) -> List[str]:
    """지정된 디렉토리에서 특정 확장자의 파일 목록 반환"""
    if not os.path.exists(directory):
        return []
    
    if extension:
        return [f for f in os.listdir(directory) if f.endswith(extension)]
    return os.listdir(directory)
EOL