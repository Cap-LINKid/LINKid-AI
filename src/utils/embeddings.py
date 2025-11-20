from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def get_embedding_model():
    """
    Embedding 모델 반환
    환경 변수에 따라 OpenAI, Anthropic, Google, 또는 로컬 모델 사용
    """
    provider = os.getenv("EMBEDDING_MODEL", "openai").lower()
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
    
    elif provider == "anthropic":
        # Anthropic은 embedding 모델이 없으므로 OpenAI 사용 권장
        # 필요시 다른 모델로 대체
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Anthropic은 embedding 모델이 없습니다. OpenAI를 사용하거나 다른 모델을 설정하세요.")
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
    
    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
    
    elif provider == "local":
        # 로컬 모델 사용 (sentence-transformers 등)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    else:
        raise ValueError(f"지원하지 않는 embedding provider: {provider}")


def get_embedding_dimension() -> int:
    """
    현재 설정된 embedding 모델의 차원 수 반환
    """
    provider = os.getenv("EMBEDDING_MODEL", "openai").lower()
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    # OpenAI 모델 차원
    if provider == "openai":
        if "text-embedding-3-small" in model_name:
            return 1536
        elif "text-embedding-3-large" in model_name:
            return 3072
        elif "text-embedding-ada-002" in model_name:
            return 1536
        else:
            return 1536  # 기본값
    
    # Google 모델 차원
    elif provider == "google":
        return 768  # Google embedding 모델 기본 차원
    
    # 로컬 모델 차원
    elif provider == "local":
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if "MiniLM-L12" in model_name:
            return 384
        elif "MiniLM-L6" in model_name:
            return 384
        else:
            return 768  # 기본값
    
    # 기본값
    return 1536


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    텍스트 리스트를 embedding으로 변환
    
    Args:
        texts: 임베딩할 텍스트 리스트
    
    Returns:
        임베딩 벡터 리스트
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    단일 쿼리 텍스트를 embedding으로 변환
    
    Args:
        query: 임베딩할 쿼리 텍스트
    
    Returns:
        임베딩 벡터
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(query)

