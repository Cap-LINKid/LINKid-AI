from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import psycopg2
from langchain_postgres import PGVector
from langchain_core.documents import Document

from src.utils.embeddings import get_embedding_model, get_embedding_dimension, embed_query

load_dotenv()


def get_postgres_connection_string() -> str:
    """
    PostgreSQL 연결 문자열 생성
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "linkid_ai")
    user = os.getenv("POSTGRES_USER", "linkid_user")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    if not password:
        raise ValueError("POSTGRES_PASSWORD 환경 변수가 설정되지 않았습니다.")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_vector_store(
    collection_name: Optional[str] = None,
    connection_string: Optional[str] = None
) -> PGVector:
    """
    PGVector 인스턴스 생성 및 반환
    
    Args:
        collection_name: 벡터 저장 컬렉션명 (기본값: 환경 변수 또는 'expert_advice')
        connection_string: PostgreSQL 연결 문자열 (기본값: 환경 변수에서 가져옴)
    
    Returns:
        PGVector 인스턴스
    """
    if collection_name is None:
        collection_name = os.getenv("VECTOR_DB_TABLE", "expert_advice")
    
    if connection_string is None:
        connection_string = get_postgres_connection_string()
    
    embedding_model = get_embedding_model()
    
    return PGVector(
        embeddings=embedding_model,
        connection=connection_string,
        collection_name=collection_name,
        use_jsonb=True,  # 메타데이터를 JSONB로 저장
        create_extension=False,  # 이미 확장이 생성되어 있으므로
    )


def search_expert_advice(
    query: str,
    top_k: int = 5,
    threshold: float = 0.7,
    filters: Optional[Dict[str, Any]] = None,
    table_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    전문가 조언 검색 (expert_advice 테이블 직접 사용)
    
    Args:
        query: 검색 쿼리 텍스트
        top_k: 반환할 결과 수
        threshold: 유사도 임계값 (0.0 ~ 1.0, 높을수록 유사해야 함)
        filters: 메타데이터 필터 딕셔너리
            - advice_type: 조언 타입 필터 (문자열 또는 리스트)
            - pattern_names: 패턴명 배열 필터 (리스트)
            - dpics_labels: DPICS 라벨 배열 필터 (리스트)
            - category: 카테고리 필터
        table_name: 테이블명 (기본값: 'expert_advice')
    
    Returns:
        검색 결과 리스트, 각 항목은 다음 필드를 포함:
        - title: 제목
        - content: 내용
        - source: 출처
        - author: 저자
        - advice_type: 조언 타입
        - relevance_score: 유사도 점수 (0.0 ~ 1.0)
        - metadata: 전체 메타데이터
    """
    if table_name is None:
        table_name = os.getenv("VECTOR_DB_TABLE", "expert_advice")
    
    # 쿼리 텍스트를 임베딩으로 변환
    try:
        query_embedding = embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return []
    
    # PostgreSQL 연결
    try:
        conn_string = get_postgres_connection_string()
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # 필터 조건 생성
        filter_conditions = []
        filter_params = []
        
        if filters:
            # advice_type 필터
            if "advice_type" in filters and filters["advice_type"]:
                advice_types = filters["advice_type"]
                if isinstance(advice_types, str):
                    advice_types = [advice_types]
                if advice_types:
                    placeholders = ",".join(["%s"] * len(advice_types))
                    filter_conditions.append(f"advice_type = ANY(ARRAY[{placeholders}])")
                    filter_params.extend(advice_types)
            
            # category 필터
            if "category" in filters and filters["category"]:
                filter_conditions.append("category = %s")
                filter_params.append(filters["category"])
            
            # pattern_names 필터 (배열에 포함) - OR 조건
            if "pattern_names" in filters and filters["pattern_names"]:
                pattern_names = filters["pattern_names"]
                if isinstance(pattern_names, str):
                    pattern_names = [pattern_names]
                if pattern_names:
                    # 하나라도 포함되면 됨 (OR 조건)
                    pattern_conditions = []
                    for pattern_name in pattern_names:
                        pattern_conditions.append("%s = ANY(pattern_names)")
                        filter_params.append(pattern_name)
                    if pattern_conditions:
                        filter_conditions.append("(" + " OR ".join(pattern_conditions) + ")")
            
            # dpics_labels 필터 (배열에 포함) - OR 조건
            if "dpics_labels" in filters and filters["dpics_labels"]:
                dpics_labels = filters["dpics_labels"]
                if isinstance(dpics_labels, str):
                    dpics_labels = [dpics_labels]
                if dpics_labels:
                    # 하나라도 포함되면 됨 (OR 조건)
                    label_conditions = []
                    for label in dpics_labels:
                        label_conditions.append("%s = ANY(dpics_labels)")
                        filter_params.append(label)
                    if label_conditions:
                        filter_conditions.append("(" + " OR ".join(label_conditions) + ")")
        
        # WHERE 절 구성
        where_clause = ""
        if filter_conditions:
            where_clause = "WHERE " + " AND ".join(filter_conditions)
        
        # SQL 쿼리 실행
        # 파라미터 순서: embedding (SELECT) -> WHERE 절 필터 파라미터 -> embedding (ORDER BY) -> LIMIT
        sql = f"""
        SELECT 
            id,
            title,
            content,
            summary,
            source,
            author,
            advice_type,
            category,
            pattern_names,
            dpics_labels,
            1 - (embedding <=> %s::vector) as similarity_score
        FROM {table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        # 파라미터 순서: embedding_str (SELECT) -> filter_params -> embedding_str (ORDER BY) -> top_k
        params = [embedding_str] + filter_params + [embedding_str, top_k]
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        # 결과 포맷팅
        formatted_results = []
        for row in results:
            (id_val, title, content, summary, source, author, 
             advice_type, category, pattern_names, dpics_labels, similarity_score) = row
            
            if similarity_score >= threshold:
                formatted_results.append({
                    "id": id_val,
                    "title": title or "",
                    "content": content or "",
                    "summary": summary or "",
                    "source": source or "",
                    "author": author or "",
                    "advice_type": advice_type or "",
                    "category": category or "",
                    "pattern_names": pattern_names or [],
                    "dpics_labels": dpics_labels or [],
                    "relevance_score": round(float(similarity_score), 3),
                    "metadata": {
                        "id": id_val,
                        "title": title,
                        "advice_type": advice_type,
                        "category": category,
                        "pattern_names": pattern_names,
                        "dpics_labels": dpics_labels,
                        "source": source,
                        "author": author
                    }
                })
        
        cursor.close()
        conn.close()
        
        # relevance_score 기준으로 정렬 (내림차순)
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return formatted_results
        
    except Exception as e:
        print(f"VectorDB 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


def add_expert_advice_documents(
    documents: List[Document],
    collection_name: Optional[str] = None
) -> List[str]:
    """
    전문가 조언 문서를 VectorDB에 추가
    
    Args:
        documents: 추가할 Document 리스트
        collection_name: 벡터 저장 컬렉션명
    
    Returns:
        추가된 문서의 ID 리스트
    """
    vector_store = get_vector_store(collection_name=collection_name)
    
    try:
        ids = vector_store.add_documents(documents)
        return ids
    except Exception as e:
        print(f"VectorDB 문서 추가 오류: {e}")
        raise


def delete_expert_advice_by_ids(
    ids: List[str],
    collection_name: Optional[str] = None
) -> bool:
    """
    ID로 전문가 조언 문서 삭제
    
    Args:
        ids: 삭제할 문서 ID 리스트
        collection_name: 벡터 저장 컬렉션명
    
    Returns:
        성공 여부
    """
    vector_store = get_vector_store(collection_name=collection_name)
    
    try:
        vector_store.delete(ids)
        return True
    except Exception as e:
        print(f"VectorDB 문서 삭제 오류: {e}")
        return False

