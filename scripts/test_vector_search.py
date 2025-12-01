#!/usr/bin/env python3
"""
PostgreSQL에서 벡터 검색을 테스트하는 스크립트

사용법:
    python scripts/test_vector_search.py [--query "검색 쿼리"]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
import numpy as np

from src.utils.embeddings import embed_query
from src.utils.vector_store import get_postgres_connection_string, search_expert_advice

load_dotenv()


def test_direct_sql_search(query_text: str, top_k: int = 5):
    """
    PostgreSQL에서 직접 SQL로 벡터 검색 테스트
    """
    print("=" * 60)
    print("PostgreSQL 직접 SQL 검색 테스트")
    print("=" * 60)
    
    # 쿼리 텍스트를 임베딩으로 변환
    print(f"\n쿼리: {query_text}")
    print("임베딩 생성 중...")
    
    try:
        query_embedding = embed_query(query_text)
        embedding_dim = len(query_embedding)
        print(f"임베딩 차원: {embedding_dim}")
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return
    
    # PostgreSQL 연결
    try:
        conn_string = get_postgres_connection_string()
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # 벡터를 PostgreSQL 형식으로 변환
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # SQL 쿼리 실행
        sql = f"""
        SELECT 
            id,
            category,
            age,
            keyword,
            type,
            reference,
            1 - (embedding <=> %s::vector) as similarity_score
        FROM expert_advice
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        print("\nSQL 쿼리 실행 중...")
        cursor.execute(sql, (embedding_str, embedding_str, top_k))
        results = cursor.fetchall()
        
        # 결과 출력
        print(f"\n검색 결과 (상위 {len(results)}개):")
        print("-" * 60)
        
        for row in results:
            id_val, category, age, keyword, type_value, reference, similarity = row
            print(f"\nID: {id_val}")
            print(f"카테고리: {category}")
            print(f"연령대(Age): {age}")
            print(f"키워드: {keyword}")
            print(f"타입(Type): {type_value}")
            print(f"출처(Reference): {reference}")
            print(f"유사도: {similarity:.4f}")
            print("-" * 60)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"SQL 실행 오류: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_search(query_text: str, top_k: int = 5):
    """
    LangChain 유틸리티를 사용한 검색 테스트
    """
    print("\n" + "=" * 60)
    print("LangChain 유틸리티 검색 테스트")
    print("=" * 60)
    
    print(f"\n쿼리: {query_text}")
    print("검색 중...")
    
    try:
        results = search_expert_advice(
            query=query_text,
            top_k=top_k,
            threshold=0.0  # 모든 결과 보기
        )
        
        print(f"\n검색 결과 (상위 {len(results)}개):")
        print("-" * 60)
        
        for result in results:
            print(f"\n제목: {result['title']}")
            print(f"카테고리: {result['category']}")
            print(f"연령대(Age): {result['metadata'].get('age', '')}")
            print(f"타입(Type): {result['advice_type']}")
            print(f"키워드: {result['metadata'].get('keyword', '')}")
            print(f"출처(Reference): {result['source']}")
            print(f"유사도: {result['relevance_score']:.4f}")
            print(f"내용 미리보기: {result['content'][:100]}...")
            print("-" * 60)
        
    except Exception as e:
        print(f"검색 오류: {e}")
        import traceback
        traceback.print_exc()


def test_filtered_search(query_text: str):
    """
    필터링과 함께 검색 테스트
    """
    print("\n" + "=" * 60)
    print("필터링 검색 테스트")
    print("=" * 60)
    
    # 타입(Type)으로 필터링 예시
    print("\n[테스트 1] Type='Negative' 필터 검색")
    try:
        results = search_expert_advice(
            query=query_text,
            top_k=3,
            filters={
                "type": "Negative"
            }
        )
        
        for result in results:
            print(f"- {result['title']} (Type: {result['advice_type']}, 유사도: {result['relevance_score']:.4f})")
    except Exception as e:
        print(f"오류: {e}")


def check_database_status():
    """
    데이터베이스 상태 확인
    """
    print("=" * 60)
    print("데이터베이스 상태 확인")
    print("=" * 60)
    
    try:
        conn_string = get_postgres_connection_string()
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # 데이터 개수 확인
        cursor.execute("SELECT COUNT(*) FROM expert_advice")
        count = cursor.fetchone()[0]
        print(f"\n전문가 조언 데이터 개수: {count}")
        
        # type별 개수 (새 스키마 기준)
        try:
            cursor.execute("""
                SELECT type, COUNT(*) 
                FROM expert_advice 
                GROUP BY type
                ORDER BY COUNT(*) DESC
            """)
            print("\nType별 개수:")
            for row in cursor.fetchall():
                print(f"  - {row[0] or '(NULL)'}: {row[1]}개")
        except Exception as e:
            print("\nType별 개수를 가져오는 데 실패했습니다:", e)
        
        # 벡터 차원 확인 (pgvector의 vector_dims 함수 사용)
        try:
            cursor.execute("""
                SELECT vector_dims(embedding) as dim
                FROM expert_advice 
                LIMIT 1
            """)
            dim = cursor.fetchone()
            if dim and dim[0]:
                print(f"\n임베딩 차원: {dim[0]}")
        except Exception as e:
            # 차원 조회 실패는 치명적이지 않으므로 경고만 출력
            print("\n임베딩 차원 정보를 가져오는 데 실패했습니다:", e)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"데이터베이스 확인 오류: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL 벡터 검색 테스트")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="검색 쿼리 텍스트 (미입력 시 인터랙티브 모드)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="반환할 결과 수"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="인터랙티브 모드로 여러 쿼리를 연속 검색"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="데이터베이스 상태만 확인"
    )
    
    args = parser.parse_args()
    
    # 데이터베이스 상태 확인
    check_database_status()
    
    if args.check_only:
        return

    # 쿼리 입력 방식 결정
    if args.interactive or not args.query:
        # 인터랙티브 모드
        print("\n인터랙티브 모드로 진입합니다.")
        print("검색 쿼리를 입력하면 직접 SQL / LangChain / 필터 검색을 순서대로 실행합니다.")
        print("종료하려면 빈 줄을 입력하거나 'q', 'quit', 'exit' 를 입력하세요.")

        while True:
            query = input("\n검색 쿼리를 입력하세요: ").strip()
            if not query or query.lower() in {"q", "quit", "exit"}:
                print("\n인터랙티브 모드를 종료합니다.")
                break

            # 직접 SQL 검색 테스트
            test_direct_sql_search(query, args.top_k)

            # LangChain 유틸리티 검색 테스트
            test_langchain_search(query, args.top_k)

            # 필터링 검색 테스트
            test_filtered_search(query)
    else:
        # 단일 쿼리 모드 (기존 동작)
        # 직접 SQL 검색 테스트
        test_direct_sql_search(args.query, args.top_k)

        # LangChain 유틸리티 검색 테스트
        test_langchain_search(args.query, args.top_k)

        # 필터링 검색 테스트
        test_filtered_search(args.query)
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

