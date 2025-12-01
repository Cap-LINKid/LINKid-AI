#!/usr/bin/env python3
"""
전문가 조언 데이터를 VectorDB에 인덱싱하는 스크립트

사용법:
    python scripts/build_vector_index.py [--data-dir data/expert_advice] [--clear]
    
옵션:
    --data-dir: 전문가 조언 JSON 파일이 있는 디렉토리 (기본값: data/expert_advice)
    --clear: 기존 데이터 삭제 후 재인덱싱
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

from src.utils.vector_store import get_postgres_connection_string
from src.utils.embeddings import embed_texts, get_embedding_dimension

load_dotenv()


def load_expert_advice_json(data_dir: str) -> List[Dict[str, Any]]:
    """
    전문가 조언 JSON 파일들을 로드
    
    Args:
        data_dir: JSON 파일이 있는 디렉토리
    
    Returns:
        전문가 조언 딕셔너리 리스트
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    all_advice = []
    
    # JSON 파일 찾기
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {data_dir}")
    
    for json_file in json_files:
        print(f"로딩 중: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_advice.extend(data)
                elif isinstance(data, dict):
                    all_advice.append(data)
        except Exception as e:
            print(f"파일 로드 오류 ({json_file.name}): {e}")
            continue
    
    print(f"총 {len(all_advice)}개의 전문가 조언을 로드했습니다.")
    return all_advice


def _build_document_from_advice(advice: Dict[str, Any]) -> Dict[str, Any]:
    """
    조언 딕셔너리에서 새 expert_advice 스키마(Category~Reference)에 맞는 필드와
    임베딩용 문서 텍스트를 생성한다.

    - 기존 JSON 스키마(title, content, ...),
      새 CSV/TSV 스키마(Category, Age, Related_DPICS, Keyword, Situation, Type, Advice, Reference)
      를 모두 최대한 지원하도록 키 매핑을 한다.
    """
    category = advice.get("Category") or advice.get("category") or ""
    age = advice.get("Age") or advice.get("age") or ""
    related_dpics = (
        advice.get("Related_DPICS")
        or advice.get("Related_DPICS (관련 패턴)")
        or advice.get("related_dpics")
        or ", ".join(advice.get("dpics_labels", []))
        or ""
    )
    keyword = (
        advice.get("Keyword")
        or advice.get("keyword")
        or ", ".join(advice.get("tags", []))
        or ""
    )
    situation = advice.get("Situation") or advice.get("situation") or ""
    type_ = (
        advice.get("Type")
        or advice.get("type")
        or advice.get("advice_type")
        or ""
    )
    advice_text = (
        advice.get("Advice")
        or advice.get("advice")
        or advice.get("content")
        or ""
    )
    reference = (
        advice.get("Reference")
        or advice.get("reference")
        or advice.get("source")
        or ""
    )

    # 모든 속성을 하나의 문서 텍스트로 합치기
    parts = [
        f"Category: {category}" if category else "",
        f"Age: {age}" if age else "",
        f"Related_DPICS: {related_dpics}" if related_dpics else "",
        f"Keyword: {keyword}" if keyword else "",
        f"Situation: {situation}" if situation else "",
        f"Type: {type_}" if type_ else "",
        f"Advice: {advice_text}" if advice_text else "",
        f"Reference: {reference}" if reference else "",
    ]
    doc_text = "\n".join(p for p in parts if p)

    return {
        "category": category,
        "age": age,
        "related_dpics": related_dpics,
        "keyword": keyword,
        "situation": situation,
        "type": type_,
        "advice_text": advice_text,
        "reference": reference,
        "doc_text": doc_text,
    }


def insert_advice_to_db(
    advice_list: List[Dict[str, Any]],
    table_name: str = "expert_advice"
) -> int:
    """
    전문가 조언을 PostgreSQL의 expert_advice 테이블에 직접 삽입
    
    Args:
        advice_list: 전문가 조언 딕셔너리 리스트
        table_name: 테이블명
    
    Returns:
        삽입된 레코드 수
    """
    conn_string = get_postgres_connection_string()
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    inserted_count = 0
    
    try:
        # 임베딩 생성용 문서와 변환된 레코드 리스트
        documents: List[str] = []
        transformed_rows: List[Dict[str, Any]] = []
        
        for advice in advice_list:
            row = _build_document_from_advice(advice)
            doc_text = row["doc_text"]
            if not doc_text.strip():
                print(f"경고: 유효한 텍스트가 없는 항목을 건너뜁니다: {advice}")
                continue
            documents.append(doc_text)
            transformed_rows.append(row)
        
        if not documents:
            print("삽입할 데이터가 없습니다.")
            return 0
        
        # 배치로 임베딩 생성
        print(f"임베딩 생성 중... (총 {len(documents)}개)")
        embeddings = embed_texts(documents)
        print(f"임베딩 생성 완료 (차원: {len(embeddings[0])})")
        
        # 데이터 삽입
        print(f"\n데이터베이스에 삽입 중...")
        
        for i, (row, embedding) in enumerate(zip(transformed_rows, embeddings)):
            try:
                # 벡터를 PostgreSQL 형식으로 변환
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                # SQL INSERT 쿼리 (새 스키마: Category~Reference + embedding)
                insert_sql = f"""
                INSERT INTO {table_name} (
                    category,
                    age,
                    related_dpics,
                    keyword,
                    situation,
                    type,
                    advice,
                    reference,
                    embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::vector
                )
                """
                
                cursor.execute(
                    insert_sql,
                    (
                        row["category"],
                        row["age"],
                        row["related_dpics"],
                        row["keyword"],
                        row["situation"],
                        row["type"],
                        row["advice_text"],
                        row["reference"],
                        embedding_str,
                    ),
                )
                
                inserted_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"진행 중: {i + 1}/{len(transformed_rows)}")
                    conn.commit()  # 중간 커밋
                    
            except Exception as e:
                print(f"삽입 오류: {e}")
                conn.rollback()
                continue
        
        # 최종 커밋
        conn.commit()
        print(f"\n총 {inserted_count}개의 레코드가 삽입되었습니다.")
        
    except Exception as e:
        conn.rollback()
        print(f"데이터 삽입 오류: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cursor.close()
        conn.close()
    
    return inserted_count


def clear_existing_data(table_name: str = "expert_advice") -> bool:
    """
    기존 데이터 삭제 (주의: 모든 데이터가 삭제됩니다)
    
    Args:
        table_name: 테이블명
    
    Returns:
        성공 여부
    """
    try:
        conn_string = get_postgres_connection_string()
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # 데이터 개수 확인
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"삭제할 데이터 개수: {count}")
        
        # 데이터 삭제
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
        
        print(f"테이블 {table_name}의 모든 데이터를 삭제했습니다.")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"데이터 삭제 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="전문가 조언 데이터를 VectorDB에 인덱싱")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/expert_advice",
        help="전문가 조언 JSON 파일이 있는 디렉토리"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="기존 데이터 삭제 후 재인덱싱"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="테이블명 (기본값: 환경 변수 또는 'expert_advice')"
    )
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("=" * 60)
    print("전문가 조언 VectorDB 인덱싱 시작")
    print("=" * 60)
    
    try:
        advice_list = load_expert_advice_json(args.data_dir)
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)
    
    if not advice_list:
        print("인덱싱할 데이터가 없습니다.")
        sys.exit(1)
    
    # 테이블명 결정
    table_name = args.table_name or os.getenv("VECTOR_DB_TABLE", "expert_advice")
    
    # 기존 데이터 삭제 (옵션)
    if args.clear:
        print("\n기존 데이터 삭제 중...")
        clear_existing_data(table_name)
    
    # Embedding 차원 확인
    embedding_dim = get_embedding_dimension()
    print(f"\nEmbedding 차원: {embedding_dim}")
    print("데이터베이스의 embedding 컬럼 차원과 일치하는지 확인하세요.")
    
    # 데이터베이스 연결 테스트
    try:
        conn_string = get_postgres_connection_string()
        conn = psycopg2.connect(conn_string)
        conn.close()
        print("데이터베이스 연결 성공")
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        print("\n환경 변수를 확인하세요:")
        print("  - POSTGRES_HOST")
        print("  - POSTGRES_PORT")
        print("  - POSTGRES_DB")
        print("  - POSTGRES_USER")
        print("  - POSTGRES_PASSWORD")
        sys.exit(1)
    
    # 데이터 삽입
    print(f"\n테이블 '{table_name}'에 데이터 삽입 중...")
    try:
        inserted_count = insert_advice_to_db(advice_list, table_name=table_name)
        
        print("\n" + "=" * 60)
        print(f"인덱싱 완료! 총 {inserted_count}개의 전문가 조언이 추가되었습니다.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n인덱싱 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

