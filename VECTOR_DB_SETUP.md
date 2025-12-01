# VectorDB 인프라 구축 가이드

## 1. PostgreSQL + pgvector 설치

### macOS (Homebrew)
```bash
# PostgreSQL 설치
brew install postgresql@15

# PostgreSQL 시작
brew services start postgresql@15

# pgvector 확장 설치
brew install pgvector
```

### Docker 사용 (권장)
```bash
# pgvector가 포함된 PostgreSQL 이미지 사용
docker run -d \
  --name linkid-postgres \
  -e POSTGRES_USER=linkid_user \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=linkid_ai \
  -p 5432:5432 \
  pgvector/pgvector:pg15
```

### Linux (Ubuntu/Debian)
```bash
# PostgreSQL 설치
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-contrib-15

# pgvector 설치
sudo apt-get install postgresql-15-pgvector
```

## 2. 데이터베이스 설정

### 데이터베이스 및 사용자 생성
```bash
# PostgreSQL 접속
psql -U postgres

# 데이터베이스 생성
CREATE DATABASE linkid_ai;

# 사용자 생성 및 권한 부여
CREATE USER linkid_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE linkid_ai TO linkid_user;

# 데이터베이스 전환
\c linkid_ai

# pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

# 스키마 생성
\i data/sql/vector_db_schema.sql

# 종료
\q
```

또는 SQL 파일 직접 실행:
```bash
psql -U linkid_user -d linkid_ai -f data/sql/vector_db_schema.sql
```

## 3. 환경 변수 설정

`.env` 파일에 다음 변수들을 추가하세요:

```bash
# PostgreSQL 연결 설정
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=linkid_ai
POSTGRES_USER=linkid_user
POSTGRES_PASSWORD=your_password

# VectorDB 설정
VECTOR_DB_TABLE=expert_advice

# Embedding 모델 설정
EMBEDDING_MODEL=openai  # openai | anthropic | google | local
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=1536  # 모델에 따라 변경 (text-embedding-3-small: 1536)

# VectorDB 검색 설정
VECTOR_SEARCH_TOP_K_NEEDS_IMPROVEMENT=2
VECTOR_SEARCH_TOP_K_CHALLENGE=5
VECTOR_SEARCH_THRESHOLD=0.7

# Feature flag
USE_VECTOR_DB=true

# Embedding 모델 API 키 (필요한 경우)
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 4. Python 패키지 설치

```bash
# 가상환경 활성화 (선택사항)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 5. 데이터 인덱싱

전문가 조언 데이터를 VectorDB에 인덱싱:

```bash
# 기본 인덱싱
python scripts/build_vector_index.py

# 기존 데이터 삭제 후 재인덱싱
python scripts/build_vector_index.py --clear

# 다른 데이터 디렉토리 사용
python scripts/build_vector_index.py --data-dir path/to/data

# 다른 테이블명 사용
python scripts/build_vector_index.py --table-name custom_table
```

## 6. 검증

### Python에서 테스트
```python
from src.utils.vector_store import search_expert_advice

# 검색 테스트
results = search_expert_advice(
    query="훈육에서 공포를 주지 않고 단호하게 말하는 법",
    top_k=3,
    filters={
        # 새 스키마 기준 필터 예시
        "type": ["Negative"],       # Type 컬럼 (예: Positive / Negative / Additional 등)
        "age": "유아~초등",          # Age 컬럼
        # pattern_names는 Related_DPICS 문자열에 부분 일치로 매핑됨
        "pattern_names": ["11 Negative", "15 Negative"]
    }
)

for result in results:
    print(f"제목: {result['title']}")
    print(f"카테고리: {result['category']}")
    print(f"연령대: {result['metadata'].get('age', '')}")
    print(f"타입: {result['advice_type']}")
    print(f"키워드: {result['metadata'].get('keyword', '')}")
    print(f"출처(Reference): {result['source']}")
    print(f"유사도: {result['relevance_score']}")
    print("-" * 60)
```

### PostgreSQL에서 직접 확인
```sql
-- 데이터 개수 확인
SELECT COUNT(*) FROM expert_advice;

-- 샘플 데이터 확인 (새 스키마 기준)
SELECT 
    id,
    category,
    age,
    keyword,
    type,
    reference
FROM expert_advice
LIMIT 5;

-- 벡터 차원 확인
SELECT vector_dims(embedding) as dimension
FROM expert_advice
LIMIT 1;
```

## 7. 문제 해결

### pgvector 확장이 활성화되지 않는 경우
```sql
-- 확장 목록 확인
SELECT * FROM pg_extension;

-- pgvector 확장 설치 (수동)
CREATE EXTENSION vector;
```

### Embedding 차원 불일치 오류
- 데이터베이스 스키마의 `embedding vector(1536)` 차원과 실제 모델 차원이 일치하는지 확인
- 다른 모델 사용 시 `data/sql/vector_db_schema.sql`에서 차원 수정 필요

### 연결 오류
- PostgreSQL이 실행 중인지 확인: `brew services list` (macOS) 또는 `systemctl status postgresql` (Linux)
- 방화벽 설정 확인
- 연결 정보 확인: `.env` 파일의 `POSTGRES_*` 변수들

### 검색 결과가 없는 경우
- 데이터가 인덱싱되었는지 확인
- `threshold` 값을 낮춰보기 (예: 0.5)
- 필터 조건이 너무 엄격한지 확인

## 8. 다음 단계

인프라 구축이 완료되면:
1. `key_moments_agent.py`에 VectorDB 통합
2. `coaching_agent.py`에 VectorDB 통합
3. 테스트 및 검증

자세한 내용은 `VECTOR_DB_INTEGRATION.md`를 참고하세요.

