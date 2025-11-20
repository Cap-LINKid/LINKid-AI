# pgVector 기반 VectorDB 도입 전략

## 1. 기술 스택

### 필수 구성요소
- **PostgreSQL**: 12+ (pgvector 확장 지원)
- **pgvector 확장**: 벡터 검색 기능
- **LangChain PGVector**: LangChain 통합
- **Embedding 모델**: OpenAI/Anthropic/Google API 또는 로컬 모델

### 의존성 추가
```bash
# requirements.txt에 추가 필요
psycopg2-binary>=2.9.9  # PostgreSQL 드라이버
langchain-postgres>=0.0.6  # LangChain PGVector 통합 (또는 직접 구현)
```

## 2. 데이터베이스 스키마 설계

### 전문가 조언 테이블 (expert_advice)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE expert_advice (
    id SERIAL PRIMARY KEY,
    
    -- 기본 정보
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,  -- 벡터화할 메인 콘텐츠
    summary TEXT,  -- 간단 요약
    
    -- 카테고리 및 분류
    advice_type VARCHAR(50) NOT NULL,  -- 'coaching', 'challenge_guide', 'qa_tip', 'pattern_advice'
    category VARCHAR(100),  -- '긍정강화', '공감', '지시' 등
    
    -- 메타데이터 (검색 필터링용)
    pattern_names TEXT[],  -- ['긍정기회놓치기', '명령과제시'] 등
    dpics_labels TEXT[],  -- ['PR', 'RD', 'CMD'] 등
    interaction_stages TEXT[],  -- ['공감적 협력', '개선이 필요한 상호작용'] 등
    severity_levels TEXT[],  -- ['low', 'medium', 'high']
    
    -- 관련 정보
    related_challenges TEXT[],  -- 관련 챌린지 ID 또는 이름
    tags TEXT[],  -- ['칭찬', '긍정강화', '공감'] 등
    
    -- 벡터 임베딩
    embedding vector(1536),  -- OpenAI text-embedding-3-small 기준 (다른 모델은 차원 변경)
    
    -- 메타데이터
    source VARCHAR(200),  -- 출처 (예: 'DPICS 가이드', '전문가 조언')
    author VARCHAR(200),  -- 저자/연구자 (예: '오은영 박사 연구', 'DPICS 연구팀')
    priority INTEGER DEFAULT 0,  -- 우선순위 (높을수록 우선)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 인덱스
    CONSTRAINT valid_advice_type CHECK (advice_type IN ('coaching', 'challenge_guide', 'qa_tip', 'pattern_advice', 'general'))
);

-- 벡터 검색 성능을 위한 인덱스
CREATE INDEX ON expert_advice USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- 데이터 크기에 따라 조정

-- 메타데이터 필터링을 위한 인덱스
CREATE INDEX idx_pattern_names ON expert_advice USING GIN (pattern_names);
CREATE INDEX idx_dpics_labels ON expert_advice USING GIN (dpics_labels);
CREATE INDEX idx_advice_type ON expert_advice (advice_type);
CREATE INDEX idx_category ON expert_advice (category);
```

## 3. 전문가 조언 데이터 구성

### 3.1 데이터 구조 (JSON 예시)

```json
{
  "id": "advice_001",
  "title": "긍정기회놓치기 패턴 개선 가이드",
  "content": "아이가 긍정적인 행동을 보였을 때, 즉시 칭찬하는 것이 중요합니다. 예를 들어, 아이가 장난감을 정리했다면 '와, 장난감을 깔끔하게 정리했구나! 정말 잘했어!'라고 구체적으로 칭찬하세요. 칭찬을 놓치면 아이는 자신의 긍정적 행동이 인정받지 못한다고 느낄 수 있습니다. 긍정적 행동을 즉시 인정하고 칭찬하는 습관을 기르면, 아이의 자존감 향상과 긍정적 행동의 증가로 이어집니다.",
  "summary": "아이의 긍정적 행동에 즉시 구체적으로 칭찬하기",
  "advice_type": "pattern_advice",
  "category": "긍정강화",
  "pattern_names": ["긍정기회놓치기"],
  "dpics_labels": ["PR", "BD"],
  "interaction_stages": ["개선이 필요한 상호작용", "균형잡힌 대화"],
  "severity_levels": ["medium"],
  "related_challenges": ["긍정강화 챌린지"],
  "tags": ["칭찬", "긍정강화", "즉시반응"],
  "source": "DPICS 가이드",
  "author": "오은영 박사 연구",
  "priority": 5
}
```

### 3.2 데이터 카테고리별 구성

#### A. 패턴별 조언 (pattern_advice)

**긍정기회놓치기**
- 제목: "긍정기회놓치기 패턴 개선 가이드"
- 내용: 아이의 긍정적 행동 즉시 칭찬 방법
- 패턴: ["긍정기회놓치기"]
- DPICS 라벨: ["PR", "BD"]

**명령과제시**
- 제목: "명령과제시 패턴 개선: 선택권 제공하기"
- 내용: 명령 대신 질문이나 선택권을 제공하는 방법
- 패턴: ["명령과제시"]
- DPICS 라벨: ["CMD", "Q"]

**비판적반응**
- 제목: "비판적반응 줄이기: 공감적 대응하기"
- 내용: 비판 대신 공감과 이해를 표현하는 방법
- 패턴: ["비판적반응"]
- DPICS 라벨: ["NEG", "RD"]

**공감부족**
- 제목: "공감 표현하기: 아이의 감정 인정하기"
- 내용: 아이의 감정을 반영하고 공감하는 방법
- 패턴: ["공감부족"]
- DPICS 라벨: ["RD"]

**반영부족**
- 제목: "반영적 진술 사용하기: 아이의 말 되돌려주기"
- 내용: 반영적 진술을 통해 아이의 말을 이해하고 있음을 보여주는 방법
- 패턴: ["반영부족"]
- DPICS 라벨: ["RD"]

#### B. DPICS 라벨별 조언 (coaching)

**PR (칭찬)**
- 제목: "효과적인 칭찬 방법"
- 내용: 구체적이고 즉시적인 칭찬의 중요성과 방법
- DPICS 라벨: ["PR"]

**RD (반영적 진술)**
- 제목: "반영적 진술로 공감 표현하기"
- 내용: 아이의 말과 감정을 반영하여 공감을 표현하는 방법
- DPICS 라벨: ["RD"]

**CMD (지시)**
- 제목: "지시 대신 질문하기"
- 내용: 명령 대신 질문이나 선택권을 제공하는 방법
- DPICS 라벨: ["CMD"]

**Q (질문)**
- 제목: "열린 질문으로 대화 이끌기"
- 내용: 아이의 생각과 감정을 이끌어내는 질문 방법
- DPICS 라벨: ["Q"]

**NEG (부정적 발화)**
- 제목: "부정적 발화 줄이기"
- 내용: 비판이나 부정적 표현을 줄이고 긍정적으로 전환하는 방법
- DPICS 라벨: ["NEG"]

#### C. 상호작용 단계별 조언 (coaching)

**공감적 협력**
- 제목: "공감적 협력 단계 유지하기"
- 내용: 높은 공감과 협력 수준을 유지하는 방법
- 상호작용 단계: ["공감적 협력"]

**개선이 필요한 상호작용**
- 제목: "개선이 필요한 상호작용 개선하기"
- 내용: 부정적 패턴을 줄이고 긍정적 상호작용으로 전환하는 방법
- 상호작용 단계: ["개선이 필요한 상호작용"]

**지시적 상호작용**
- 제목: "지시적 상호작용에서 공감으로 전환하기"
- 내용: 지시 중심에서 공감 중심으로 대화 스타일을 바꾸는 방법
- 상호작용 단계: ["지시적 상호작용"]

#### D. 챌린지 가이드 (challenge_guide)

**긍정강화 챌린지**
- 제목: "7일 긍정강화 챌린지 가이드"
- 내용: 일주일 동안 아이의 긍정적 행동을 즉시 칭찬하는 챌린지 가이드
- 관련 챌린지: ["긍정강화 챌린지"]
- 패턴: ["긍정기회놓치기"]

**공감 표현 챌린지**
- 제목: "7일 공감 표현 챌린지 가이드"
- 내용: 일주일 동안 반영적 진술과 공감 표현을 늘리는 챌린지 가이드
- 관련 챌린지: ["공감 표현 챌린지"]
- 패턴: ["공감부족", "반영부족"]

**질문하기 챌린지**
- 제목: "7일 질문하기 챌린지 가이드"
- 내용: 일주일 동안 명령 대신 질문으로 대화를 이끄는 챌린지 가이드
- 관련 챌린지: ["질문하기 챌린지"]
- 패턴: ["명령과제시"]

#### E. QA 팁 (qa_tip)

**자주 묻는 질문들**
- Q: "아이가 말을 안 들을 때 어떻게 해야 하나요?"
- A: "명령 대신 질문이나 선택권을 제공해보세요. 예를 들어 '지금 정리할까?' 대신 '지금 정리할까, 아니면 5분 후에 할까?'라고 물어보세요."

- Q: "칭찬을 자주 해야 하나요?"
- A: "네, 특히 아이의 긍정적 행동을 즉시 구체적으로 칭찬하는 것이 중요합니다. '잘했어'보다는 '장난감을 정리해서 방이 깔끔해졌구나, 정말 잘했어!'처럼 구체적으로 칭찬하세요."

- Q: "아이가 화가 났을 때 어떻게 대응해야 하나요?"
- A: "먼저 아이의 감정을 인정하고 반영해주세요. '화가 났구나'라고 말하며 아이의 감정을 이해하고 있음을 보여주세요. 그 후에 문제를 해결하는 방법을 함께 찾아보세요."

### 3.3 데이터 수집 전략

1. **초기 데이터 소스**
   - DPICS 가이드 문서
   - 부모-자녀 상호작용 전문가 조언
   - 기존 시스템에서 생성된 성공 사례
   - 학술 논문 및 연구 자료

2. **데이터 확장**
   - 사용자 피드백 기반 개선
   - 새로운 패턴 발견 시 조언 추가
   - 챌린지 결과 기반 개선된 가이드 추가

3. **데이터 품질 관리**
   - 전문가 검토
   - 실제 사용 사례 검증
   - 정기적인 업데이트

## 4. 검색 전략

### 4.1 쿼리 생성 전략

**coaching_agent에서 사용:**
```python
# 패턴 기반 쿼리
query = f"{pattern_name} 패턴 개선 방법"

# 스타일 분석 기반 쿼리
query = f"{dpics_label} 사용법과 {interaction_stage} 단계 조언"

# 복합 쿼리
query = f"{pattern_name} 패턴에서 {dpics_label} 라벨 사용하기"
```

**challenge_agent에서 사용:**
```python
# 챌린지 스펙 기반 쿼리
query = f"{challenge_title} 챌린지 가이드와 평가 기준"

# 패턴 기반 챌린지 조언
query = f"{pattern_name} 패턴 개선 챌린지"
```

### 4.2 검색 파라미터

- **top_k**: 3-5개 (너무 많으면 노이즈, 너무 적으면 부족)
- **유사도 임계값**: 0.7 이상 (모델에 따라 조정)
- **메타데이터 필터링**: 
  - 패턴명 일치
  - DPICS 라벨 일치
  - 상호작용 단계 일치
  - 심각도 레벨 필터링

### 4.3 하이브리드 검색 (선택사항)

벡터 검색 + 키워드 검색 조합:
- 벡터 검색: 의미적 유사도
- 키워드 검색: 정확한 용어 매칭 (PostgreSQL full-text search)
- 가중치 조합으로 최종 점수 계산

## 5. 구현 단계

### Phase 1: 인프라 구축
1. PostgreSQL + pgvector 설치 및 설정
2. 데이터베이스 스키마 생성
3. LangChain PGVector 통합 유틸리티 작성
4. Embedding 모델 설정

### Phase 2: 데이터 준비
1. 전문가 조언 데이터 수집 및 정리
2. JSON 형식으로 데이터 구조화
3. 초기 데이터 인덱싱 스크립트 작성
4. Embedding 생성 및 DB 저장

### Phase 3: RAG 통합
1. `coaching_agent.py`에 VectorDB 검색 통합
2. `challenge_agent.py`에 VectorDB 검색 통합
3. 검색 결과를 프롬프트 컨텍스트에 추가
4. Feature flag로 점진적 활성화

### Phase 4: 최적화
1. 검색 품질 개선 (쿼리 리라이팅)
2. 성능 최적화 (인덱스 튜닝)
3. 메타데이터 필터링 정교화
4. 사용자 피드백 기반 개선

## 6. 환경 변수

```bash
# PostgreSQL 연결
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=linkid_ai
POSTGRES_USER=linkid_user
POSTGRES_PASSWORD=your_password

# VectorDB 설정
VECTOR_DB_TABLE=expert_advice
EMBEDDING_MODEL=openai  # openai | anthropic | google | local
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=1536  # 모델에 따라 변경
VECTOR_SEARCH_TOP_K=5
VECTOR_SEARCH_THRESHOLD=0.7
USE_VECTOR_DB=true  # Feature flag
```

## 7. 파일 구조

```
src/
├── utils/
│   ├── vector_store.py      # NEW: pgVector 유틸리티
│   └── embeddings.py        # NEW: Embedding 모델 래퍼
├── vs/
│   ├── ddl.py               # 기존
│   └── expert_knowledge.py  # NEW: 전문가 조언 데이터 로더
data/
└── expert_advice/           # NEW: 전문가 조언 원본 데이터
    ├── pattern_advice.json
    ├── coaching_tips.json
    ├── challenge_guides.json
    └── qa_tips.json
scripts/
└── build_vector_index.py    # NEW: 인덱스 구축 스크립트
```

## 8. 예상 데이터 규모

- **초기 데이터**: 50-100개 조언 항목
- **확장 후**: 200-500개 조언 항목
- **벡터 차원**: 1536 (OpenAI text-embedding-3-small 기준)
- **인덱스 크기**: 약 1-5MB (데이터 크기에 따라)

## 9. 성능 고려사항

- **인덱스 튜닝**: ivfflat 인덱스의 lists 파라미터 조정
- **캐싱**: 자주 검색되는 쿼리 결과 캐싱
- **비동기 처리**: Embedding 생성 및 검색 비동기화
- **연결 풀링**: PostgreSQL 연결 풀 사용

