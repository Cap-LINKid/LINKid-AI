# LinkID AI - 부모-자녀 대화 분석 AI 시스템

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [주요 기능](#주요-기능)
4. [기술 스택](#기술-스택)
5. [API 명세](#api-명세)
6. [설치 및 실행](#설치-및-실행)
7. [프로젝트 구조](#프로젝트-구조)
8. [주요 에이전트 설명](#주요-에이전트-설명)
9. [벡터 데이터베이스 통합](#벡터-데이터베이스-통합)
10. [배포 방법](#배포-방법)

---

## 프로젝트 개요

**LinkID AI**는 부모-자녀 간 대화를 분석하여 양육 코칭을 제공하는 AI 시스템입니다. LangGraph 기반의 멀티 에이전트 아키텍처를 사용하여 대화를 다각도로 분석하고, DPICS(Dyadic Parent-Child Interaction Coding System) 기반의 전문적인 분석 결과를 제공합니다.

### 핵심 가치
- **과학적 분석**: DPICS 기반 발화 분류 및 패턴 감지
- **실시간 피드백**: 대화 분석 결과를 즉시 제공
- **개인화된 코칭**: 벡터 검색 기반 전문가 조언 제공
- **챌린지 추적**: 부모의 양육 개선 목표를 설정하고 평가

---

## 시스템 아키텍처

### LangGraph 기반 멀티 에이전트 파이프라인

```
┌─────────────┐
│   입력      │  utterances_ko (한국어 대화)
│  challenge_specs (챌린지 스펙)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  순차 처리 단계 (Sequential Processing)         │
├─────────────────────────────────────────────────┤
│  ① preprocess          → 발화자 정규화          │
│  ② translate_ko_to_en  → 영어 번역             │
│  ③ label_utterances    → DPICS 라벨링          │
│  ④ detect_patterns     → 패턴 감지             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  병렬 분석 단계 (Parallel Analysis)             │
├─────────────────────────────────────────────────┤
│  ⑤ summarize          │ 대화 요약              │
│  ⑥ key_moments        │ 핵심 순간 추출         │
│  ⑦ analyze_style      │ 상호작용 스타일 분석   │
│  ⑧ coaching_plan      │ 코칭 계획 생성        │
│  ⑨ challenge_eval     │ 챌린지 평가            │
│  ⑩ summary_diagnosis   │ 요약 진단             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  최종 집계 (Aggregation)                        │
├─────────────────────────────────────────────────┤
│  ⑪ aggregate_result   → 통합 결과 생성         │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   출력      │  result (JSON)
└─────────────┘
```

### 주요 특징
- **순차 처리**: 전처리 → 번역 → 라벨링 → 패턴 감지
- **병렬 처리**: 6개 분석 에이전트가 동시 실행되어 처리 시간 단축
- **상태 관리**: LangGraph의 StateGraph로 각 단계의 결과를 상태로 관리
- **비동기 API**: FastAPI 기반 비동기 처리 및 실행 상태 추적

---

## 주요 기능

### 1. 대화 전처리 및 번역
- 발화자 자동 인식 및 정규화 (MOM/CHI)
- 한국어 → 영어 번역 (DPICS 분석을 위해)

### 2. DPICS 기반 발화 분류
- **부모 발화 분류**: RD(반영적 진술), PR(칭찬), CMD(지시), Q(질문), NT(정보 제공), NEG(부정적 발화) 등
- **자녀 발화 분류**: NT(자발적 발화), Q(응답형 발화), BD(감정 표현), OTH(모방 발화) 등
- DPICS-ELECTRA 모델을 활용한 자동 라벨링

### 3. 패턴 감지
- **긍정적 패턴**: 공감적 반응, 긍정적 강화 등
- **부정적 패턴**: 긍정적 기회 놓치기, 명령과 제시, 비판적 반응, 공감 부족, 반영 부족 등
- 패턴별 발생 빈도 및 대화 예시 추출

### 4. 핵심 순간 캡처
- **긍정적 순간**: 잘된 상호작용 사례
- **개선 필요 순간**: 개선이 필요한 대화 장면
- 각 순간에 대한 이유 설명 및 개선 방안 제시

### 5. 상호작용 스타일 분석
- 부모/자녀 발화 비율 분석
- DPICS 라벨별 사용 빈도 계산
- 상호작용 단계 판단 (공감적 협력, 지시적 상호작용, 개선이 필요한 상호작용 등)

### 6. 개인화된 코칭 계획
- 감지된 패턴 기반 맞춤형 코칭 조언
- 벡터 검색을 통한 전문가 조언 참조
- 7일 챌린지 제안 (목표, 액션, 기간)

### 7. 챌린지 평가
- 이전에 설정한 챌린지의 달성 여부 평가
- 챌린지 조건 충족 횟수 계산
- 증거 대화 추출 및 분석

### 8. 요약 진단
- 상호작용 단계 판단
- 긍정/부정 비율 계산
- 종합 진단 결과 제공

---

## 기술 스택

### 핵심 프레임워크
- **LangGraph**: 멀티 에이전트 워크플로우 관리
- **LangChain**: LLM 통합 및 벡터 검색
- **FastAPI**: RESTful API 서버
- **Uvicorn**: ASGI 서버

### LLM 제공자 지원
- OpenAI (GPT-4o-mini, GPT-4o 등)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (로컬 모델)

### 데이터베이스
- **PostgreSQL**: 메타데이터 및 실행 상태 저장
- **pgvector**: 벡터 검색 (전문가 조언 RAG)
- **SQLite**: 로컬 실행 상태 추적 (선택적)

### ML 모델
- **DPICS-ELECTRA**: 발화 분류 모델 (로컬 모델)
- **Transformers**: Hugging Face 기반 모델 로딩

### 기타
- **Python 3.11+**
- **Docker**: 컨테이너화
- **psycopg2**: PostgreSQL 드라이버
- **python-dotenv**: 환경 변수 관리

---

## API 명세

### 기본 정보
- **Base URL**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs` (Swagger UI)

### 주요 엔드포인트

#### 1. 대화 분석 요청 (비동기)
```http
POST /analyze
Content-Type: application/json
```

**Request Body:**
```json
{
  "utterances_ko": [
    {
      "speaker": "parent",
      "text": "오늘 뭐 했어?",
      "timestamp": 1000
    },
    {
      "speaker": "child",
      "text": "그림 그렸어!",
      "timestamp": 2000
    }
  ],
  "challenge_specs": [
    {
      "challenge_id": "chg-001",
      "title": "긍정적 기회 놓치기 3회 도전",
      "goal": "아이가 성취나 행동을 공유할 때 즉시 긍정적으로 반응하기",
      "actions": [
        {
          "action_id": "act-001",
          "content": "칭찬 문장은 짧게, 즉시 반응하기"
        }
      ]
    }
  ],
  "meta": {}
}
```

**Response:**
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "분석이 시작되었습니다.",
  "progress_percentage": 0
}
```

#### 2. 실행 상태 조회
```http
GET /status/{execution_id}
```

**Response (진행 중):**
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "analysis_status": "style_analysis",
  "progress_percentage": 70,
  "status_message": "스타일을 분석하는 중입니다",
  "created_at": "2025-01-15T10:30:00",
  "updated_at": "2025-01-15T10:30:15",
  "current_node": "analyze_style",
  "error": null,
  "result": null
}
```

**Response (완료):**
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "analysis_status": "completed",
  "progress_percentage": 100,
  "status_message": "분석이 완료되었습니다",
  "created_at": "2025-01-15T10:30:00",
  "updated_at": "2025-01-15T10:30:45",
  "current_node": null,
  "error": null,
  "result": {
    "summary_diagnosis": {
      "stage_name": "공감적 협력",
      "positive_ratio": 0.7,
      "negative_ratio": 0.3
    },
    "key_moment_capture": {
      "key_moments": {
        "positive": [...],
        "needs_improvement": [...],
        "pattern_examples": [...]
      }
    },
    "style_analysis": {...},
    "coaching_and_plan": {...},
    "growth_report": {...}
  }
}
```

#### 3. 모든 실행 목록 조회
```http
GET /executions
```

**Response:**
```json
{
  "executions": [
    {
      "execution_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2025-01-15T10:30:00",
      "progress_percentage": 100
    }
  ]
}
```

### 분석 상태 (Analysis Status)
- `translating` (10%): 대화를 번역하는 중입니다
- `labeling` (30%): 발화를 분류하는 중입니다
- `pattern_detection` (50%): 패턴을 감지하는 중입니다
- `style_analysis` (70%): 스타일을 분석하는 중입니다
- `coaching_generation` (85%): 코칭을 생성하는 중입니다
- `report_finalizing` (95%): 리포트를 작성하는 중입니다
- `completed` (100%): 분석이 완료되었습니다
- `failed` (0%): 분석 중 오류가 발생했습니다

상세한 API 명세는 `API_STATUS_SPEC.md` 파일을 참고하세요.

---

## 설치 및 실행

### 사전 요구사항
- Python 3.11 이상
- PostgreSQL 12+ (pgvector 확장 지원)
- Docker (선택적)

### 1. 저장소 클론
```bash
git clone <repository-url>
cd linkid-ai
```

### 2. 가상 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# LLM 제공자 설정
MODEL_PROVIDER=openai  # openai | anthropic | google | ollama
MODEL_NAME=gpt-4o-mini
MINI_MODEL_NAME=gpt-4o-mini

# API 키
OPENAI_API_KEY=your_openai_api_key
# 또는
ANTHROPIC_API_KEY=your_anthropic_api_key
# 또는
GOOGLE_API_KEY=your_google_api_key

# PostgreSQL 설정 (벡터 DB)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=linkid_ai
POSTGRES_USER=linkid_user
POSTGRES_PASSWORD=your_password

# VectorDB 설정
VECTOR_DB_TABLE=expert_advice
EMBEDDING_MODEL=openai  # openai | anthropic | google | local
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=1536
VECTOR_SEARCH_TOP_K=5
VECTOR_SEARCH_THRESHOLD=0.7
USE_VECTOR_DB=true

# Ollama 설정 (로컬 모델 사용 시)
OLLAMA_BASE_URL=http://localhost:11434
```

### 5. PostgreSQL 및 pgvector 설정
```bash
# PostgreSQL에 접속하여 pgvector 확장 설치
psql -U linkid_user -d linkid_ai
CREATE EXTENSION IF NOT EXISTS vector;

# 스키마 생성
psql -U linkid_user -d linkid_ai -f data/sql/vector_db_schema.sql
```

### 6. 벡터 인덱스 구축 (선택적)
```bash
python scripts/build_vector_index.py
```

### 7. 서버 실행
```bash
# 개발 모드
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 또는 스크립트 사용
./run_server.sh
```

### 8. API 테스트
```bash
# Swagger UI 접속
open http://localhost:8000/docs

# 또는 curl로 테스트
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "utterances_ko": [
      {"speaker": "parent", "text": "오늘 뭐 했어?"},
      {"speaker": "child", "text": "그림 그렸어!"}
    ]
  }'
```

---

## 프로젝트 구조

```
linkid-ai/
├── src/
│   ├── api/                    # FastAPI 애플리케이션
│   │   ├── main.py            # API 엔드포인트 정의
│   │   ├── status.py          # 실행 상태 관리
│   │   └── storage.py         # 상태 저장소
│   │
│   ├── expert/                # 전문 에이전트 노드
│   │   ├── preprocess_agent.py        # 전처리
│   │   ├── translate_agent.py        # 번역
│   │   ├── label_agent.py             # DPICS 라벨링
│   │   ├── pattern_agent.py           # 패턴 감지
│   │   ├── summarize_agent.py        # 요약
│   │   ├── key_moments_agent.py       # 핵심 순간 추출
│   │   ├── style_agent.py             # 스타일 분석
│   │   ├── coaching_agent.py         # 코칭 계획
│   │   ├── challenge_agent.py         # 챌린지 평가
│   │   ├── summary_diagnosis_agent.py # 요약 진단
│   │   └── aggregate_agent.py         # 결과 집계
│   │
│   ├── router/                # LangGraph 라우터
│   │   ├── router.py         # 그래프 정의
│   │   └── states.py          # 상태 스키마
│   │
│   ├── utils/                 # 유틸리티
│   │   ├── agent.py          # LLM 에이전트 헬퍼
│   │   ├── common_prompts.py # 공통 프롬프트
│   │   ├── embeddings.py     # 임베딩 모델
│   │   ├── vector_store.py   # 벡터 검색
│   │   ├── dpics_electra.py  # DPICS 모델
│   │   └── pattern_manager.py # 패턴 관리
│   │
│   └── vs/                    # 벡터 스토어 관련
│       └── ddl.py            # DDL 헬퍼
│
├── data/
│   ├── ddl/                   # DDL 예시 데이터
│   ├── expert_advice/          # 전문가 조언 원본 데이터
│   ├── expert_advice2/        # 추가 전문가 조언
│   └── sql/                   # SQL 스크립트
│       └── vector_db_schema.sql
│
├── models/                    # 로컬 ML 모델
│   └── dpics-electra/        # DPICS-ELECTRA 모델
│
├── scripts/                   # 유틸리티 스크립트
│   ├── build_vector_index.py # 벡터 인덱스 구축
│   └── test_vector_search.py # 벡터 검색 테스트
│
├── Dockerfile                 # Docker 이미지 정의
├── docker-compose.yml         # Docker Compose 설정
├── requirements.txt          # Python 의존성
├── langgraph.json           # LangGraph 설정
└── README.md                 # 이 파일
```

---

## 주요 에이전트 설명

### 1. Preprocess Agent (`preprocess_agent.py`)
- **역할**: 발화자 정규화 및 전처리
- **입력**: `utterances_ko` (원본 한국어 대화)
- **출력**: `utterances_normalized` (정규화된 발화, MOM/CHI 형식)
- **기능**: LLM을 사용하여 발화자 자동 인식 및 정규화

### 2. Translate Agent (`translate_agent.py`)
- **역할**: 한국어 → 영어 번역
- **입력**: `utterances_normalized`
- **출력**: `utterances_en` (영어 번역된 발화)
- **기능**: DPICS 분석을 위한 영어 번역

### 3. Label Agent (`label_agent.py`)
- **역할**: DPICS 기반 발화 분류
- **입력**: `utterances_en`
- **출력**: `utterances_labeled` (라벨이 부여된 발화)
- **기능**: DPICS-ELECTRA 모델을 사용한 자동 라벨링

### 4. Pattern Agent (`pattern_agent.py`)
- **역할**: 상호작용 패턴 감지
- **입력**: `utterances_labeled`
- **출력**: `patterns` (감지된 패턴 리스트)
- **기능**: 긍정/부정 패턴 자동 감지 및 빈도 계산

### 5. Summarize Agent (`summarize_agent.py`)
- **역할**: 대화 요약
- **입력**: `utterances_labeled`, `patterns`
- **출력**: `summary` (대화 요약 텍스트)
- **기능**: 전체 대화의 핵심 내용 요약

### 6. Key Moments Agent (`key_moments_agent.py`)
- **역할**: 핵심 순간 추출
- **입력**: `utterances_labeled`, `patterns`
- **출력**: `key_moments` (긍정/개선 필요 순간 리스트)
- **기능**: 중요한 상호작용 순간 식별 및 분석

### 7. Style Agent (`style_agent.py`)
- **역할**: 상호작용 스타일 분석
- **입력**: `utterances_labeled`
- **출력**: `style_analysis` (스타일 분석 결과)
- **기능**: DPICS 라벨별 비율 계산 및 상호작용 단계 판단

### 8. Coaching Agent (`coaching_agent.py`)
- **역할**: 개인화된 코칭 계획 생성
- **입력**: `patterns`, `key_moments`, `style_analysis`
- **출력**: `coaching_plan` (코칭 계획 및 챌린지)
- **기능**: 벡터 검색을 통한 전문가 조언 참조 및 맞춤형 코칭 생성

### 9. Challenge Agent (`challenge_agent.py`)
- **역할**: 챌린지 평가
- **입력**: `utterances_labeled`, `challenge_specs`
- **출력**: `challenge_evaluations` (챌린지 평가 결과)
- **기능**: 이전 챌린지의 달성 여부 평가 및 증거 추출

### 10. Summary Diagnosis Agent (`summary_diagnosis_agent.py`)
- **역할**: 요약 진단
- **입력**: `utterances_labeled`, `patterns`
- **출력**: `summary_diagnosis` (진단 결과)
- **기능**: 긍정/부정 비율 계산 및 상호작용 단계 판단

### 11. Aggregate Agent (`aggregate_agent.py`)
- **역할**: 최종 결과 집계
- **입력**: 모든 병렬 분석 결과
- **출력**: `result` (통합 JSON 결과)
- **기능**: 모든 분석 결과를 하나의 JSON으로 통합

---

## 벡터 데이터베이스 통합

### 개요
전문가 조언을 벡터 검색으로 제공하기 위해 PostgreSQL + pgvector를 사용합니다.

### 스키마
```sql
CREATE TABLE expert_advice (
    id SERIAL PRIMARY KEY,
    category TEXT,
    age TEXT,
    related_dpics TEXT,
    keyword TEXT,
    situation TEXT,
    type TEXT,
    advice TEXT,
    reference TEXT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 검색 전략
1. **쿼리 생성**: 패턴명, DPICS 라벨, 상호작용 단계 등을 기반으로 검색 쿼리 생성
2. **벡터 검색**: pgvector의 코사인 유사도 검색 사용
3. **메타데이터 필터링**: 패턴명, 타입, 카테고리 등으로 필터링
4. **상위 K개 결과**: 유사도 임계값 이상의 상위 5개 결과 반환

### 사용 예시
```python
from src.utils.vector_store import search_expert_advice

# 패턴 기반 검색
results = search_expert_advice(
    query="긍정적 기회 놓치기 패턴 개선 방법",
    top_k=5,
    threshold=0.7,
    filters={
        "pattern_names": ["긍정적 기회 놓치기"],
        "type": "pattern_advice"
    }
)
```

상세한 벡터 DB 전략은 `VECTOR_DB_STRATEGY.md` 파일을 참고하세요.

---

## 배포 방법

### Docker를 사용한 배포

#### 1. Docker 이미지 빌드
```bash
docker build -t linkid-ai:latest .
```

#### 2. Docker Compose 사용 (권장)
```bash
docker-compose up -d
```

#### 3. 환경 변수 파일 설정
`.env` 파일을 생성하고 필요한 환경 변수를 설정하세요.

#### 4. 컨테이너 실행
```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  linkid-ai:latest
```

### 프로덕션 배포

#### 1. 환경 변수 설정
프로덕션 환경에서는 환경 변수를 안전하게 관리하세요:
- Kubernetes Secrets
- AWS Secrets Manager
- 환경 변수 주입

#### 2. 데이터베이스 설정
- PostgreSQL 인스턴스 생성
- pgvector 확장 설치
- 스키마 생성 및 벡터 인덱스 구축

#### 3. 서버 실행
```bash
# Gunicorn 사용 (권장)
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 모니터링
- API 엔드포인트: `/status/{execution_id}`로 실행 상태 추적
- 로그: 각 노드의 실행 로그 확인
- 에러 처리: 실패한 실행은 `status: "failed"` 및 `error` 필드에 상세 정보 포함

---

## 보안 주의사항

- `.env` 파일과 API 키는 절대 Git에 커밋하지 마세요.
- 프로덕션 환경에서는 환경 변수로 API 키를 주입하세요.
- PostgreSQL 비밀번호는 안전하게 관리하세요.
- API 엔드포인트에 인증/인가를 추가하는 것을 권장합니다.

---

## 라이선스

[라이선스 정보를 여기에 추가하세요]

---

## 문의 및 지원

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

## 참고 문서

- `API_STATUS_SPEC.md`: 상세 API 명세서
- `VECTOR_DB_STRATEGY.md`: 벡터 DB 전략
- `VECTOR_DB_INTEGRATION.md`: 벡터 DB 통합 가이드
- `VECTOR_DB_SETUP.md`: 벡터 DB 설정 가이드
- `VECTOR_DB_TROUBLESHOOTING.md`: 벡터 DB 문제 해결
