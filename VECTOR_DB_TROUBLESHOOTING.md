# VectorDB 검색 문제 해결 가이드

## 문제: expert_references가 null로 나오는 경우

### 원인 분석

1. **threshold가 너무 높음**: 기본값 0.7인데 실제 유사도는 0.3~0.5 정도
2. **USE_VECTOR_DB 환경 변수 미설정**: false로 되어 있으면 검색이 실행되지 않음
3. **필터링 조건이 너무 엄격함**: `type` / `pattern_names` 필터가 실제 데이터와 잘 매칭되지 않을 수 있음  
   (특히 `pattern_names`는 DB의 `Related_DPICS` 문자열에 부분 일치로 매핑됨)

### 해결 방법

#### 방법 1: threshold 조정 (권장)

`.env` 파일에 다음을 추가/수정:

```bash
# threshold를 낮춰서 더 많은 결과 반환
VECTOR_SEARCH_THRESHOLD=0.4  # 기본값 0.7에서 0.4로 변경
```

**권장 threshold 값:**
- `needs_improvement`: 0.4 (정확한 조언 필요)
- `challenge`: 0.5 (다양한 가이드 필요)

#### 방법 2: 더 관련성 높은 데이터 추가

현재 데이터는 이미 있지만, 검색 쿼리와 더 유사한 내용으로 데이터를 추가하면 유사도가 높아집니다.

**추가할 데이터 예시:**

```json
{
  "title": "명령과제시 패턴: 아이의 감정을 고려한 대화법",
  "content": "아이의 감정을 무시하고 명령만 내리지 말고, 아이의 의견과 감정을 먼저 들어주세요. '왜 엄마가 만들고 있는 거 뺏어가'라고 말하기보다는 '지금 엄마가 만들고 있는데, 네가 하고 싶은 게 있구나. 같이 할까, 아니면 조금 기다려줄까?'라고 물어보세요. 아이의 감정을 인정하고 선택권을 제공하면 협조적이 됩니다.",
  "summary": "명령 대신 아이의 감정을 인정하고 선택권 제공하기",
  "advice_type": "pattern_advice",
  "category": "자율성",
  "pattern_names": ["명령과제시"],
  "dpics_labels": ["CMD", "Q", "RD"],
  "interaction_stages": ["지시적 상호작용", "개선이 필요한 상호작용"],
  "severity_levels": ["medium"],
  "related_challenges": ["질문하기 챌린지"],
  "tags": ["질문", "선택권", "감정인정", "협조"],
  "source": "DPICS 가이드",
  "author": "오은영 박사 연구",
  "priority": 5
}
```

#### 방법 3: 검색 쿼리 개선

현재 쿼리 예시: `"{pattern_hint} 패턴 개선 방법"`

더 구체적인 쿼리로 변경:
- `"{pattern_hint} 패턴 개선: 아이의 감정 고려하기"`
- `"{pattern_hint} 패턴에서 공감적 대응 방법"`

또한, 새 스키마에서는 `filters`를 다음과 같이 사용하는 것이 좋습니다:

```python
filters = {
    "type": ["Negative", "Positive"],  # DB의 type 컬럼
    # pattern_names는 Related_DPICS 문자열에 부분 일치로 매핑됨
    "pattern_names": ["명령과제시", "11 Negative"],
}
```

### 현재 데이터 확인

데이터베이스에 다음 데이터가 이미 있습니다:

1. **명령과제시**:
   - "명령과제시 패턴 개선: 선택권 제공하기"
   - pattern_names: ["명령과제시"]

2. **비판적반응**:
   - "비판적반응 줄이기: 공감적 대응하기"
   - pattern_names: ["비판적반응"]

### 테스트 방법

```python
# threshold를 낮춰서 테스트
from src.utils.vector_store import search_expert_advice

results = search_expert_advice(
    query="명령과제시 패턴 개선 방법",
    top_k=3,
    threshold=0.4,  # 0.7에서 0.4로 낮춤
    filters={
        # 새 스키마 기준 예시
        "type": ["Negative", "Positive"],
        "pattern_names": ["명령과제시"]  # Related_DPICS에 부분 일치
    }
)
```

### 환경 변수 확인

`.env` 파일에 다음이 설정되어 있는지 확인:

```bash
USE_VECTOR_DB=true  # 필수!
VECTOR_SEARCH_THRESHOLD=0.4  # 권장: 0.4
VECTOR_SEARCH_TOP_K_NEEDS_IMPROVEMENT=2
```

### 즉시 해결 방법

1. **threshold 낮추기** (가장 빠른 해결책):
   ```bash
   # .env 파일에 추가
   VECTOR_SEARCH_THRESHOLD=0.4
   ```

2. **USE_VECTOR_DB 확인**:
   ```bash
   # .env 파일에 추가
   USE_VECTOR_DB=true
   ```

3. **재시작**: 환경 변수 변경 후 애플리케이션 재시작

