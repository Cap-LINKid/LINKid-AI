# VectorDB 통합 유스케이스 및 구현 전략

## 유스케이스 개요

### 1. 핵심순간 (key_moments) - needs_improvement 개선
**목적**: 개선이 필요한 순간에서 더 나은 응답(`better_response`) 생성 시 전문가 조언 참고

**플로우**:
```
needs_improvement 감지 
  → VectorDB 검색 (패턴명, reason 기반)
  → 전문가 조언 조회 (top 2-3개)
  → LLM이 better_response 생성 시 참고
  → 레퍼런스 정보 포함 (예: "참고: 오은영 박사 연구, DPICS 가이드")
```

### 2. 챌린지 생성 (coaching_plan) - 개선 방법 찾기
**목적**: 챌린지 생성 시 문제 인식 후 개선 방법을 VectorDB에서 찾아 레퍼런스와 함께 제공

**플로우**:
```
패턴 분석 → 문제 인식
  → VectorDB 검색 (패턴명, 챌린지 가이드)
  → 개선 방법 및 챌린지 가이드 조회
  → 챌린지 생성 (actions, goal에 반영)
  → 레퍼런스 정보를 challenge 객체에 포함
```

## 데이터 구조 확장

### 1. needs_improvement 응답 구조 확장

**현재 구조**:
```python
{
    "dialogue": [...],
    "reason": "...",
    "better_response": "...",
    "pattern_hint": "..."
}
```

**확장 후 구조**:
```python
{
    "dialogue": [...],
    "reason": "...",
    "better_response": "...",
    "pattern_hint": "...",
    "expert_references": [  # NEW
        {
            "title": "긍정기회놓치기 패턴 개선 가이드",
            "source": "DPICS 가이드",
            "author": "오은영 박사 연구",
            "excerpt": "아이의 긍정적 행동에 즉시 구체적으로 칭찬하기...",
            "relevance_score": 0.85
        }
    ]
}
```

### 2. coaching_plan 챌린지 구조 확장

**현재 구조**:
```python
{
    "challenge": {
        "title": "...",
        "goal": "...",
        "actions": [...],
        "period_days": 7,
        "suggested_period": {...}
    }
}
```

**확장 후 구조**:
```python
{
    "challenge": {
        "title": "...",
        "goal": "...",
        "actions": [...],
        "period_days": 7,
        "suggested_period": {...},
        "improvement_methods": [  # NEW
            {
                "method": "구체적이고 즉시적인 칭찬 사용하기",
                "description": "아이의 긍정적 행동을 즉시 구체적으로 칭찬하는 방법...",
                "references": [
                    {
                        "title": "7일 긍정강화 챌린지 가이드",
                        "source": "DPICS 가이드",
                        "excerpt": "..."
                    }
                ]
            }
        ],
        "references": [  # NEW - 전체 레퍼런스
            {
                "title": "긍정기회놓치기 패턴 개선 가이드",
                "source": "DPICS 가이드",
                "author": "오은영 박사 연구",
                "type": "pattern_advice"
            }
        ]
    }
}
```

## 검색 쿼리 전략

### 1. key_moments - needs_improvement 검색

**검색 쿼리 생성**:
```python
def build_needs_improvement_query(moment: Dict[str, Any]) -> str:
    """
    needs_improvement 순간에서 검색 쿼리 생성
    """
    pattern_hint = moment.get("pattern_hint", "")
    reason = moment.get("reason", "")
    
    # 패턴 힌트가 있으면 우선 사용
    if pattern_hint:
        query = f"{pattern_hint} 패턴 개선 방법"
    else:
        # reason에서 키워드 추출
        query = f"{reason} 개선 방법"
    
    return query
```

**메타데이터 필터링 (새 스키마 기준)**:
```python
filters = {
    # type / advice_type: DB의 type 컬럼에 매핑
    "type": ["Negative", "Positive"],
    # category / age 필터
    "category": "훈육",
    "age": "유아~초등",
    # pattern_names: Related_DPICS 문자열에 부분 일치 (예: "11 Negative", "15 Negative")
    "pattern_names": [pattern_name] if pattern_name else None,
}
```

**검색 파라미터**:
- `top_k`: 2-3개 (너무 많으면 노이즈)
- `threshold`: 0.7 이상
- 우선순위: `pattern_advice` > `coaching`

### 2. coaching_plan - 챌린지 생성 검색

**검색 쿼리 생성**:
```python
def build_challenge_query(patterns: List[Dict], key_moments: Dict) -> str:
    """
    챌린지 생성 시 검색 쿼리 생성
    """
    # 가장 빈번한 패턴 찾기
    if not patterns:
        return "부모-자녀 상호작용 개선 챌린지"
    
    # 패턴별 발생 횟수 계산
    pattern_counts = {}
    for p in patterns:
        name = p.get("pattern_name", "")
        if name:
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
    
    # 가장 빈번한 패턴
    most_frequent = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None
    
    if most_frequent:
        query = f"{most_frequent} 패턴 개선 챌린지 가이드"
    else:
        query = "부모-자녀 상호작용 개선 챌린지 가이드"
    
    return query
```

**메타데이터 필터링 (새 스키마 기준)**:
```python
filters = {
    # 챌린지/패턴 조언 등 타입 기반 필터
    "type": ["Negative", "Additional"],
    # 가장 빈번한 패턴 이름을 Related_DPICS에 부분 일치로 사용
    "pattern_names": [most_frequent_pattern] if most_frequent_pattern else None,
}
```

**검색 파라미터**:
- `top_k`: 3-5개
- `threshold`: 0.7 이상
- 우선순위: `challenge_guide` > `pattern_advice` > `coaching`

## 레퍼런스 표시 형식

### 1. needs_improvement 레퍼런스

**better_response에 포함**:
```
[better_response 텍스트]

참고:
- "긍정기회놓치기 패턴 개선 가이드" (오은영 박사 연구, DPICS 가이드)
- "아이의 긍정적 행동에 즉시 구체적으로 칭찬하기" (오은영 박사 연구, DPICS 가이드)
```

**또는 별도 필드로 분리** (권장):
```python
{
    "better_response": "아이의 긍정적 행동을 즉시 구체적으로 칭찬하세요...",
    "expert_references": [
        {
            "title": "긍정기회놓치기 패턴 개선 가이드",
            "source": "DPICS 가이드",
            "author": "오은영 박사 연구",  # source에 포함되거나 별도 필드
            "excerpt": "아이의 긍정적 행동에 즉시 구체적으로 칭찬하는 것이 중요합니다..."
        }
    ]
}
```

### 2. coaching_plan 레퍼런스

**challenge 객체에 포함**:
```python
{
    "challenge": {
        "title": "긍정기회놓치기 3회 도전",
        "goal": "아이의 긍정적 행동을 즉시 구체적으로 칭찬하기",
        "actions": [
            "매일 아이의 긍정적 행동 3개 이상 찾아 칭찬하기",
            "칭찬 시 구체적인 행동 언급하기",
            "칭찬 내용 기록하기"
        ],
        "references": [
            {
                "title": "7일 긍정강화 챌린지 가이드",
                "source": "DPICS 가이드",
                "type": "challenge_guide"
            },
            {
                "title": "긍정기회놓치기 패턴 개선 가이드",
                "source": "DPICS 가이드",
                "type": "pattern_advice"
            }
        ]
    }
}
```

**프론트엔드 표시 예시**:
```
챌린지: 긍정기회놓치기 3회 도전

목표: 아이의 긍정적 행동을 즉시 구체적으로 칭찬하기

액션:
1. 매일 아이의 긍정적 행동 3개 이상 찾아 칭찬하기
2. 칭찬 시 구체적인 행동 언급하기
3. 칭찬 내용 기록하기

참고 자료:
- "7일 긍정강화 챌린지 가이드" (오은영 박사 연구, DPICS 가이드)
- "긍정기회놓치기 패턴 개선 가이드" (오은영 박사 연구, DPICS 가이드)
```

## 구현 단계

### Phase 1: VectorDB 유틸리티 구현

1. **`src/utils/vector_store.py` 생성**
   - pgVector 연결 및 검색 함수
   - 쿼리 생성 헬퍼 함수
   - 메타데이터 필터링 함수

2. **`src/utils/embeddings.py` 생성**
   - Embedding 모델 래퍼
   - 텍스트 → 벡터 변환

### Phase 2: key_moments_agent 통합

1. **`key_moments_node` 수정**
   - needs_improvement 생성 전 VectorDB 검색
   - 검색 결과를 프롬프트에 포함
   - 레퍼런스 정보를 응답에 추가

2. **프롬프트 수정**
   - 전문가 조언을 참고하여 better_response 생성하도록 지시
   - 레퍼런스 정보 포함하도록 지시

### Phase 3: coaching_agent 통합

1. **`coaching_plan_node` 수정**
   - 챌린지 생성 전 VectorDB 검색
   - 개선 방법을 actions에 반영
   - 레퍼런스 정보를 challenge 객체에 추가

2. **프롬프트 수정**
   - 전문가 조언을 참고하여 챌린지 생성하도록 지시
   - 레퍼런스 정보 포함하도록 지시

### Phase 4: 테스트 및 최적화

1. 검색 품질 테스트
2. 레퍼런스 표시 형식 검증
3. 성능 최적화

## 코드 예시

### VectorDB 검색 함수 (요약)

실제 구현은 `src/utils/vector_store.py`의 `search_expert_advice`를 사용하며,
다음과 같이 **쿼리 + 필터 + 유사도 임계값**을 인자로 받습니다.

```python
from typing import Any, Dict, List, Optional
from src.utils.vector_store import search_expert_advice

results: List[Dict[str, Any]] = search_expert_advice(
    query="훈육에서 공포를 주지 않고 단호하게 말하는 법",
    top_k=3,
    threshold=0.3,
    filters={
        "type": ["Negative"],
        "age": "유아~초등",
        "pattern_names": ["11 Negative", "15 Negative"],  # Related_DPICS 부분 일치
    },
)

for r in results:
    print(r["title"], r["relevance_score"])
```

### key_moments_agent 통합

```python
# src/expert/key_moments_agent.py 수정

def key_moments_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... 기존 코드 ...
    
    # needs_improvement에 VectorDB 검색 통합
    if key_moments_content.needs_improvement:
        from src.utils.vector_store import search_expert_advice
        
        for moment in key_moments_content.needs_improvement:
            # 검색 쿼리 생성
            query = build_needs_improvement_query({
                "pattern_hint": moment.pattern_hint,
                "reason": moment.reason
            })
            
            # VectorDB 검색
            expert_advice = search_expert_advice(
                query=query,
                top_k=2,
                filters={
                    "advice_type": ["pattern_advice", "coaching"],
                    "pattern_names": [moment.pattern_hint] if moment.pattern_hint else None
                }
            )
            
            # 레퍼런스 정보 추가 (별도 처리 필요)
            # 레퍼런스 정보 추가
            moment.expert_references = [
                {
                    "title": advice["title"],
                    "source": advice["source"],
                    "author": advice.get("author", ""),
                    "excerpt": advice["content"][:200] + "..." if len(advice["content"]) > 200 else advice["content"],
                    "relevance_score": advice["relevance_score"]
                }
                for advice in expert_advice
            ]
```

### coaching_agent 통합

```python
# src/expert/coaching_agent.py 수정

def coaching_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... 기존 코드 ...
    
    # 챌린지 생성 전 VectorDB 검색
    from src.utils.vector_store import search_expert_advice
    
    # 가장 빈번한 패턴 찾기
    most_frequent_pattern = find_most_frequent_pattern(patterns)
    
    # 검색 쿼리 생성
    query = build_challenge_query(patterns, key_moments)
    
    # VectorDB 검색
    expert_advice = search_expert_advice(
        query=query,
        top_k=5,
        filters={
            "advice_type": ["challenge_guide", "pattern_advice"],
            "pattern_names": [most_frequent_pattern] if most_frequent_pattern else None
        }
    )
    
    # 프롬프트에 전문가 조언 추가
    expert_advice_str = format_expert_advice_for_prompt(expert_advice)
    
    # 프롬프트 수정
    res = (_COACHING_PROMPT | llm).invoke({
        "summary": summary,
        "style_analysis": style_str,
        "patterns": patterns_str,
        "key_moments": key_moments_str,
        "expert_advice": expert_advice_str,  # NEW
    })
    
    # ... 기존 파싱 코드 ...
    
    # 레퍼런스 정보 추가
    if "challenge" in coaching_data:
        coaching_data["challenge"]["references"] = [
            {
                "title": advice["title"],
                "source": advice["source"],
                "author": advice.get("author", ""),
                "type": advice["advice_type"]
            }
            for advice in expert_advice
        ]
```

## 데이터베이스 스키마 업데이트

전문가 조언 테이블에 `author` 필드 추가 (선택사항):

```sql
-- author 필드는 이미 스키마에 포함되어 있음
-- 기존 데이터에 author가 없는 경우 source로 채우기
UPDATE expert_advice 
SET author = source 
WHERE author IS NULL OR author = '';
```

## 환경 변수

```bash
# VectorDB 검색 설정
VECTOR_SEARCH_TOP_K_NEEDS_IMPROVEMENT=2  # needs_improvement용
VECTOR_SEARCH_TOP_K_CHALLENGE=5  # 챌린지 생성용
VECTOR_SEARCH_THRESHOLD=0.7
USE_VECTOR_DB=true  # Feature flag
```

