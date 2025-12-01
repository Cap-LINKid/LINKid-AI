# VectorDB 검색 플로우 상세 설명

## 개요

전문가 조언 VectorDB는 두 가지 주요 유스케이스에서 사용됩니다:
1. **핵심순간 (key_moments) - needs_improvement**: 개선이 필요한 순간에 대한 대안 생성
2. **코칭 계획 (coaching_plan) - 챌린지 생성**: 챌린지 생성 시 개선 방법 찾기

---

## 유스케이스 1: 핵심순간 - needs_improvement 개선

### 플로우 다이어그램

```
┌─────────────────────────────────────┐
│  key_moments_node 실행              │
│  - utterances_labeled 분석          │
│  - patterns 분석                    │
│  - LLM으로 핵심순간 추출            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  needs_improvement 순간 감지          │
│  - dialogue: 대화 발췌                │
│  - reason: 개선이 필요한 이유           │
│  - pattern_hint: 관련 패턴 힌트        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  VectorDB 검색 쿼리 생성            │
│  query = f"{pattern_hint} 패턴      │
│           개선 방법"                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  search_expert_advice() 실행        │
│  - top_k: 2-3개                     │
│  - threshold: 0.3~0.7               │
│  - filters: {                       │
│      type: ["Negative","Positive"], │
│      pattern_names: [pattern_name]  │
│    }                                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  검색 결과 반환                     │
│  - title, content, source, author   │
│  - relevance_score                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM 프롬프트에 전문가 조언 추가    │
│  - better_response 생성 시 참고     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  응답 구조 확장                     │
│  {                                  │
│    "better_response": "...",        │
│    "expert_references": [...]       │
│  }                                  │
└─────────────────────────────────────┘
```

### 상세 단계

#### 1단계: needs_improvement 순간 추출
```python
# key_moments_agent.py에서 실행
needs_improvement = [
    {
        "dialogue": [
            {"speaker": "parent", "text": "왜 그렇게 했어?"},
            {"speaker": "child", "text": "..."}
        ],
        "reason": "비판적 반응으로 아이의 감정을 이해하지 못함",
        "pattern_hint": "비판적반응"  # 또는 "공감부족"
    }
]
```

#### 2단계: 검색 쿼리 생성
```python
def build_needs_improvement_query(moment: Dict[str, Any]) -> str:
    pattern_hint = moment.get("pattern_hint", "")
    reason = moment.get("reason", "")
    
    if pattern_hint:
        # 패턴 힌트가 있으면 우선 사용
        query = f"{pattern_hint} 패턴 개선 방법"
        # 예: "비판적반응 패턴 개선 방법"
    else:
        # reason에서 키워드 추출
        query = f"{reason} 개선 방법"
        # 예: "비판적 반응으로 아이의 감정을 이해하지 못함 개선 방법"
    
    return query
```

#### 3단계: VectorDB 검색 실행
```python
from src.utils.vector_store import search_expert_advice

# 각 needs_improvement 순간에 대해 검색
for moment in needs_improvement:
    # 쿼리 생성
    query = build_needs_improvement_query(moment)
    
    # 검색 실행
    expert_advice = search_expert_advice(
        query=query,
        top_k=2,  # 2-3개만 (너무 많으면 노이즈)
        threshold=0.3,  # 새 스키마 기준 기본값 낮춤 (코드 기본: env 또는 0.3)
        filters={
            # type/advice_type -> DB의 type 컬럼에 매핑
            "type": ["Negative", "Positive"],
            # pattern_hint를 Related_DPICS 문자열에 부분 일치로 사용
            "pattern_names": [moment.get("pattern_hint")] if moment.get("pattern_hint") else None,
        },
    )
    
    # 결과 예시:
    # [
    #     {
    #         "title": "비판적반응 줄이기: 공감적 대응하기",
    #         "content": "아이의 행동에 비판적으로 반응하기 전에...",
    #         "source": "DPICS 가이드",
    #         "author": "오은영 박사 연구",
    #         "relevance_score": 0.85
    #     },
    #     ...
    # ]
```

#### 4단계: LLM 프롬프트에 통합
```python
# 전문가 조언을 프롬프트에 추가
expert_advice_str = "\n\n".join([
    f"참고: {advice['title']}\n{advice['content'][:200]}..."
    for advice in expert_advice
])

# LLM 프롬프트 수정
prompt = f"""
당신은 부모-자녀 상호작용 전문가입니다.
다음 개선이 필요한 순간에 대해 더 나은 응답을 제안해주세요.

대화:
{moment['dialogue']}

문제점:
{moment['reason']}

전문가 조언:
{expert_advice_str}

위 전문가 조언을 참고하여 better_response를 생성하세요.
레퍼런스 정보도 함께 제공하세요.
"""
```

#### 5단계: 응답 구조 확장
```python
needs_improvement_list.append({
    "dialogue": dialogue_with_ko,
    "reason": moment.reason,
    "better_response": moment.better_response,  # LLM이 생성
    "pattern_hint": moment.pattern_hint,
    "expert_references": [  # NEW: 레퍼런스 추가
        {
            "title": advice["title"],
            "source": advice["source"],
            "author": advice["author"],
            "excerpt": advice["content"][:200],
            "relevance_score": advice["relevance_score"]
        }
        for advice in expert_advice
    ]
})
```

---

## 유스케이스 2: 코칭 계획 - 챌린지 생성

### 플로우 다이어그램

```
┌─────────────────────────────────────┐
│  coaching_plan_node 실행            │
│  - summary, patterns, key_moments   │
│    분석                             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  가장 빈번한 패턴 찾기              │
│  - pattern_counts 계산              │
│  - most_frequent_pattern 선택       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  VectorDB 검색 쿼리 생성            │
│  query = f"{most_frequent_pattern}  │
│           패턴 개선 챌린지 가이드"   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  search_expert_advice() 실행        │
│  - top_k: 3-5개                     │
│  - threshold: 0.3~0.7               │
│  - filters: {                       │
│      type: ["Negative","Additional"],│
│      pattern_names: [most_frequent] │
│    }                                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  검색 결과 반환                     │
│  - 챌린지 가이드 우선               │
│  - 패턴 조언 보조                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM 프롬프트에 전문가 조언 추가    │
│  - challenge 생성 시 참고           │
│  - actions, goal에 반영             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  응답 구조 확장                     │
│  {                                  │
│    "challenge": {                   │
│      "title": "...",                │
│      "goal": "...",                 │
│      "actions": [...],              │
│      "references": [...]            │
│    }                                │
│  }                                  │
└─────────────────────────────────────┘
```

### 상세 단계

#### 1단계: 가장 빈번한 패턴 찾기
```python
# coaching_agent.py에서 실행
patterns = [
    {"pattern_name": "긍정기회놓치기", "occurrences": 3},
    {"pattern_name": "명령과제시", "occurrences": 1},
    {"pattern_name": "비판적반응", "occurrences": 2}
]

# 패턴별 발생 횟수 계산
pattern_counts = {}
for p in patterns:
    name = p.get("pattern_name", "")
    if name:
        pattern_counts[name] = pattern_counts.get(name, 0) + p.get("occurrences", 1)

# 가장 빈번한 패턴
most_frequent = max(pattern_counts.items(), key=lambda x: x[1])[0]
# 예: "긍정기회놓치기" (3회)
```

#### 2단계: 검색 쿼리 생성
```python
def build_challenge_query(patterns: List[Dict], key_moments: Dict) -> str:
    if not patterns:
        return "부모-자녀 상호작용 개선 챌린지"
    
    # 가장 빈번한 패턴 찾기
    pattern_counts = {}
    for p in patterns:
        name = p.get("pattern_name", "")
        if name:
            pattern_counts[name] = pattern_counts.get(name, 0) + p.get("occurrences", 1)
    
    most_frequent = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None
    
    if most_frequent:
        query = f"{most_frequent} 패턴 개선 챌린지 가이드"
        # 예: "긍정기회놓치기 패턴 개선 챌린지 가이드"
    else:
        query = "부모-자녀 상호작용 개선 챌린지 가이드"
    
    return query
```

#### 3단계: VectorDB 검색 실행
```python
from src.utils.vector_store import search_expert_advice

# 쿼리 생성
query = build_challenge_query(patterns, key_moments)

# 검색 실행
expert_advice = search_expert_advice(
    query=query,
    top_k=5,  # 3-5개 (챌린지 가이드 + 패턴 조언)
    threshold=0.3,
    filters={
        "type": ["Negative", "Additional"],
        "pattern_names": [most_frequent] if most_frequent else None,
    },
)

# 결과 예시:
# [
#     {
#         "title": "7일 긍정강화 챌린지 가이드",
#         "content": "이 챌린지는 일주일 동안...",
#         "advice_type": "challenge_guide",
#         "relevance_score": 0.92
#     },
#     {
#         "title": "긍정기회놓치기 패턴 개선 가이드",
#         "content": "아이의 긍정적 행동에 즉시...",
#         "advice_type": "pattern_advice",
#         "relevance_score": 0.88
#     },
#     ...
# ]
```

#### 4단계: LLM 프롬프트에 통합
```python
# 전문가 조언을 프롬프트에 추가
expert_advice_str = "\n\n".join([
    f"[{advice['advice_type']}] {advice['title']}\n{advice['content']}"
    for advice in expert_advice
])

# LLM 프롬프트 수정
prompt = f"""
당신은 전문 부모 코칭 전문가입니다.
다음 정보를 바탕으로 개인화된 코칭 계획을 작성해주세요.

대화 요약:
{summary}

탐지된 패턴:
{patterns_str}

핵심 순간:
{key_moments_str}

전문가 조언 및 챌린지 가이드:
{expert_advice_str}

위 전문가 조언을 참고하여:
1. 챌린지 제목, 목표, 액션을 생성하세요
2. 전문가 조언의 구체적인 방법을 actions에 반영하세요
3. 레퍼런스 정보를 포함하세요
"""
```

#### 5단계: 응답 구조 확장
```python
coaching_data = {
    "summary": "...",
    "challenge": {
        "title": "긍정기회놓치기 3회 도전",
        "goal": "아이의 긍정적 행동을 즉시 구체적으로 칭찬하기",
        "actions": [
            "매일 아이의 긍정적 행동 3개 이상 찾아 칭찬하기",  # 전문가 조언 반영
            "칭찬 시 구체적인 행동 언급하기",  # 전문가 조언 반영
            "칭찬 내용 기록하기"
        ],
        "period_days": 7,
        "suggested_period": {...},
        "references": [  # NEW: 레퍼런스 추가
            {
                "title": "7일 긍정강화 챌린지 가이드",
                "source": "DPICS 가이드",
                "author": "오은영 박사 연구",
                "type": "challenge_guide"
            },
            {
                "title": "긍정기회놓치기 패턴 개선 가이드",
                "source": "DPICS 가이드",
                "author": "오은영 박사 연구",
                "type": "pattern_advice"
            }
        ]
    },
    "qa_tips": [...]
}
```

---

## 검색 전략 요약

### 1. 쿼리 생성 전략

| 유스케이스 | 쿼리 패턴 | 예시 |
|-----------|----------|------|
| needs_improvement | `{pattern_hint} 패턴 개선 방법` | "비판적반응 패턴 개선 방법" |
| challenge | `{most_frequent_pattern} 패턴 개선 챌린지 가이드` | "긍정기회놓치기 패턴 개선 챌린지 가이드" |

### 2. 필터링 전략

| 유스케이스 | type / advice_type 예시 | pattern_names 필터 (Related_DPICS) |
|-----------|-------------------------|-----------------------------------|
| needs_improvement | `Negative`, `Positive` 등 | 패턴 이름 또는 DPICS 코드 부분 일치 |
| challenge | `Negative`, `Additional` 등 | 가장 빈번한 패턴 이름 기반 |

### 3. 검색 파라미터

| 유스케이스 | top_k | threshold | 이유 |
|-----------|-------|-----------|------|
| needs_improvement | 2-3 | 0.7 | 정확한 조언만 필요, 노이즈 최소화 |
| challenge | 3-5 | 0.7 | 다양한 가이드와 조언 필요 |

### 4. 결과 활용

| 유스케이스 | 활용 방법 | 레퍼런스 표시 |
|-----------|----------|--------------|
| needs_improvement | `better_response` 생성 시 참고 | `expert_references` 필드에 포함 |
| challenge | `actions`, `goal` 생성 시 참고 | `challenge.references` 필드에 포함 |

---

## 실제 검색 예시

### 예시 1: needs_improvement 검색

**입력:**
```python
moment = {
    "pattern_hint": "긍정기회놓치기",
    "reason": "아이의 긍정적 행동에 칭찬하지 않음"
}
```

**검색 쿼리:**
```
"긍정기회놓치기 패턴 개선 방법"
```

**검색 결과:**
```python
[
    {
        "title": "긍정기회놓치기 패턴 개선 가이드",
        "content": "아이가 긍정적인 행동을 보였을 때, 즉시 칭찬하는 것이 중요합니다...",
        "source": "DPICS 가이드",
        "author": "오은영 박사 연구",
        "relevance_score": 0.89
    },
    {
        "title": "효과적인 칭찬 방법",
        "content": "칭찬은 구체적이고 즉시적일수록 효과적입니다...",
        "source": "DPICS 가이드",
        "author": "오은영 박사 연구",
        "relevance_score": 0.82
    }
]
```

**better_response 생성:**
```
"아이의 긍정적 행동을 즉시 구체적으로 칭찬하세요. 
예를 들어, 아이가 장난감을 정리했다면 
'와, 장난감을 깔끔하게 정리했구나! 정말 잘했어!'라고 
구체적으로 칭찬하는 것이 중요합니다."

참고: "긍정기회놓치기 패턴 개선 가이드" (오은영 박사 연구, DPICS 가이드)
```

### 예시 2: challenge 검색

**입력:**
```python
patterns = [
    {"pattern_name": "긍정기회놓치기", "occurrences": 3}
]
```

**검색 쿼리:**
```
"긍정기회놓치기 패턴 개선 챌린지 가이드"
```

**검색 결과:**
```python
[
    {
        "title": "7일 긍정강화 챌린지 가이드",
        "content": "이 챌린지는 일주일 동안 아이의 긍정적 행동을...",
        "advice_type": "challenge_guide",
        "relevance_score": 0.91
    },
    {
        "title": "긍정기회놓치기 패턴 개선 가이드",
        "content": "아이의 긍정적 행동에 즉시 구체적으로 칭찬하기...",
        "advice_type": "pattern_advice",
        "relevance_score": 0.87
    }
]
```

**challenge 생성:**
```python
{
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
            "author": "오은영 박사 연구"
        }
    ]
}
```

---

## 성능 최적화 팁

1. **캐싱**: 동일한 패턴에 대한 검색 결과는 캐싱
2. **배치 검색**: 여러 needs_improvement 순간을 한 번에 검색
3. **인덱스 활용**: PostgreSQL 인덱스가 제대로 생성되었는지 확인
4. **임계값 조정**: 검색 품질에 따라 threshold 조정 (0.6 ~ 0.8)

