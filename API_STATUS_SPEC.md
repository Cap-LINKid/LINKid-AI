# 실행 상태 추적 API 명세서

## 개요
분석 요청의 진행 상태를 실시간으로 추적할 수 있는 API입니다.

## API 엔드포인트

### 1. 분석 요청 (비동기)
**POST** `/analyze`

대화 분석을 시작하고 실행 ID를 반환합니다.

**Request:**
```json
{
  "utterances_ko": [
    "엄마: 오늘 뭐 했어?",
    "아이: 그림 그렸어!"
  ],
  "challenge_spec": {...},
  "meta": {}
}
```

**Response:**
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "분석을 준비하는 중입니다",
  "progress_percentage": 0
}
```

---

### 2. 실행 상태 조회
**GET** `/status/{execution_id}`

특정 실행의 현재 상태를 조회합니다.

**Response:**
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

**완료된 경우:**
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
            "summary_diagnosis": {...},
            "key_moment_capture": {...},
            "style_analysis": {...},
            "coaching_and_plan": {...},
            "growth_report": {...}
        }
    }
```

    ---

## Result 구조 상세

완료된 분석의 `result` 필드는 다음 구조를 가집니다:

### 1. summary_diagnosis (요약 진단)

```json
{
"stage_name": "공감적 협력",
"positive_ratio": 0.7,
"negative_ratio": 0.3
}
```

- `stage_name`: 상호작용 단계 이름 (예: "공감적 협력", "지시적 상호작용", "균형잡힌 대화", "개선이 필요한 상호작용")
- `positive_ratio`: 긍정적 상호작용 비율 (0.0 ~ 1.0)
- `negative_ratio`: 부정적 상호작용 비율 (0.0 ~ 1.0)
- `positive_ratio`와 `negative_ratio`의 합은 1.0

### 2. key_moment_capture (핵심 순간 캡처)

```json
{
"key_moments": {
    "positive": [
    {
        "dialogue": [
        {"speaker": "child", "text": "이거 싫어, 안 해!"},
        {"speaker": "parent", "text": "많이 속상하구나."}
        ],
        "reason": "아이가 부정적 감정을 표현했을 때, 부모님이 이를 지시로 억누르지 않고 감정을 그대로 읽어주셨어요. 아이의 감정 조절 능력을 키우는 순간입니다.",
        "pattern_hint": "긍정적 상호작용 (Emotional Coaching)"
    }
    ],
    "needs_improvement": [
    {
        "dialogue": [
        {"speaker": "child", "text": "엄마, 나 이거 다 만들었어!"},
        {"speaker": "parent", "text": "어, 근데 손은 씻었어?"}
        ],
        "reason": "아이가 자신의 성과를 공유한 결정적 순간이었는데, 검증/전환형 질문이 먼저 나와 성취감을 줄일 수 있습니다.",
        "better_response": "우와, 혼자서 멋진 성을 완성했네!",
        "pattern_hint": "긍정적 기회 놓치기"
    }
    ],
    "pattern_examples": [
    {
        "pattern_name": "긍정적 기회 놓치기",
        "occurrences": 2,
        "dialogue": [
        {"speaker": "child", "text": "나 블록 10개나 쌓았어!"},
        {"speaker": "parent", "text": "응, 좋아."}
        ],
        "problem_explanation": "아이의 성취 공유에 즉각적 칭찬이 없으면 동기가 떨어집니다.",
        "suggested_response": "우와, 불안한 줄 알았는데 무너지지 않게 조심했구나!"
    }
    ]
}
}
```

### 3. style_analysis (스타일 분석)

```json
{
"interaction_style": {
    "parent_analysis": {
    "categories": [
        {
        "name": "반영적 듣기",
        "ratio": 0.20,
        "label": "RD"
        },
        {
        "name": "칭찬",
        "ratio": 0.20,
        "label": "PR"
        },
        {
        "name": "지시형 발화",
        "ratio": 0.20,
        "label": "CMD"
        },
        {
        "name": "질문",
        "ratio": 0.15,
        "label": "Q"
        },
        {
        "name": "정보 제공",
        "ratio": 0.10,
        "label": "NT"
        },
        {
        "name": "부정적 피드백",
        "ratio": 0.15,
        "label": "NEG"
        }
    ]
    },
    "child_analysis": {
    "categories": [
        {
        "name": "자발적 발화",
        "ratio": 0.14,
        "label": "NT"
        },
        {
        "name": "응답형 발화",
        "ratio": 0.20,
        "label": "Q"
        },
        {
        "name": "감정 표현",
        "ratio": 0.25,
        "label": "BD"
        },
        {
        "name": "모방 발화",
        "ratio": 0.10,
        "label": "OTH"
        }
    ]
    }
},
"summary": "아이의 자발적 발화가 14%로 낮은 편이며, 이는 부모님의 질문형 발화가 부족해 아이가 대화 주도권을 얻기 어려운 상황을 의미할 수 있습니다."
}
```

### 4. coaching_and_plan (코칭 및 계획)

```json
{
"coaching_plan": {
    "summary": "이번 상호작용에서는 아이의 감정을 잘 읽어주셨습니다(긍정 패턴). 그러나 아이가 성취를 자랑할 때('긍정적 기회 놓치기' 패턴 3회) 이를 놓치는 경향이 보였습니다. 이는 부모님의 '반영적 듣기' 비율(20%)이 평균(30%)보다 낮기 때문입니다.",
    "challenge": {
    "title": "긍정적 기회 놓치기 3회 도전",
    "goal": "아이가 성취나 행동을 공유할 때 즉시 긍정적으로 반응하기",
    "period_days": 7,
    "suggested_period": {
        "start": "2025-01-15",
        "end": "2025-01-22"
    },
    "actions": [
        "칭찬 문장은 짧게, 즉시 반응하기",
        "아이의 행동 중 구체적인 한 부분을 칭찬하기",
        "부정적 피드백은 3초 후에 말하기"
    ]
    },
    "qa_tips": [
    {
        "question": "'긍정적 기회'는 언제 발생하나요?",
        "answer": "아이가 자신의 성취나 사회적 행동을 표현할 때(예: '나 그림 그렸어', '동생이랑 놀았어') 발생합니다."
    },
    {
        "question": "어떻게 반응해야 하나요?",
        "answer": "반영적 듣기(예: '그림 그렸구나!')나 즉시 칭찬(예: '멋지다!')으로 긍정적인 반응을 보여주세요."
    }
    ]
}
}
```

### 5. growth_report (성장 리포트)

```json
{
"analysis_session": {
    "comment": "이번 대화에서는 감정 읽기가 잘 되었고, 지시형 발화는 짧게 마무리하려는 시도가 있었습니다."
},
"current_metrics": [
    {
    "key": "reflective_listening_ratio",
    "label": "반영적 듣기",
    "value": 0.35,
    "value_type": "ratio"
    },
    {
    "key": "directive_speech_ratio",
    "label": "지시형 발화",
    "value": 0.28,
    "value_type": "ratio"
    },
    {
    "key": "missed_positive_opportunity_count",
    "label": "긍정적 기회 놓치기 패턴",
    "value": 2,
    "value_type": "count"
    }
],
"challenge_evaluation": {
    "challenge_id": "chg-2025-01-positive-opportunity",
    "title": "긍정적 기회 놓치기 3회 도전",
    "goal_description": "아이가 성취나 행동을 공유했을 때 즉시 긍정적으로 반응한다.",
    "required_count": 3,
    "matched_count": 2,
    "is_success": false,
    "notes": "공유 장면은 3회였으나, 그중 2회만 즉시 긍정 반응을 해 목표에 조금 못 미쳤습니다.",
    "evidences": [
    {
        "index": 1,
        "dialogue": [
        {"speaker": "child", "text": "엄마, 이거 나 혼자 만들었어!"},
        {"speaker": "parent", "text": "와, 네가 혼자 했구나! 멋지다."}
        ],
        "matched": true,
        "reason": "성취 공유에 즉시 긍정 반응을 보여 챌린지 조건을 만족."
    },
    {
        "index": 2,
        "dialogue": [
        {"speaker": "child", "text": "이것도 봐줘."},
        {"speaker": "parent", "text": "잠깐만, 정리부터 하자."}
        ],
        "matched": false,
        "reason": "긍정적 반응보다 지시가 먼저 나와 조건 불충족."
    }
    ]
}
}
```

---

**실패한 경우:**
```json
{
    "execution_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "failed",
    "analysis_status": "failed",
    "progress_percentage": 0,
    "status_message": "분석 중 오류가 발생했습니다",
    "created_at": "2025-01-15T10:30:00",
    "updated_at": "2025-01-15T10:30:20",
    "current_node": null,
    "error": "API key is invalid",
    "result": null
}
```

---

### 3. 모든 실행 목록 조회
**GET** `/executions`

모든 실행 목록을 조회합니다.

**Response:**
```json
{
  "executions": [
    {
      "execution_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2025-01-15T10:30:00",
      "progress_percentage": 100
    },
    {
      "execution_id": "660e8400-e29b-41d4-a716-446655440001",
      "status": "running",
      "created_at": "2025-01-15T10:35:00",
      "progress_percentage": 40
    }
  ]
}
```

---

## 분석 상태 (AnalysisStatus)

각 상태는 이름, 진행률, 메시지를 포함합니다:

- `translating` (10%): "대화를 번역하는 중입니다"
- `labeling` (30%): "발화를 분류하는 중입니다"
- `pattern_detection` (50%): "패턴을 감지하는 중입니다"
- `style_analysis` (70%): "스타일을 분석하는 중입니다"
- `coaching_generation` (85%): "코칭을 생성하는 중입니다"
- `report_finalizing` (95%): "리포트를 작성하는 중입니다"
- `completed` (100%): "분석이 완료되었습니다"
- `failed` (0%): "분석 중 오류가 발생했습니다"


## 실행 상태 값

- `pending`: 대기 중
- `running`: 실행 중
- `completed`: 완료됨
- `failed`: 실패함

## 노드 실행 순서

1. **순차 처리:**
   - `preprocess` → `translate_ko_to_en` → `label_utterances` → `detect_patterns`

2. **병렬 처리:**
   - `summarize`, `key_moments`, `analyze_style`, `coaching_plan`, `challenge_eval`, `summary_diagnosis` (동시 실행)

3. **최종 집계:**
   - `aggregate_result` (모든 병렬 노드 완료 후 실행)

---

## Spring Boot RestClient 사용 예시

```java
@Service
public class LinkIdAiClient {
    
    private final RestClient restClient;
    private final String baseUrl = "http://localhost:8000";
    
    public LinkIdAiClient(RestClient.Builder restClientBuilder) {
        this.restClient = restClientBuilder
            .baseUrl(baseUrl)
            .build();
    }
    
    // 분석 요청
    public ExecutionResponse startAnalysis(DialogueRequest request) {
        return restClient.post()
            .uri("/analyze")
            .contentType(MediaType.APPLICATION_JSON)
            .body(request)
            .retrieve()
            .body(ExecutionResponse.class);
    }
    
    // 상태 조회
    public ExecutionStatusResponse getStatus(String executionId) {
        return restClient.get()
            .uri("/status/{executionId}", executionId)
            .retrieve()
            .body(ExecutionStatusResponse.class);
    }
    
    // 폴링으로 완료 대기
    public ExecutionStatusResponse waitForCompletion(String executionId, int maxWaitSeconds) {
        long startTime = System.currentTimeMillis();
        while (System.currentTimeMillis() - startTime < maxWaitSeconds * 1000) {
            ExecutionStatusResponse status = getStatus(executionId);
            if ("completed".equals(status.getStatus()) || "failed".equals(status.getStatus())) {
                return status;
            }
            try {
                Thread.sleep(1000); // 1초 대기
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        throw new RuntimeException("Timeout waiting for completion");
    }
}
```

