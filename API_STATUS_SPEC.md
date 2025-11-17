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
  "nodes": {
    "preprocess": {
      "status": "completed",
      "started_at": "2025-01-15T10:30:01",
      "completed_at": "2025-01-15T10:30:02"
    },
    "translate_ko_to_en": {
      "status": "completed",
      "started_at": "2025-01-15T10:30:02",
      "completed_at": "2025-01-15T10:30:05"
    },
    "label_utterances": {
      "status": "completed",
      "started_at": "2025-01-15T10:30:05",
      "completed_at": "2025-01-15T10:30:08"
    },
    "detect_patterns": {
      "status": "completed",
      "started_at": "2025-01-15T10:30:08",
      "completed_at": "2025-01-15T10:30:10"
    },
    "summarize": {
      "status": "running",
      "started_at": "2025-01-15T10:30:10"
    },
    "key_moments": {
      "status": "running",
      "started_at": "2025-01-15T10:30:10"
    },
    "analyze_style": {
      "status": "running",
      "started_at": "2025-01-15T10:30:10"
    },
    "coaching_plan": {
      "status": "pending"
    },
    "challenge_eval": {
      "status": "pending"
    },
    "aggregate_result": {
      "status": "pending"
    }
  },
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
  "nodes": {
    "preprocess": {"status": "completed", ...},
    "translate_ko_to_en": {"status": "completed", ...},
    "label_utterances": {"status": "completed", ...},
    "detect_patterns": {"status": "completed", ...},
    "summarize": {"status": "completed", ...},
    "key_moments": {"status": "completed", ...},
    "analyze_style": {"status": "completed", ...},
    "coaching_plan": {"status": "completed", ...},
    "challenge_eval": {"status": "completed", ...},
    "aggregate_result": {"status": "completed", ...}
  },
  "error": null,
  "result": {
    "summary": {...},
    "style_analysis": {...},
    "coaching_plan": {...},
    ...
  }
}
```

**실패한 경우:**
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "created_at": "2025-01-15T10:30:00",
  "updated_at": "2025-01-15T10:30:20",
  "current_node": "analyze_style",
  "progress_percentage": 60,
  "nodes": {
    "analyze_style": {
      "status": "failed",
      "started_at": "2025-01-15T10:30:10",
      "completed_at": "2025-01-15T10:30:20",
      "error": "API key is invalid"
    },
    ...
  },
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

## 노드 상태 값

- `pending`: 아직 실행되지 않음
- `running`: 현재 실행 중
- `completed`: 완료됨
- `failed`: 실패함
- `skipped`: 건너뜀

## 실행 상태 값

- `pending`: 대기 중
- `running`: 실행 중
- `completed`: 완료됨
- `failed`: 실패함

## 노드 실행 순서

1. **순차 처리:**
   - `preprocess` → `translate_ko_to_en` → `label_utterances` → `detect_patterns`

2. **병렬 처리:**
   - `summarize`, `key_moments`, `analyze_style`, `coaching_plan`, `challenge_eval` (동시 실행)

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

