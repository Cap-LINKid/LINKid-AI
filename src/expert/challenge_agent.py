from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_ACTION_DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyzing parent-child interactions. "
            "Analyze the utterances and identify which parent utterances demonstrate the specific action. "
            "Return ONLY a JSON object with: {{relevant_indices: [list of utterance indices]}}. "
            "relevant_indices: list of indices (0-based) where the parent performed the action. No extra text.\n\n"
            "*** Important: Avoid Duplicate Detection ***\n"
            "- If multiple utterances express the same action with similar wording or meaning, select only ONE representative utterance (preferably the first or most clear one).\n"
            "- If utterances are very close in sequence (within 2-3 utterances) and express the same action, select only ONE.\n"
            "- Only include distinct instances where the action is clearly performed in a meaningfully different context or moment.\n"
            "- Focus on quality over quantity: it's better to return fewer, distinct instances than many similar ones.\n\n"
            "*** Important: Semantic and Polarity Check ***\n"
            "- The action description (e.g., '아이의 감정을 수용하고 긍정적으로 대화하기') often describes a clearly positive behavior.\n"
            "- **CRITICAL: Exclude CMD/NEG labeled utterances**: Utterances with label 'CMD' or 'NEG' (or any label containing 'CMD' or 'NEG') MUST be excluded from relevant_indices, regardless of their content. These labels indicate negative or controlling behaviors that should not be considered as successful action performance.\n"
            "- **CRITICAL: Q label context check**: Utterances with label 'Q' (or any label containing 'Q') require special attention. For Q-labeled utterances:\n"
            "  * Check the context: Look at the utterances immediately before and after (within 2-3 utterances) the Q-labeled utterance.\n"
            "  * Include ONLY if the context is positive: The Q-labeled utterance should be part of a positive interaction (e.g., genuine curiosity, supportive questioning, empathetic inquiry).\n"
            "  * Exclude if context is negative: If the surrounding context shows negative patterns (e.g., interrogation, criticism, scolding, dismissive questioning, or the child's responses indicate distress/defensiveness), exclude the Q-labeled utterance from relevant_indices.\n"
            "  * Examples of negative Q contexts to exclude: Questions that are part of a scolding sequence, questions that dismiss the child's feelings, questions that are clearly controlling or manipulative.\n"
            "  * Examples of positive Q contexts to include: Questions that show genuine interest in the child's feelings, questions that validate emotions, questions that are supportive and empathetic.\n"
            "- **Exclude clearly negative utterances**: Do NOT mark utterances as relevant if they are unambiguously criticizing, scolding, dismissing emotions, or controlling in a negative way (e.g., \"왜 이렇게 화가 나냐고 도대체 몰라\", \"예쁘게 말해\" - these are clearly negative).\n"
            "- **Include neutral or positive utterances**: If an utterance is neutral, supportive, or shows any attempt at the positive behavior (even if imperfect), mark it as relevant.\n"
            "- **Default to inclusion**: If the utterance is not clearly negative (i.e., \"누가봐도 부정적인 것이 아니라면\") and does NOT have CMD/NEG label, and if it has Q label, the context is positive, include it as a successful action performance.\n"
            "- A relevant utterance should show the parent performing or attempting the positive behavior described in the action (e.g., accepting feelings, validating emotions, speaking calmly and supportively, asking about feelings, etc.)."
        ),
    ),
    (
        "human",
        (
            "Action to detect:\n{action_content}\n\n"
            "Challenge context:\n{challenge_name}\n\n"
            "All utterances (with index):\n{utterances_with_index}\n\n"
            "Identify parent utterances that demonstrate the action. "
            "Avoid duplicates: if multiple utterances express the same action, select only the most representative one.\n"
            "**CRITICAL**: MUST exclude utterances with label 'CMD' or 'NEG' (or any label containing 'CMD' or 'NEG') from relevant_indices.\n"
            "**CRITICAL for Q-labeled utterances**: For utterances with label 'Q' (or containing 'Q'), you MUST check the surrounding context (2-3 utterances before and after). "
            "Include the Q-labeled utterance ONLY if the context shows a positive interaction (genuine curiosity, supportive questioning, empathetic inquiry). "
            "Exclude it if the context shows negative patterns (interrogation, criticism, scolding, dismissive questioning, or child's distress/defensiveness).\n"
            "**Important**: Only exclude utterances that are clearly negative (criticizing, scolding, dismissing). "
            "If an utterance is neutral or shows any positive attempt, include it. "
            "When the utterance is not clearly negative (\"누가봐도 부정적인 것이 아니라면\") and does NOT have CMD/NEG label, and if it has Q label, the context is positive, mark it as relevant. "
            "Return JSON with relevant_indices only."
        ),
    ),
])

_SUMMARY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyzing parent-child interactions. "
            "Using ONLY the dialogue content provided, generate a brief summary (in Korean) describing why this interaction moment demonstrates successful action performance.\n"
            "- **CRITICAL: Use ONLY actual dialogue, NEVER action description examples**:\n"
            "  - The 'Action performed' field contains EXAMPLE phrases (e.g., '그런 마음이었구나, 궁금한 건 이해해').\n"
            "  - You MUST NEVER quote or use these example phrases unless they appear EXACTLY in the 'Dialogue context'.\n"
            "  - If the example phrase does not appear word-for-word in the dialogue context, you MUST use the actual parent utterance from the dialogue instead.\n"
            "  - DO NOT fabricate or assume the parent said the example phrase. Only quote what actually appears in the dialogue context.\n"
            "- **Important**: This summary is generated ONLY when the action was successfully detected. Focus on the positive aspects that made it successful.\n"
            "- The summary MUST be grounded in the 'Dialogue context' utterances, not in the action description text.\n"
            "- If you quote what the parent said, you MUST copy the exact Korean utterance from the dialogue context (e.g., 부모: ...), without changing or inventing new wording.\n"
            "- **Focus on success**: Describe what the parent actually said/did (from dialogue context) and why it was effective.\n"
            "- **Do NOT mention negative aspects**: Even if the dialogue contains some negative elements, do NOT mention them in the summary. Focus only on the positive action that was successfully performed.\n"
            "- **Explain the positive impact**: Describe what positive effect the parent's words/actions had (e.g., how it helped the child feel understood, validated, or supported).\n"
            "- Prefer to include at least one short direct quote from the parent's actual utterance in quotation marks, taken verbatim from the dialogue context.\n"
            "- Structure: (1) What the parent actually said/did (with direct quote from dialogue), (2) Why it was effective or what positive impact it had.\n"
            "The Korean summary MUST be written in polite formal speech (존댓말, e.g., '~합니다', '~합니다.'). "
            "Return ONLY the summary text, no extra explanation."
        ),
    ),
    (
        "human",
        (
            "Action performed:\n{action_content}\n\n"
            "Challenge context:\n{challenge_name}\n\n"
            "Dialogue context:\n{dialogue_context}\n\n"
            "Generate a brief summary explaining why this interaction moment demonstrates successful action performance.\n"
            "**CRITICAL**: The 'Action performed' field contains EXAMPLE phrases. You MUST NEVER quote or use these example phrases unless they appear EXACTLY word-for-word in the 'Dialogue context' above.\n"
            "Only quote what the parent actually said in the dialogue context. If the example phrase is not in the dialogue, use the actual parent utterance from the dialogue instead.\n"
            "Focus on what the parent actually said/did (from dialogue context) that was positive and effective, and what positive impact it had. "
            "Do NOT mention any negative aspects."
        ),
    ),
])


def _format_timestamp(timestamp_ms: Optional[int]) -> str:
    """밀리초 타임스탬프를 MM:SS 형식으로 변환"""
    if timestamp_ms is None:
        return "00:00"
    
    total_seconds = timestamp_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    return f"{minutes:02d}:{seconds:02d}"


def _get_timestamp_ms(utterances: List[Dict[str, Any]], utterance_idx: int) -> int:
    """발화 인덱스에 해당하는 타임스탬프를 밀리초로 반환"""
    if utterance_idx < 0 or utterance_idx >= len(utterances):
        return 0
    
    utt = utterances[utterance_idx]
    if not isinstance(utt, dict):
        return 0
    
    timestamp_ms = utt.get("timestamp")
    if timestamp_ms is None or timestamp_ms == 0:
        timestamp_ms = utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
    
    if timestamp_ms is not None and timestamp_ms != 0:
        try:
            return int(timestamp_ms)
        except (ValueError, TypeError):
            return 0
    
    return 0


def _find_utterance_timestamp(
    utterances: List[Dict[str, Any]], 
    utterance_idx: int
) -> str:
    """발화 인덱스에 해당하는 타임스탬프를 찾아서 반환"""
    if utterance_idx < 0 or utterance_idx >= len(utterances):
        return "00:00"
    
    utt = utterances[utterance_idx]
    if not isinstance(utt, dict):
        return "00:00"
    
    # timestamp 필드에서 직접 가져오기
    timestamp_ms = utt.get("timestamp")
    
    # timestamp가 None이거나 0이면 다른 필드에서 찾기 시도
    if timestamp_ms is None or timestamp_ms == 0:
        # utterances가 원본 데이터 구조를 가지고 있을 수 있으므로
        # 다른 가능한 필드명도 확인
        timestamp_ms = utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
    
    if timestamp_ms is not None and timestamp_ms != 0:
        try:
            return _format_timestamp(int(timestamp_ms))
        except (ValueError, TypeError):
            return "00:00"
    
    return "00:00"


def _create_situation_summary(
    utterances: List[Dict[str, Any]], 
    parent_idx: int,
    action_content: str,
    challenge_name: str = "",
    context_window: int = 2
) -> str:
    """발화 인덱스를 중심으로 상황 요약 생성 (LLM 사용)"""
    if parent_idx < 0 or parent_idx >= len(utterances):
        return "상황이 감지되었습니다."
    
    # 주변 발화 수집 (한국어 원본 우선 사용)
    start_idx = max(0, parent_idx - context_window)
    end_idx = min(len(utterances), parent_idx + context_window + 1)
    
    dialogue_context_parts = []
    
    for idx in range(start_idx, end_idx):
        if idx >= len(utterances):
            break
        utt = utterances[idx]
        if not isinstance(utt, dict):
            continue
        
        speaker = utt.get("speaker", "").lower()
        # 한국어 원본 우선 사용
        text = (utt.get("original_ko", "") or 
                utt.get("korean", "") or 
                utt.get("text", "")).strip()
        
        if not text:
            continue
        
        # 발화자 표시
        if speaker in ["parent", "mom", "mother", "부모", "엄마", "아빠"]:
            dialogue_context_parts.append(f"부모: {text}")
        elif speaker in ["child", "chi", "kid", "아이", "자녀"]:
            dialogue_context_parts.append(f"아이: {text}")
        else:
            dialogue_context_parts.append(f"{speaker}: {text}")
    
    if not dialogue_context_parts:
        return "상황이 감지되었습니다."
    
    dialogue_context = "\n".join(dialogue_context_parts)
    
    # LLM을 사용하여 요약 생성
    try:
        llm = get_llm(mini=True)  # 빠른 응답을 위해 mini 모델 사용
        
        res = (_SUMMARY_GENERATION_PROMPT | llm).invoke({
            "action_content": action_content,
            "challenge_name": challenge_name or "",
            "dialogue_context": dialogue_context,
        })
        content = getattr(res, "content", "") or str(res)
        
        # 응답 정리 (앞뒤 공백 제거, 따옴표 제거)
        summary = content.strip().strip('"').strip("'")
        
        if summary:
            return summary
    except Exception as e:
        print(f"Summary generation LLM error: {e}")
    
    # 폴백: 간단한 요약 생성
    if dialogue_context_parts:
        first_part = dialogue_context_parts[0]
        return f"{first_part} 상황이 감지되었습니다."
    
    return "상황이 감지되었습니다."


def _find_relevant_utterances_with_llm(
    utterances: List[Dict[str, Any]], 
    action_content: str, 
    challenge_name: str
) -> List[int]:
    """LLM을 사용하여 action과 관련된 발화 인덱스 찾기"""
    if not utterances or not action_content:
        return []
    
    llm = get_llm(mini=True)  # 빠른 응답을 위해 mini 모델 사용
    
    # 발화를 인덱스와 함께 포맷팅 (부모 발화만 포함)
    utterances_with_index = []
    for idx, utt in enumerate(utterances):
        speaker = utt.get("speaker", "")
        text = (utt.get("text", "") or 
                utt.get("original_ko", "") or 
                utt.get("korean", "") or 
                utt.get("english", ""))
        label = utt.get("label", "")
        
        # 부모 발화만 포함
        if speaker.lower() in ["parent", "mom", "mother", "부모", "엄마", "아빠"]:
            utterances_with_index.append(
                f"[{idx}] [{speaker}] [{label}] {text}"
            )
    
    if not utterances_with_index:
        return []
    
    utterances_str = "\n".join(utterances_with_index)
    
    try:
        res = (_ACTION_DETECTION_PROMPT | llm).invoke({
            "action_content": action_content,
            "challenge_name": challenge_name,
            "utterances_with_index": utterances_str,
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 객체 파싱
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            if isinstance(result, dict):
                relevant_indices = result.get("relevant_indices", [])
                # 정수 리스트로 변환 및 유효성 검사
                valid_indices = [
                    int(idx) for idx in relevant_indices 
                    if isinstance(idx, (int, str)) and str(idx).isdigit() and 0 <= int(idx) < len(utterances)
                ]
                
                # CMD/NEG 라벨이 있는 발화 필터링 (코드 레벨 가드레일)
                filtered_indices = []
                for idx in valid_indices:
                    if 0 <= idx < len(utterances):
                        utt = utterances[idx]
                        if isinstance(utt, dict):
                            label = str(utt.get("label", "")).upper()
                            # CMD나 NEG가 포함된 라벨은 제외
                            if "CMD" in label or "NEG" in label:
                                print(f"DEBUG: [Action Detection] Excluding utterance [{idx}] with label '{label}' (CMD/NEG filter)")
                                continue
                    filtered_indices.append(idx)
                
                return filtered_indices
    except Exception as e:
        print(f"Action detection LLM error: {e}")
    
    # 폴백: 빈 리스트 반환
    return []


def _evaluate_action(
    challenge_spec: Dict[str, Any],
    action_content: str,
    action_id: int,
    utterances: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """단일 action 평가 - 수행된 action이 있으면 challenge_evaluation 반환"""
    challenge_name = challenge_spec.get("title", "")
    
    # LLM을 사용하여 action과 관련된 발화 인덱스 찾기
    relevant_indices = _find_relevant_utterances_with_llm(
        utterances, action_content, challenge_name
    )
    
    # 수행된 action이 없으면 None 반환
    if not relevant_indices:
        return None
    
    # 중복 제거: 시간적으로 가까운 인스턴스들을 그룹화 (15초 이내)
    # 타임스탬프 기준으로 정렬
    indices_with_time = [
        (idx, _get_timestamp_ms(utterances, idx))
        for idx in relevant_indices
    ]
    indices_with_time.sort(key=lambda x: x[1])  # 타임스탬프 기준 정렬
    
    # 시간적으로 가까운 인스턴스들을 그룹화 (15초 = 15000ms 이내)
    TIME_THRESHOLD_MS = 15000
    deduplicated_indices = []
    
    if indices_with_time:
        current_group = [indices_with_time[0]]
        for idx, ts_ms in indices_with_time[1:]:
            # 현재 그룹의 마지막 타임스탬프와 비교
            last_ts = current_group[-1][1]
            if ts_ms - last_ts <= TIME_THRESHOLD_MS:
                # 같은 그룹에 추가
                current_group.append((idx, ts_ms))
            else:
                # 새로운 그룹 시작: 이전 그룹에서 하나만 선택 (첫 번째 것)
                deduplicated_indices.append(current_group[0][0])
                current_group = [(idx, ts_ms)]
        
        # 마지막 그룹 처리
        if current_group:
            deduplicated_indices.append(current_group[0][0])
    
    # action당 하나의 instance만 선택 (첫 번째 것)
    if not deduplicated_indices:
        return None
    
    # 첫 번째 인덱스만 사용
    selected_idx = deduplicated_indices[0]
    
    # instance 생성 (순환 참조 방지를 위해 모든 값을 기본 타입으로 변환)
    timestamp = _find_utterance_timestamp(utterances, selected_idx)
    summary = _create_situation_summary(utterances, selected_idx, action_content, challenge_name)
    
    instances = [{
        "timestamp": str(timestamp),
        "summary": str(summary)
    }]
    
    # challenge_evaluation 반환 (순환 참조 방지를 위해 모든 값을 기본 타입으로 변환)
    return {
        "challenge_name": str(challenge_name),
        "action_id": int(action_id),
        "detected_count": 1,  # instance가 하나이므로 항상 1
        "description": str(action_content),
        "instances": instances
    }


def challenge_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑨ challenge_eval: 챌린지 판정
    challenge_specs를 받아서 각 challenge의 각 action에 대해 수행 여부를 확인하고,
    수행된 action이 있으면 challenge_evaluation 형태로 반환
    """
    # challenge_specs 우선, 없으면 challenge_spec을 리스트로 변환
    challenge_specs = state.get("challenge_specs") or []
    if not challenge_specs:
        challenge_spec = state.get("challenge_spec") or {}
        if challenge_spec:
            challenge_specs = [challenge_spec]
    
    utterances_labeled = state.get("utterances_labeled") or []
    utterances_ko = state.get("utterances_ko") or []
    
    # utterances_labeled 우선 사용, 없으면 utterances_ko 사용
    # 순환 참조 방지를 위해 필요한 필드만 추출하여 새로운 리스트 생성
    utterances_raw = utterances_labeled or utterances_ko or []
    utterances = []
    for utt in utterances_raw:
        if isinstance(utt, dict):
            # timestamp 추출 (여러 가능한 필드명 확인)
            timestamp = utt.get("timestamp") or utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
            # 필요한 필드만 추출하여 새로운 딕셔너리 생성
            utterances.append({
                "speaker": str(utt.get("speaker", "")),
                "text": str(utt.get("text", "") or utt.get("original_ko", "") or utt.get("korean", "") or ""),
                "label": str(utt.get("label", "")),
                "timestamp": timestamp,  # timestamp 포함 (int 또는 None)
                "original_ko": str(utt.get("original_ko", "") or utt.get("korean", "") or ""),
                "korean": str(utt.get("korean", "") or ""),
                "english": str(utt.get("english", "") or "")
            })
        else:
            utterances.append(utt)
    
    if not challenge_specs:
        return {
            "challenge_eval": {},
            "challenge_evals": []
        }
    
    # 각 challenge별로 action 평가
    challenge_evals = []
    challenge_evaluations = []
    
    for challenge_spec in challenge_specs:
        # challenge_spec에서 필요한 값만 추출 (순환 참조 방지)
        challenge_id = str(challenge_spec.get("challenge_id", ""))
        challenge_name = str(challenge_spec.get("title", ""))
        actions = challenge_spec.get("actions", [])
        
        # actions를 복사 (순환 참조 방지)
        if isinstance(actions, list):
            actions = list(actions)  # 얕은 복사
        
        if not actions:
            # actions가 없으면 빈 결과 반환
            challenge_evals.append({
                "challenge_id": challenge_id,
                "challenge_title": challenge_name
            })
            continue
        
        # 각 action별로 평가
        action_evaluations = []
        challenge_actions_dict = {}  # action_id를 키로 사용하여 중복 제거
        
        for action_idx, action in enumerate(actions, start=1):
            action_idx_real = int(action.get("action_id", 1))
            # action이 문자열인 경우와 딕셔너리인 경우 모두 처리
            if isinstance(action, str):
                action_content = str(action)
            elif isinstance(action, dict):
                action_content = str(action.get("content", "") or action.get("text", ""))
            else:
                continue
            
            if not action_content:
                continue
            
            # action 평가 (challenge_spec 대신 필요한 값만 전달)
            evaluation = _evaluate_action(
                {"title": challenge_name},  # challenge_spec 대신 필요한 값만 전달
                action_content, 
                action_idx_real,
                utterances
            )
            
            # 수행된 action이 있으면 challenge_actions_dict에 추가 (중복 제거)
            if evaluation:
                action_id = int(evaluation.get("action_id", 0))
                
                # challenge_evaluations용 action 데이터 생성
                action_data = {
                    "action_id": action_id,
                    "detected_count": int(evaluation.get("detected_count", 0)),
                    "description": str(evaluation.get("description", "") or ""),
                    "instances": [
                        {
                            "timestamp": inst.get("timestamp", "00:00"),
                            "summary": inst.get("summary", "")
                        }
                        for inst in evaluation.get("instances", [])
                    ]
                }
                
                # 같은 action_id가 이미 있으면, detected_count가 더 큰 것을 선택
                if action_id in challenge_actions_dict:
                    existing_count = challenge_actions_dict[action_id].get("detected_count", 0)
                    new_count = action_data.get("detected_count", 0)
                    if new_count > existing_count:
                        challenge_actions_dict[action_id] = action_data
                else:
                    challenge_actions_dict[action_id] = action_data
                
                # action_evaluations에는 evaluation 전체가 아닌 필요한 정보만 포함 (순환 참조 방지)
                action_evaluations.append({
                    "action_id": action_idx_real,
                    "action_content": action_content,
                    "description": str(evaluation.get("description", "") or ""),
                    "detected_count": evaluation.get("detected_count", 0)
                })
        
        # challenge_actions_dict에서 이 챌린지에 대해 "가장 잘 수행된" 하나의 액션만 선택
        challenge_actions: List[Dict[str, Any]] = []
        if challenge_actions_dict:
            # detected_count가 가장 큰 action_id 선택 (동점이면 먼저 생성된 것 우선)
            best_action_id = max(
                challenge_actions_dict.items(),
                key=lambda item: item[1].get("detected_count", 0)
            )[0]
            best_action = challenge_actions_dict.get(best_action_id)
            if best_action:
                challenge_actions = [best_action]
                # action_evaluations에서도 동일한 action_id만 남기기
                action_evaluations = [
                    ae for ae in action_evaluations
                    if int(ae.get("action_id", 0)) == int(best_action_id)
                ]
        
        # challenge_evaluations에 challenge 단위로 추가 (actions가 있는 경우만)
        if challenge_actions:
            challenge_evaluation = {
                "challenge_name": str(challenge_name),
                "actions": challenge_actions
            }
            challenge_evaluations.append(challenge_evaluation)
        
        # challenge_eval 결과 생성 (순환 참조 방지)
        challenge_evals.append({
            "challenge_id": challenge_id,
            "challenge_title": challenge_name,
            "action_evaluations": action_evaluations
        })
    
    # 하위 호환성을 위해 첫 번째 챌린지 결과를 challenge_eval로도 반환
    first_eval = challenge_evals[0].copy() if challenge_evals else {}
    
    return {
        "challenge_eval": first_eval,  # 하위 호환성
        "challenge_evals": challenge_evals,  # 여러 챌린지 결과
        "challenge_evaluations": challenge_evaluations  # challenge 단위로 그룹화된 challenge_evaluation 리스트
    }
