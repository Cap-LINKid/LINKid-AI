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
            "relevant_indices: list of indices (0-based) where the parent performed the action. No extra text."
        ),
    ),
    (
        "human",
        (
            "Action to detect:\n{action_content}\n\n"
            "Challenge context:\n{challenge_name}\n\n"
            "All utterances (with index):\n{utterances_with_index}\n\n"
            "Identify parent utterances that demonstrate the action and return JSON with relevant_indices only."
        ),
    ),
])

_SUMMARY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyzing parent-child interactions. "
            "Generate a brief summary (in Korean) describing the situation where the parent performed the specific action. "
            "The summary should be concise (1-2 sentences) and describe what happened in this interaction moment. "
            "Return ONLY the summary text, no extra explanation."
        ),
    ),
    (
        "human",
        (
            "Action performed:\n{action_content}\n\n"
            "Challenge context:\n{challenge_name}\n\n"
            "Dialogue context:\n{dialogue_context}\n\n"
            "Generate a brief summary describing this interaction moment."
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
                return valid_indices
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
    
    # instances 생성 (순환 참조 방지를 위해 모든 값을 기본 타입으로 변환)
    instances = []
    for idx in relevant_indices[:10]:  # 최대 10개
        timestamp = _find_utterance_timestamp(utterances, idx)
        summary = _create_situation_summary(utterances, idx, action_content, challenge_name)
        
        instances.append({
            "timestamp": str(timestamp),
            "summary": str(summary)
        })
    
    # challenge_evaluation 반환 (순환 참조 방지를 위해 모든 값을 기본 타입으로 변환)
    return {
        "challenge_name": str(challenge_name),
        "action_id": int(action_id),
        "detected_count": int(len(relevant_indices)),
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
        for action_idx, action in enumerate(actions, start=1):
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
                action_idx, 
                utterances
            )
            
            # 수행된 action이 있으면 challenge_evaluations에 추가
            if evaluation:
                # evaluation은 이미 _evaluate_action에서 기본 타입으로 변환되어 반환됨
                # 깊은 복사하여 순환 참조 방지
                evaluation_copy = {
                    "challenge_name": evaluation.get("challenge_name", ""),
                    "action_id": evaluation.get("action_id", 0),
                    "detected_count": evaluation.get("detected_count", 0),
                    "description": evaluation.get("description", ""),
                    "instances": [
                        {
                            "timestamp": inst.get("timestamp", "00:00"),
                            "summary": inst.get("summary", "")
                        }
                        for inst in evaluation.get("instances", [])
                    ]
                }
                challenge_evaluations.append(evaluation_copy)
                # action_evaluations에는 evaluation 전체가 아닌 필요한 정보만 포함 (순환 참조 방지)
                action_evaluations.append({
                    "action_id": action_idx,
                    "action_content": action_content,
                    "detected_count": evaluation.get("detected_count", 0)
                })
        
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
        "challenge_evaluations": challenge_evaluations  # 수행된 action들의 challenge_evaluation 리스트
    }
