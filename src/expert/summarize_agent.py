from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_COMMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a parenting coach analyzing parent-child dialogue. "
            "Write a brief comment (1-2 sentences in Korean) summarizing today's interaction. "
            "Focus on key observations about communication patterns. "
            "Return ONLY the comment text, no extra formatting."
        ),
    ),
    (
        "human",
        (
            "스타일 분석:\n{style_analysis}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "오늘의 대화에 대한 간단한 코멘트를 작성해주세요."
        ),
    ),
])




def _extract_metrics_from_style_analysis(style_analysis: Dict[str, Any]) -> list:
    """style_analysis에서 메트릭 추출 (style_agent.py의 결과 재사용)"""
    metrics = []
    
    if not style_analysis:
        return metrics
    
    # style_analysis가 {"style_analysis": {...}} 형식일 수 있음
    actual_style_analysis = style_analysis
    if isinstance(style_analysis, dict) and "style_analysis" in style_analysis:
        actual_style_analysis = style_analysis["style_analysis"]
    
    if not isinstance(actual_style_analysis, dict) or "interaction_style" not in actual_style_analysis:
        return metrics
    
    parent_analysis = actual_style_analysis.get("interaction_style", {}).get("parent_analysis", {})
    categories = parent_analysis.get("categories", [])
    
    if not categories:
        return metrics
    
    # 주요 메트릭 키 매핑 (label 기반)
    metric_key_mapping = {
        "RD": "reflective_listening_ratio",
        "CMD": "directive_speech_ratio",
        "PR": "praise_ratio",
        "Q": "question_ratio",
        "NEG": "negative_feedback_ratio",
        "NT": "neutral_talk_ratio",
        "BD": "behavior_description_ratio",
    }
    
    for cat in categories:
        if not isinstance(cat, dict):
            continue
            
        label = cat.get("label", "")
        name = cat.get("name", "")
        ratio = cat.get("ratio", 0.0)
        
        # 라벨로 메트릭 키 찾기
        metric_key = metric_key_mapping.get(label)
        if metric_key:
            metrics.append({
                "key": metric_key,
                "label": name,  # style_agent.py에서 이미 한국어 이름으로 변환됨
                "value": ratio,
                "value_type": "ratio"
            })
    
    return metrics


def _extract_pattern_count_metrics(patterns: list, key_moments: Dict[str, Any]) -> list:
    """패턴에서 카운트 메트릭 추출"""
    metrics = []
    
    # key_moments에서 pattern_examples 가져오기
    pattern_examples = []
    if isinstance(key_moments, dict):
        pattern_examples = key_moments.get("pattern_examples", [])
    
    # 패턴별 카운트 집계
    pattern_counts = {}
    for p in pattern_examples:
        pattern_name = p.get("pattern_name", "")
        occurrences = p.get("occurrences", 0)
        if pattern_name:
            pattern_counts[pattern_name] = occurrences
    
    # 주요 패턴 메트릭 추가
    for pattern_name, count in pattern_counts.items():
        if "긍정적 기회 놓치기" in pattern_name or "긍정기회놓치기" in pattern_name:
            metrics.append({
                "key": "missed_positive_opportunity_count",
                "label": "긍정적 기회 놓치기 패턴",
                "value": count,
                "value_type": "count"
            })
    
    return metrics


def _format_timestamp(timestamp_ms: Optional[int]) -> str:
    """밀리초 타임스탬프를 MM:SS 형식으로 변환"""
    if timestamp_ms is None:
        return "00:00"
    
    total_seconds = timestamp_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    return f"{minutes:02d}:{seconds:02d}"


def _create_situation_description(dialogue_list: list, context_hint: str = "", is_positive: bool = True) -> str:
    """dialogue를 기반으로 직접적인 상황 묘사 생성"""
    if not dialogue_list:
        return context_hint if context_hint else "상황이 감지되었습니다."
    
    # dialogue에서 발화자와 텍스트 추출
    parent_texts = []
    child_texts = []
    
    for d in dialogue_list:
        speaker = d.get("speaker", "").lower()
        text = d.get("text", "").strip()
        if not text:
            continue
            
        if speaker in ["parent", "mom", "mother", "부모", "엄마", "아빠"]:
            parent_texts.append(text)
        elif speaker in ["child", "chi", "kid", "아이", "자녀"]:
            child_texts.append(text)
    
    # 상황 묘사 생성 (더 구체적으로)
    if is_positive:
        # 긍정적 순간: 부모의 긍정적 응답을 구체적으로 묘사
        if child_texts and parent_texts:
            child_said = child_texts[0]
            parent_response = parent_texts[0]
            
            # context_hint에서 핵심 키워드 추출 (예: "감정을 먼저 짚어주셨습니다", "반영형으로 응답하셨습니다")
            if context_hint:
                # context_hint에서 핵심 동작 추출
                key_actions = []
                if "감정" in context_hint or "공감" in context_hint:
                    key_actions.append("감정을 먼저 짚어주셨습니다")
                if "반영" in context_hint or "그랬구나" in context_hint or "그렇구나" in context_hint:
                    key_actions.append("반영형으로 응답하셨습니다")
                if "칭찬" in context_hint or "좋아" in context_hint:
                    key_actions.append("칭찬해주셨습니다")
                if "선택" in context_hint or "선택권" in context_hint:
                    key_actions.append("선택권을 제시하셨습니다")
                
                if key_actions:
                    child_summary = child_said[:25] + "..." if len(child_said) > 25 else child_said
                    return f"아이가 '{child_summary}'라고 말했을 때 {key_actions[0]}."
            
            # context_hint가 없거나 키워드를 찾지 못한 경우
            child_summary = child_said[:25] + "..." if len(child_said) > 25 else child_said
            parent_summary = parent_response[:30] + "..." if len(parent_response) > 30 else parent_response
            return f"아이가 '{child_summary}'라고 말했을 때 '{parent_summary}'라고 응답하셨습니다."
        elif parent_texts:
            parent_said = parent_texts[0]
            parent_summary = parent_said[:40] + "..." if len(parent_said) > 40 else parent_said
            return f"'{parent_summary}'라고 말씀하셨습니다."
        else:
            return context_hint if context_hint else "긍정적 상호작용이 있었습니다."
    else:
        # 개선이 필요한 순간: 상황을 객관적으로 묘사
        if child_texts and parent_texts:
            child_said = child_texts[0]
            parent_said = parent_texts[0]
            child_summary = child_said[:25] + "..." if len(child_said) > 25 else child_said
            parent_summary = parent_said[:30] + "..." if len(parent_said) > 30 else parent_said
            return f"아이가 '{child_summary}'라고 말했을 때 '{parent_summary}'라고 응답하셨습니다."
        elif parent_texts:
            parent_said = parent_texts[0]
            parent_summary = parent_said[:40] + "..." if len(parent_said) > 40 else parent_said
            return f"'{parent_summary}'라고 말씀하셨습니다."
        else:
            return context_hint if context_hint else "개선이 필요한 순간이 있었습니다."


def _find_utterance_timestamp(dialogue_list: list, utterances_labeled: list, utterances_ko: list = None) -> str:
    """dialogue_list에서 첫 번째 발화의 타임스탬프를 찾아서 반환"""
    if not dialogue_list:
        return "00:00"
    
    # dialogue_list의 첫 번째 발화 텍스트로 매칭
    first_dialogue = dialogue_list[0]
    dialogue_text = first_dialogue.get("text", "")
    dialogue_speaker = first_dialogue.get("speaker", "").lower()
    
    # utterances_labeled에서 매칭되는 발화 찾기
    if utterances_labeled:
        for idx, utt in enumerate(utterances_labeled):
            original_ko = utt.get("original_ko", utt.get("korean", utt.get("text", "")))
            utt_speaker = utt.get("speaker", "").lower()
            
            # 발화자 매칭
            speaker_match = False
            if dialogue_speaker in ['parent', 'mom', 'mother']:
                speaker_match = utt_speaker in ['parent', 'mom', 'mother', '엄마', '아빠']
            elif dialogue_speaker in ['child', 'chi', 'kid']:
                speaker_match = utt_speaker in ['child', 'chi', 'kid', '아이']
            else:
                speaker_match = True  # 발화자가 없으면 무시
            
            # 텍스트 매칭 (더 정확한 매칭)
            text_match = (
                dialogue_text in original_ko or 
                original_ko in dialogue_text or
                dialogue_text.strip() == original_ko.strip()
            )
            
            if speaker_match and text_match:
                # utterances_labeled에 timestamp가 있으면 사용
                timestamp_ms = utt.get("timestamp")
                if timestamp_ms is not None:
                    return _format_timestamp(timestamp_ms)
                
                # utterances_ko에서 timestamp 찾기 (인덱스 기반)
                if utterances_ko and idx < len(utterances_ko):
                    ko_utt = utterances_ko[idx]
                    if isinstance(ko_utt, dict):
                        timestamp_ms = ko_utt.get("timestamp")
                        if timestamp_ms is not None:
                            return _format_timestamp(timestamp_ms)
    
    # utterances_ko에서 직접 찾기 (텍스트 매칭)
    if utterances_ko:
        for ko_utt in utterances_ko:
            if isinstance(ko_utt, dict):
                ko_text = ko_utt.get("text", "")
                if dialogue_text in ko_text or ko_text in dialogue_text:
                    timestamp_ms = ko_utt.get("timestamp")
                    if timestamp_ms is not None:
                        return _format_timestamp(timestamp_ms)
    
    return "00:00"


def _is_relevant_to_challenge(challenge_name: str, description: str, pattern_or_moment: Dict[str, Any]) -> bool:
    """패턴이나 moment가 챌린지와 관련이 있는지 확인"""
    # 챌린지 키워드 추출
    challenge_keywords = []
    challenge_text = f"{challenge_name} {description}".lower()
    
    # 챌린지별 키워드 매핑
    if "질문" in challenge_text or "question" in challenge_text or "q" in challenge_text:
        challenge_keywords.extend(["질문", "question", "q", "어떻게", "무엇", "어디", "언제", "왜", "어떤"])
    if "반영" in challenge_text or "reflection" in challenge_text or "rd" in challenge_text:
        challenge_keywords.extend(["반영", "reflection", "rd", "그랬구나", "그렇구나", "그런가", "그런구나"])
    if "칭찬" in challenge_text or "praise" in challenge_text or "pr" in challenge_text:
        challenge_keywords.extend(["칭찬", "praise", "pr", "좋아", "잘했", "멋져"])
    if "긍정" in challenge_text or "positive" in challenge_text:
        challenge_keywords.extend(["긍정", "positive", "좋아", "칭찬"])
    if "명령" in challenge_text or "command" in challenge_text or "cmd" in challenge_text:
        challenge_keywords.extend(["명령", "command", "cmd", "해라", "해야", "하세요"])
    
    # 패턴명 확인
    pattern_name = pattern_or_moment.get("pattern_name", "").lower()
    if pattern_name and any(kw in pattern_name for kw in challenge_keywords if len(kw) > 2):
        return True
    
    # dialogue 내용 확인
    dialogue_list = pattern_or_moment.get("dialogue", [])
    dialogue_text = " ".join([d.get("text", "") for d in dialogue_list]).lower()
    if dialogue_text and any(kw in dialogue_text for kw in challenge_keywords if len(kw) > 2):
        return True
    
    # reason이나 problem_explanation 확인
    reason = (pattern_or_moment.get("reason", "") or 
              pattern_or_moment.get("problem_explanation", "") or 
              pattern_or_moment.get("suggested_response", "")).lower()
    if reason and any(kw in reason for kw in challenge_keywords if len(kw) > 2):
        return True
    
    # 챌린지 이름이 패턴명에 포함되거나 그 반대인 경우
    if challenge_name and pattern_name:
        challenge_name_lower = challenge_name.lower()
        if challenge_name_lower in pattern_name or pattern_name in challenge_name_lower:
            return True
    
    return False


def _format_challenge_evaluation(challenge_eval: Dict[str, Any], challenge_spec: Dict[str, Any], 
                                  utterances_labeled: list, key_moments: Dict[str, Any], 
                                  utterances_ko: list = None) -> Dict[str, Any]:
    """challenge_eval을 요청된 형식으로 변환"""
    if not challenge_eval:
        return {}
    
    # challenge_spec에서 정보 가져오기
    challenge_name = challenge_spec.get("title", "")
    description = challenge_spec.get("goal", "") or challenge_spec.get("description", "")
    
    # actions 가져오기 (challenge_spec.challenge.actions 또는 challenge_spec.actions)
    actions = []
    if "challenge" in challenge_spec and isinstance(challenge_spec["challenge"], dict):
        actions = challenge_spec["challenge"].get("actions", [])
    if not actions:
        actions = challenge_spec.get("actions", [])
    
    # challenge_eval에서 정보 가져오기
    is_success = challenge_eval.get("challenge_met", False)
    
    # instances 생성 (key_moments의 positive, needs_improvement, pattern_examples 활용)
    instances = []
    detected_count = 0
    
    if isinstance(key_moments, dict):
        # 챌린지가 성공했을 때는 positive moments 사용, 실패했을 때는 pattern_examples 사용
        if is_success:
            # 성공 시: positive moments 사용 (챌린지와 관련된 것만 필터링)
            positive_moments = key_moments.get("positive", [])
            
            # 챌린지와 관련된 positive moments만 필터링
            relevant_moments = [
                moment for moment in positive_moments
                if _is_relevant_to_challenge(challenge_name, description, moment)
            ]
            
            detected_count = len(relevant_moments)
            
            for moment in relevant_moments[:10]:  # 최대 10개
                dialogue_list = moment.get("dialogue", [])
                if dialogue_list:
                    # timestamp 찾기
                    timestamp = _find_utterance_timestamp(dialogue_list, utterances_labeled, utterances_ko)
                    
                    # summary 생성: dialogue 기반 직접적인 상황 묘사
                    summary = _create_situation_description(dialogue_list, moment.get("reason", ""), is_positive=True)
                    
                    instances.append({
                        "timestamp": timestamp,
                        "summary": summary,
                        "dialogue": dialogue_list
                    })
        else:
            # 실패 시: pattern_examples 사용 (챌린지와 관련된 것만 필터링)
            pattern_examples = key_moments.get("pattern_examples", [])
            
            # 챌린지와 관련된 패턴만 필터링
            relevant_patterns = [
                pattern for pattern in pattern_examples
                if _is_relevant_to_challenge(challenge_name, description, pattern)
            ]
            
            # detected_count 계산 (occurrences 합계)
            detected_count = sum(p.get("occurrences", 1) for p in relevant_patterns)
            
            # instances 생성
            for pattern in relevant_patterns[:10]:  # 최대 10개
                dialogue_list = pattern.get("dialogue", [])
                if dialogue_list:
                    # timestamp 찾기
                    timestamp = _find_utterance_timestamp(dialogue_list, utterances_labeled, utterances_ko)
                    
                    # summary 생성: dialogue 기반 직접적인 상황 묘사
                    problem_explanation = pattern.get("problem_explanation", "") or pattern.get("suggested_response", "")
                    summary = _create_situation_description(dialogue_list, problem_explanation, is_positive=False)
                    
                    instances.append({
                        "timestamp": timestamp,
                        "summary": summary,
                        "dialogue": dialogue_list
                    })
    
    result = {
        "challenge_name": challenge_name,
        "detected_count": detected_count,
        "description": description,
        "instances": instances
    }
    
    # actions가 있으면 포함
    if actions:
        result["actions"] = actions
    
    return result


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑤ summarize: 오늘의 진단 (새로운 JSON 형식)
    """
    utterances_labeled = state.get("utterances_labeled") or []
    utterances_ko = state.get("utterances_ko") or []
    patterns = state.get("patterns") or []
    style_analysis = state.get("style_analysis") or {}
    challenge_eval = state.get("challenge_eval") or {}
    challenge_spec = state.get("challenge_spec") or {}
    key_moments = state.get("key_moments") or {}
    meta = state.get("meta") or {}
    
    if not utterances_labeled:
        return {
            "summary": {
                "analysis_session": {
                    "comment": "대화 내용이 없어 분석할 수 없습니다."
                },
                "current_metrics": [],
                "challenge_evaluation": {}
            }
        }
    
    # 코멘트 생성 (LLM)
    llm = get_llm(mini=False)
    style_str = json.dumps(style_analysis, ensure_ascii=False, indent=2) if style_analysis else "(없음)"
    patterns_str = "\n".join([
        f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    try:
        res = (_COMMENT_PROMPT | llm).invoke({
            "style_analysis": style_str,
            "patterns": patterns_str,
        })
        comment = getattr(res, "content", "") or str(res)
        comment = comment.strip()
    except Exception as e:
        print(f"Comment generation error: {e}")
        comment = "대화 패턴을 분석했습니다."
    
    # 메트릭 추출
    current_metrics = _extract_metrics_from_style_analysis(style_analysis)
    pattern_metrics = _extract_pattern_count_metrics(patterns, key_moments)
    current_metrics.extend(pattern_metrics)
    
    # 챌린지 평가 가져오기 (challenge_eval_node에서 생성된 challenge_evaluations 사용)
    # summarize_node는 challenge_eval_node와 병렬 실행되므로, challenge_evaluations가 없을 수 있음
    challenge_evaluations = state.get("challenge_evaluations") or []
    
    # 단일 챌린지인 경우 challenge_evaluation으로도 반환 (하위 호환성)
    challenge_evaluation = challenge_evaluations[0] if challenge_evaluations else {}
    
    return {
        "summary": {
            "analysis_session": {
                "comment": comment
            },
            "current_metrics": current_metrics,
            "challenge_evaluation": challenge_evaluation,  # 하위 호환성 (단일)
            "challenge_evaluations": challenge_evaluations  # 여러 챌린지
        }
    }

