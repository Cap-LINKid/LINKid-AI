from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm
from src.utils.pattern_manager import is_negative_pattern, normalize_pattern_name


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
    # "긍정적 기회 놓치기" 또는 "긍정기회놓치기" 패턴 확인 (정규화된 이름으로 비교)
    for pattern_name, count in pattern_counts.items():
        normalized_pattern = normalize_pattern_name(pattern_name)
        # 정규화된 패턴명에 "긍정기회놓치기" 또는 "긍정적기회놓치기"가 포함되어 있는지 확인
        if "긍정기회놓치기" in normalized_pattern or "긍정적기회놓치기" in normalized_pattern:
            metrics.append({
                "key": "missed_positive_opportunity_count",
                "label": "긍정적 기회 놓치기 패턴",
                "value": count,
                "value_type": "count"
            })
    
    return metrics


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
                "current_metrics": []
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
    
    # challenge_evaluations는 aggregate_result_node에서 처리됨
    # (summarize_node와 challenge_eval_node가 병렬 실행되므로 여기서는 포함하지 않음)
    
    return {
        "summary": {
            "analysis_session": {
                "comment": comment
            },
            "current_metrics": current_metrics
        }
    }

