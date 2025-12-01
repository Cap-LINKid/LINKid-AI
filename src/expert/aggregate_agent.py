from __future__ import annotations

from typing import Dict, Any


def _calculate_pi_ndi_scores(utterances_labeled: list) -> Dict[str, int]:
    """
    DPICS 라벨링 결과를 기반으로 PI score와 NDI score 계산
    부모 발화와 아이 발화 모두 포함하여 계산
    
    PI (Positive Interaction) score: 긍정적 상호작용 비율
    - PR (Praise) + RD (Reflection) 비율을 0-100 점수로 변환
    
    NDI (Negative Directiveness Index) score: 부정적 지시성 지수
    - NEG (Negative) + CMD (Command) 비율을 0-100 점수로 변환
    
    Args:
        utterances_labeled: 라벨링된 발화 리스트 (부모 + 아이 발화 모두 포함)
        
    Returns:
        {"pi_score": int, "ndi_score": int}
    """
    if not utterances_labeled:
        return {"pi_score": 50, "ndi_score": 50}
    
    # 모든 발화 사용 (부모 + 아이)
    total = len(utterances_labeled)
    
    # PI score 계산: PR (Praise) + RD (Reflection) 비율
    positive_count = sum(
        1 for utt in utterances_labeled 
        if utt.get("label") in ["PR", "RD"]
    )
    pi_ratio = positive_count / total if total > 0 else 0.0
    pi_score = int(round(pi_ratio * 100))
    
    # NDI score 계산: NEG (Negative) + CMD (Command) 비율
    negative_count = sum(
        1 for utt in utterances_labeled 
        if utt.get("label") in ["NEG", "CMD"]
    )
    ndi_ratio = negative_count / total if total > 0 else 0.0
    ndi_score = int(round(ndi_ratio * 100))
    
    return {
        "pi_score": min(100, max(0, pi_score)),  # 0-100 범위로 제한
        "ndi_score": min(100, max(0, ndi_score))  # 0-100 범위로 제한
    }


def aggregate_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑩ aggregate_result: 최종 JSON 집계
    모든 분석 결과를 하나의 JSON으로 통합
    """
    summary = state.get("summary", {})
    style_analysis = state.get("style_analysis", {})
    challenge_eval = state.get("challenge_eval", {})
    challenge_evals = state.get("challenge_evals", [])
    challenge_evaluations = state.get("challenge_evaluations", [])
    challenge_spec = state.get("challenge_spec", {})
    challenge_specs = state.get("challenge_specs", [])
    utterances_labeled = state.get("utterances_labeled", [])
    utterances_ko = state.get("utterances_ko", [])
    key_moments = state.get("key_moments", {})
    patterns = state.get("patterns", [])
    summary_diagnosis = state.get("summary_diagnosis", {})
    coaching_plan = state.get("coaching_plan", {})
    
    # summary가 새로운 형식인 경우 업데이트 (growth_report 형식)
    if isinstance(summary, dict) and "analysis_session" in summary:
        # style_analysis에서 current_metrics 업데이트 (병렬 실행으로 인해 summarize_node에서 없을 수 있음)
        from src.expert.summarize_agent import _extract_metrics_from_style_analysis, _extract_pattern_count_metrics
        
        # style_analysis가 dict인지 확인 ({"style_analysis": {...}} 형식일 수 있음)
        actual_style_analysis = style_analysis
        if isinstance(style_analysis, dict) and "style_analysis" in style_analysis:
            actual_style_analysis = style_analysis["style_analysis"]
        
        # 메트릭 추출
        style_metrics = _extract_metrics_from_style_analysis(actual_style_analysis)
        pattern_metrics = _extract_pattern_count_metrics(patterns, key_moments)
        
        # current_metrics 업데이트 (기존 것과 합치기)
        existing_metrics = summary.get("current_metrics", [])
        # 기존 메트릭과 새 메트릭을 합치되, 중복 제거 (key 기준)
        metric_keys = {m.get("key") for m in existing_metrics}
        new_metrics = [m for m in style_metrics + pattern_metrics if m.get("key") not in metric_keys]
        summary["current_metrics"] = existing_metrics + new_metrics
        
        # challenge_evaluation 업데이트 (새로운 방식: challenge_eval_node에서 직접 반환)
        if not summary.get("challenge_evaluations") and challenge_evaluations:
            summary["challenge_evaluations"] = challenge_evaluations
            # # 하위 호환성을 위해 첫 번째 챌린지도 challenge_evaluation으로 설정
            # if challenge_evaluations:
            #     summary["challenge_evaluation"] = challenge_evaluations[0]
    
    # coaching_plan에서 coaching_plan 추출 ({"coaching_plan": {...}} 형식일 수 있음)
    actual_coaching_plan = coaching_plan
    if isinstance(coaching_plan, dict) and "coaching_plan" in coaching_plan:
        actual_coaching_plan = coaching_plan["coaching_plan"]
    
    # key_moments에서 key_moments 추출 ({"key_moments": {...}} 형식일 수 있음)
    actual_key_moments = key_moments
    if isinstance(key_moments, dict) and "key_moments" in key_moments:
        actual_key_moments = key_moments["key_moments"]
    
    # summary_diagnosis에서 summary_diagnosis 추출 ({"summary_diagnosis": {...}} 형식일 수 있음)
    actual_summary_diagnosis = summary_diagnosis
    if isinstance(summary_diagnosis, dict) and "summary_diagnosis" in summary_diagnosis:
        actual_summary_diagnosis = summary_diagnosis["summary_diagnosis"]
    
    # PI/NDI 점수 계산
    scores = _calculate_pi_ndi_scores(utterances_labeled)
    
    # 요청된 구조로 result 구성
    result = {
        "summary_diagnosis": actual_summary_diagnosis,
        "key_moment_capture": {
            "key_moments": actual_key_moments
        },
        "style_analysis": style_analysis,
        "coaching_and_plan": {
            "coaching_plan": actual_coaching_plan
        },
        "growth_report": summary,
        "scores": scores,
    }
    
    # LangGraph UI에서 확인할 때 status API 응답과 동일한 형태로 보이도록
    # 중간 노드 결과들을 모두 제거하고 result만 포함
    # (status API는 nodes 필드만 제거하지만, LangGraph에서는 모든 중간 결과 제거)
    return {"result": result}

