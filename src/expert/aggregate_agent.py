from __future__ import annotations

from typing import Dict, Any


def aggregate_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑩ aggregate_result: 최종 JSON 집계
    모든 분석 결과를 하나의 JSON으로 통합
    """
    summary = state.get("summary", {})
    style_analysis = state.get("style_analysis", {})
    challenge_eval = state.get("challenge_eval", {})
    challenge_spec = state.get("challenge_spec", {})
    utterances_labeled = state.get("utterances_labeled", [])
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
        
        # challenge_evaluation 업데이트
        if challenge_eval and challenge_spec:
            from src.expert.summarize_agent import _format_challenge_evaluation
            challenge_evaluation = _format_challenge_evaluation(
                challenge_eval, challenge_spec, utterances_labeled, key_moments
            )
            summary["challenge_evaluation"] = challenge_evaluation
    
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
    }
    
    # LangGraph UI와 API 모두에서 동일한 형식으로 반환
    # result 필드에 최종 결과를 넣고, state의 다른 필드들은 유지
    return {"result": result}

