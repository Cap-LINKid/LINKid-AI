from __future__ import annotations

from typing import Dict, Any


def _calculate_ratios_from_labels(utterances_labeled: list) -> tuple[float, float]:
    """라벨 기반으로 긍정/부정 비율 계산"""
    if not utterances_labeled:
        return 0.5, 0.5
    
    parent_utterances = [
        utt for utt in utterances_labeled 
        if utt.get("speaker", "").lower() in ["parent", "mom", "mother", "dad", "father"]
    ]
    
    if not parent_utterances:
        return 0.5, 0.5
    
    total = len(parent_utterances)
    
    # 긍정적 라벨: RD (반영적 듣기), PR (칭찬)
    positive_count = sum(
        1 for utt in parent_utterances 
        if utt.get("label") in ["RD", "PR"]
    )
    
    # 부정적 라벨: NEG (부정적 발화), CMD (지시형 발화) - 높은 비율일 때 부정적
    negative_count = sum(
        1 for utt in parent_utterances 
        if utt.get("label") in ["NEG"]
    )
    
    # CMD는 비율이 높을 때만 부정적으로 간주 (30% 이상)
    cmd_count = sum(
        1 for utt in parent_utterances 
        if utt.get("label") == "CMD"
    )
    cmd_ratio = cmd_count / total if total > 0 else 0.0
    if cmd_ratio > 0.3:
        negative_count += cmd_count * 0.5  # CMD는 부분적으로만 부정적
    
    positive_ratio = positive_count / total if total > 0 else 0.0
    negative_ratio = negative_count / total if total > 0 else 0.0
    
    # 정규화 (합이 1.0이 되도록)
    total_ratio = positive_ratio + negative_ratio
    if total_ratio > 0:
        positive_ratio = positive_ratio / total_ratio
        negative_ratio = negative_ratio / total_ratio
    else:
        # 중립적인 경우
        positive_ratio = 0.5
        negative_ratio = 0.5
    
    return round(positive_ratio, 2), round(negative_ratio, 2)


def _determine_stage_name(
    positive_ratio: float, 
    negative_ratio: float, 
    utterances_labeled: list,
    patterns: list
) -> str:
    """비율과 패턴을 기반으로 상호작용 단계 이름 결정"""
    if not utterances_labeled:
        return "분석 불가"
    
    # 부모 발화 필터링
    parent_utterances = [
        utt for utt in utterances_labeled 
        if utt.get("speaker", "").lower() in ["parent", "mom", "mother", "dad", "father"]
    ]
    
    if not parent_utterances:
        return "분석 불가"
    
    total = len(parent_utterances)
    
    # 라벨별 비율 계산
    rd_count = sum(1 for utt in parent_utterances if utt.get("label") == "RD")
    pr_count = sum(1 for utt in parent_utterances if utt.get("label") == "PR")
    cmd_count = sum(1 for utt in parent_utterances if utt.get("label") == "CMD")
    neg_count = sum(1 for utt in parent_utterances if utt.get("label") == "NEG")
    q_count = sum(1 for utt in parent_utterances if utt.get("label") == "Q")
    
    rd_ratio = rd_count / total if total > 0 else 0.0
    pr_ratio = pr_count / total if total > 0 else 0.0
    cmd_ratio = cmd_count / total if total > 0 else 0.0
    neg_ratio = neg_count / total if total > 0 else 0.0
    
    # 패턴 정보 확인
    has_negative_pattern = any(
        "부정" in str(p.get("pattern_name", "")).lower() or 
        "negative" in str(p.get("pattern_name", "")).lower() or
        "개선" in str(p.get("description", "")).lower()
        for p in patterns
    )
    
    # 단계 결정 로직
    # 1. 공감적 협력: RD와 PR 비율이 높고, NEG가 낮은 경우
    if (rd_ratio + pr_ratio) >= 0.4 and neg_ratio < 0.2 and not has_negative_pattern:
        return "공감적 협력"
    
    # 2. 개선이 필요한 상호작용: NEG 비율이 높거나 부정적 패턴이 있는 경우
    if neg_ratio >= 0.3 or (has_negative_pattern and neg_ratio >= 0.15):
        return "개선이 필요한 상호작용"
    
    # 3. 지시적 상호작용: CMD 비율이 높고, RD/PR이 낮은 경우
    if cmd_ratio >= 0.3 and (rd_ratio + pr_ratio) < 0.2:
        return "지시적 상호작용"
    
    # 4. 균형잡힌 대화: 그 외의 경우
    return "균형잡힌 대화"


def summary_diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    summary_diagnosis: 전체 상호작용 진단 (단계, 긍정/부정 비율)
    utterances_labeled에서 직접 라벨 정보를 사용하여 Python으로 계산
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not utterances_labeled:
        return {
            "summary_diagnosis": {
                "stage_name": "분석 불가",
                "positive_ratio": 0.5,
                "negative_ratio": 0.5
            }
        }
    
    # utterances_labeled에서 직접 Python으로 비율 계산
    positive_ratio, negative_ratio = _calculate_ratios_from_labels(utterances_labeled)
    
    # 비율과 패턴을 기반으로 단계 이름 결정
    stage_name = _determine_stage_name(
        positive_ratio, 
        negative_ratio, 
        utterances_labeled, 
        patterns
    )
    
    return {
        "summary_diagnosis": {
            "stage_name": stage_name,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }
    }

