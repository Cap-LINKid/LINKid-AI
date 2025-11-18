from __future__ import annotations

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm


class SummaryDiagnosis(BaseModel):
    """요약 진단"""
    stage_name: str = Field(description="상호작용 단계 이름 (예: '공감적 협력', '지시적 상호작용' 등)")
    positive_ratio: float = Field(description="긍정적 상호작용 비율 (0.0 ~ 1.0)")
    negative_ratio: float = Field(description="부정적 상호작용 비율 (0.0 ~ 1.0)")


_SUMMARY_DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 상호작용을 진단하는 전문가입니다. "
            "라벨 분포와 패턴을 분석하여 상호작용의 단계와 긍정/부정 비율을 판단하세요.\n\n"
            "단계 이름 예시:\n"
            "- '공감적 협력': 반영적 듣기와 칭찬이 많고, 부정적 발화가 적은 경우\n"
            "- '지시적 상호작용': 지시형 발화가 많고, 질문과 칭찬이 적은 경우\n"
            "- '균형잡힌 대화': 다양한 발화 유형이 균형있게 분포된 경우\n"
            "- '개선이 필요한 상호작용': 부정적 발화가 많고, 긍정적 발화가 적은 경우\n\n"
            "긍정적 비율 계산 기준:\n"
            "- 반영적 듣기(RD), 칭찬(PR) 비율이 높을수록 긍정적\n"
            "- 부정적 발화(NEG), 지시형 발화(CMD) 비율이 낮을수록 긍정적\n\n"
            "부정적 비율 계산 기준:\n"
            "- 부정적 발화(NEG), 지시형 발화(CMD) 비율이 높을수록 부정적\n"
            "- 반영적 듣기(RD), 칭찬(PR) 비율이 낮을수록 부정적\n\n"
            "positive_ratio와 negative_ratio의 합은 1.0이 되도록 계산하세요."
        ),
    ),
    (
        "human",
        (
            "부모 발화 라벨 비율:\n{parent_ratios}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "위 정보를 바탕으로 상호작용 단계와 긍정/부정 비율을 진단해주세요."
        ),
    ),
])


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


def summary_diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    summary_diagnosis: 전체 상호작용 진단 (단계, 긍정/부정 비율)
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    style_analysis = state.get("style_analysis") or {}
    
    if not utterances_labeled:
        return {
            "summary_diagnosis": {
                "stage_name": "분석 불가",
                "positive_ratio": 0.5,
                "negative_ratio": 0.5
            }
        }
    
    # style_analysis에서 부모 발화 비율 추출
    parent_ratios_str = ""
    if isinstance(style_analysis, dict) and "style_analysis" in style_analysis:
        actual_style = style_analysis["style_analysis"]
        if "interaction_style" in actual_style:
            parent_analysis = actual_style["interaction_style"].get("parent_analysis", {})
            categories = parent_analysis.get("categories", [])
            parent_ratios_str = "\n".join([
                f"- {cat.get('name', '')} ({cat.get('label', '')}): {cat.get('ratio', 0.0) * 100:.1f}%"
                for cat in categories
            ])
    
    # 패턴 정보 포맷팅
    patterns_str = "\n".join([
        f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    # Structured LLM 사용
    structured_llm = get_structured_llm(SummaryDiagnosis, mini=False)
    
    try:
        res = (_SUMMARY_DIAGNOSIS_PROMPT | structured_llm).invoke({
            "parent_ratios": parent_ratios_str or "분석 데이터 없음",
            "patterns": patterns_str,
        })
        
        if isinstance(res, SummaryDiagnosis):
            # LLM 결과 사용
            return {
                "summary_diagnosis": {
                    "stage_name": res.stage_name,
                    "positive_ratio": res.positive_ratio,
                    "negative_ratio": res.negative_ratio
                }
            }
    except Exception as e:
        print(f"Summary diagnosis LLM error: {e}")
    
    # 폴백: 라벨 기반 계산
    positive_ratio, negative_ratio = _calculate_ratios_from_labels(utterances_labeled)
    
    # 단계 이름 결정 (비율 기반)
    if positive_ratio >= 0.6:
        stage_name = "공감적 협력"
    elif positive_ratio >= 0.4:
        stage_name = "균형잡힌 대화"
    elif negative_ratio >= 0.5:
        stage_name = "개선이 필요한 상호작용"
    else:
        stage_name = "지시적 상호작용"
    
    return {
        "summary_diagnosis": {
            "stage_name": stage_name,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }
    }

