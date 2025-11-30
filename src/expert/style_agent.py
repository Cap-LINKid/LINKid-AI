from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm

# label_mapping.json 기반 한국어 매핑
_LABEL_MAPPING_PATH = Path(__file__).parent.parent.parent / "models" / "dpics-electra" / "label_mapping.json"

# DPICS 코드 -> 한국어 이름 매핑
_DPICS_TO_KOREAN = {
    "RD": "반영적 진술",  # Reflective Statement
    "PR": "칭찬",  # Labeled Praise, Unlabeled Praise, Prosocial Talk
    "CMD": "지시",  # Command
    "Q": "질문",  # Question
    "NT": "중립적 발화",  # Neutral Talk
    "BD": "행동 설명",  # Behavior Description
    "NEG": "부정적 발화",  # Negative Talk
    "OTH": "기타",
    "IGN": "무시",
}

# label_mapping.json에서 한국어 매핑 로드
def _load_label_mapping() -> Dict[str, str]:
    """label_mapping.json을 로드하여 영어 라벨 -> 한국어 매핑 생성"""
    if _LABEL_MAPPING_PATH.exists():
        try:
            with open(_LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            
            # 영어 라벨 -> 한국어 매핑
            english_to_korean = {
                "Reflective Statement": "반영적 진술",
                "Labeled Praise": "칭찬",
                "Unlabeled Praise": "칭찬",
                "Prosocial Talk": "칭찬",
                "Command": "지시",
                "Question": "질문",
                "Neutral Talk": "중립적 발화",
                "Behavior Description": "행동 설명",
                "Negative Talk": "부정적 발화",
            }
            
            return english_to_korean
        except Exception as e:
            print(f"label_mapping.json 로드 실패: {e}")
    
    return {}

# 영어 라벨 -> 한국어 매핑
_ENGLISH_LABEL_TO_KOREAN = _load_label_mapping()

# DPICS 코드 -> 영어 라벨 -> 한국어 매핑 (dpics_electra.py의 매핑 참고)
_DPICS_CODE_TO_ENGLISH = {
    "RD": "Reflective Statement",
    "PR": "Labeled Praise",  # 또는 Unlabeled Praise, Prosocial Talk
    "CMD": "Command",
    "Q": "Question",
    "NT": "Neutral Talk",
    "BD": "Behavior Description",
    "NEG": "Negative Talk",
    "IGN": "Ignore",  # IGN은 영어 매핑이 없으므로 직접 한국어 사용
    "OTH": "Other",  # OTH도 영어 매핑이 없으므로 직접 한국어 사용
}

def _get_korean_label_name(dpics_code: str) -> str:
    """DPICS 코드를 한국어 이름으로 변환"""
    english_label = _DPICS_CODE_TO_ENGLISH.get(dpics_code)
    if english_label and english_label in _ENGLISH_LABEL_TO_KOREAN:
        return _ENGLISH_LABEL_TO_KOREAN[english_label]
    return _DPICS_TO_KOREAN.get(dpics_code, "기타")


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyzing parent-child communication patterns. "
            "Analyze the label distribution ratios for parent and child utterances and provide insights. "
            "Return ONLY a summary text in Korean (2-3 sentences) that describes the communication patterns, "
            "strengths, and areas for improvement based on the ratios. "
            "Be specific about percentages and patterns. No extra text, just the summary."
        ),
    ),
    (
        "human",
        (
            "부모 발화 라벨 비율:\n{parent_ratios}\n\n"
            "아이 발화 라벨 비율:\n{child_ratios}\n\n"
            "위 비율을 분석하여 대화 패턴에 대한 요약을 작성해주세요."
        ),
    ),
])


def analyze_style_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑦ analyze_style: 스타일/비율 분석 (라벨 기반 통계)
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    # 모든 DPICS 라벨 목록
    all_labels = ["RD", "PR", "CMD", "Q", "NT", "BD", "NEG", "IGN", "OTH"]
    
    if not utterances_labeled:
        # 모든 라벨에 대한 빈 비율 생성
        empty_label_ratios = {label: 0.0 for label in all_labels}
        empty_categories = [
            {"name": _get_korean_label_name(label), "ratio": 0.0, "label": label}
            for label in all_labels
        ]
        # ratio가 높은 순으로 정렬 (모두 0.0이지만 일관성을 위해)
        empty_categories.sort(key=lambda x: x["ratio"], reverse=True)
        
        return {
            "style_analysis": {
                "interaction_style": {
                    "parent_analysis": {
                        "categories": empty_categories,
                        "label_distribution": empty_label_ratios
                    },
                    "child_analysis": {
                        "categories": empty_categories,
                        "label_distribution": empty_label_ratios
                    }
                },
                "summary": "분석할 데이터가 없습니다."
            }
        }
    
    # 패턴/라벨 기반 통계 계산
    parent_utterances = [utt for utt in utterances_labeled if utt.get("speaker") in ["Parent", "MOM", "Dad", "Father", "Mother"]]
    child_utterances = [utt for utt in utterances_labeled if utt.get("speaker") in ["Child", "CHI", "Kid", "Son", "Daughter"]]
    
    total_parent = len(parent_utterances)
    total_child = len(child_utterances)
    
    # 부모 발화 라벨 통계
    parent_label_counts = {}
    for utt in parent_utterances:
        label = utt.get("label", "OTH")
        parent_label_counts[label] = parent_label_counts.get(label, 0) + 1
    
    # 아이 발화 라벨 통계
    child_label_counts = {}
    for utt in child_utterances:
        label = utt.get("label", "OTH")
        child_label_counts[label] = child_label_counts.get(label, 0) + 1
    
    # 모든 라벨에 대한 비율 계산 (부모 발화)
    parent_label_ratios = {}
    for label in all_labels:
        count = parent_label_counts.get(label, 0)
        ratio = (count / total_parent) if total_parent > 0 else 0.0
        parent_label_ratios[label] = round(ratio, 2)
    
    # 모든 라벨에 대한 비율 계산 (아이 발화)
    child_label_ratios = {}
    for label in all_labels:
        count = child_label_counts.get(label, 0)
        ratio = (count / total_child) if total_child > 0 else 0.0
        child_label_ratios[label] = round(ratio, 2)
    
    # LLM 기반 요약 생성
    parent_ratios_str = "\n".join([
        f"- {_get_korean_label_name(label)} ({label}): {parent_label_ratios.get(label, 0.0) * 100:.1f}%"
        for label in all_labels
    ])
    child_ratios_str = "\n".join([
        f"- {_get_korean_label_name(label)} ({label}): {child_label_ratios.get(label, 0.0) * 100:.1f}%"
        for label in all_labels
    ])
    
    try:
        llm = get_llm(mini=False)
        res = (_SUMMARY_PROMPT | llm).invoke({
            "parent_ratios": parent_ratios_str,
            "child_ratios": child_ratios_str,
        })
        summary = getattr(res, "content", "") or str(res)
        summary = summary.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        # 폴백: 간단한 요약
        summary = "대화 패턴을 분석했습니다."
    
    # 모든 라벨에 대한 카테고리 생성 (부모 발화) - 한국어 매핑 적용
    parent_categories = [
        {
            "name": _get_korean_label_name(label),
            "ratio": parent_label_ratios.get(label, 0.0),
            "label": label
        }
        for label in all_labels
    ]
    # ratio가 높은 순으로 정렬
    parent_categories.sort(key=lambda x: x["ratio"], reverse=True)
    
    # 모든 라벨에 대한 카테고리 생성 (아이 발화) - 한국어 매핑 적용
    child_categories = [
        {
            "name": _get_korean_label_name(label),
            "ratio": child_label_ratios.get(label, 0.0),
            "label": label
        }
        for label in all_labels
    ]
    # ratio가 높은 순으로 정렬
    child_categories.sort(key=lambda x: x["ratio"], reverse=True)
    
    return {
        "style_analysis": {
            "interaction_style": {
                "parent_analysis": {
                    "categories": parent_categories
                },
                "child_analysis": {
                    "categories": child_categories
                }
            },
            "summary": summary
        }
    }

