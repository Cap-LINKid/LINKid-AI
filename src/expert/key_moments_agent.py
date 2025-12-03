from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm
from src.utils.vector_store import search_expert_advice


# -------------------------------------------------------------------------
# 1. Pydantic 모델 정의 (최종 JSON 구조)
# -------------------------------------------------------------------------

class DialogueLine(BaseModel):
    speaker: str
    text: str
    


class ExpertReference(BaseModel):
    title: str
    source: str
    author: str
    excerpt: str
    relevance_score: float


class PositiveMoment(BaseModel):
    dialogue: List[DialogueLine]
    pattern_hint: str
    reason: str
    reference_descriptions: List[str]


class NeedsImprovementMoment(BaseModel):
    dialogue: List[DialogueLine]
    reason: str
    better_response: str
    reference_descriptions: List[str]
    pattern_hint: str
    expert_references: List[ExpertReference]


class PatternExample(BaseModel):
    pattern_name: str
    occurrences: int
    occurred_at: str
    dialogue: List[DialogueLine]
    problem_explanation: str
    suggested_response: str


class KeyMomentsResult(BaseModel):
    positive: List[PositiveMoment]
    needs_improvement: List[NeedsImprovementMoment]
    pattern_examples: List[PatternExample]


class KeyMomentsResponse(BaseModel):
    key_moments: KeyMomentsResult


# -------------------------------------------------------------------------
# 2. LLM 프롬프트 (최종 완성본)
# -------------------------------------------------------------------------

_GENERATE_ADVICE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 아동 심리 및 부모 교육 전문가입니다.
입력으로 부모-자녀 대화, 탐지된 패턴 정보, 전문가 조언을 바탕으로
Key Moments 분석을 생성해야 합니다.

==============================
📌 절대 지켜야 할 규칙
==============================

1) JSON 구조 절대 변경 금지
2) Positive: 반드시 전문가 excerpt 1개 포함
3) Needs Improvement: 전문가 excerpt 1~2개 포함
4) reference_descriptions: 최대 2개
5) Pattern Examples: 반드시 "1개만"
6) reason: 전문가 excerpt와 대화의 맥락과 상황을 파악하여 2~4 줄 정도로 길고 구체적으로 나올 수 있도록.
7) better_response: 부모가 실제 사용할 수 있는 대사 형태와 이런 대안이 나온 이유를 뽑힌 전문가 excerpt를 반영해서 구체적으로 작성하세요.
8) tone은 따뜻하고 전문적이지만, ~~합니다.와 같이 공손하게 말할 수 있도록한다.

==============================
📌 Positive Moment 규칙
==============================
- positive_context의 pattern과 dialogue만 사용
- 전문가 조언 excerpt 1개를 reason에 자연스럽게 섞어 쓰기
- reference_descriptions는 최대 2개

==============================
📌 Needs Improvement 규칙
==============================
- 가장 심각한 부정 패턴 하나만 사용
- reason: 상황 요약 → 문제점 → 아동 발달 영향 → 전문가 조언 인용(1~2개)
- better_response: 실제 사용할 수 있는 구체 대사

==============================
📌 Pattern Examples 규칙
==============================
- Needs Improvement 다음으로 심각한 1개의 패턴만 선택
- 이유와 조언은 전문가 excerpt와 대화의 맥락과 상황을 파악하여 구체적으로 작성할 수 있도록 한다.
- succinct한 problem_explanation & suggested_response 작성하고, 1~2줄 정도로 구체적으로 나올 수 있도록 작성한다.

==============================
📌 입력 데이터
==============================
[Positive Context]
{positive_context}

[Needs Improvement Context]
{improvement_context}

[Pattern Examples 후보]
{examples_context}

[Expert References]
{expert_references}

이 모든 정보를 바탕으로 JSON Schema에 맞는 key_moments를 생성하십시오.
"""
    ),
    (
        "human",
        "위 내용을 반영하여 key_moments JSON을 생성하세요."
    ),
])


# -------------------------------------------------------------------------
# 3. Helper 함수
# -------------------------------------------------------------------------

def _extract_dialogue(utterances: List[Dict], indices: List[int]) -> List[Dict]:
    dialogue = []
    for idx in sorted(indices):
        if 0 <= idx < len(utterances):
            utt = utterances[idx]
            speaker = "parent" if utt.get("speaker") in ["Parent", "Mom", "Dad", "부모", "A"] else "child"
            text = utt.get("original_ko") or utt.get("korean") or utt.get("text", "")
            dialogue.append({"speaker": speaker, "text": text})
    return dialogue


def _ref_desc_from_refs(refs: List[ExpertReference]) -> List[str]:
    desc = []
    for r in refs[:2]:
        desc.append(f"{r.author} - {r.title}")
    return desc[:2]

def _search_refs_for_pattern(pattern: Optional[Dict[str, Any]]) -> List[ExpertReference]:
    """하나의 패턴에 대해 전문가 DB(RAG) 검색"""
    if not pattern:
        return []

    pattern_name = pattern.get("pattern_name") or pattern.get("description") or ""
    if not pattern_name:
        return []

    try:
        raw = search_expert_advice(
            query=pattern_name,
            top_k=3,
            threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.15")),
        )
    except Exception as e:
        print(f"[VectorDB] 검색 오류 ({pattern_name}): {e}")
        return []

    refs: List[ExpertReference] = []
    for r in raw[:2]:  # 안전하게 2개까지만 가져오기
        content = r.get("content", "") or ""
        excerpt = content[:200]
        refs.append(
            ExpertReference(
                title=r.get("title", ""),
                source=r.get("source", ""),
                author=r.get("author", "전문가"),
                excerpt=excerpt,
                relevance_score=r.get("relevance_score", 0.0),
            )
        )
    return refs

# -------------------------------------------------------------------------
# 4. Main Key Moments Node
# -------------------------------------------------------------------------

def key_moments_node(state: Dict[str, Any]) -> Dict[str, Any]:
    utterances = state.get("utterances_ko") or state.get("utterances_labeled", [])
    patterns = state.get("patterns", [])

    if not patterns:
        return {"key_moments": None}

    # Severity 기준 정렬
    severity_order = {"high": 3, "medium": 2, "low": 1}
    neg_patterns = [p for p in patterns if p.get("pattern_type") == "negative"]
    pos_patterns = [p for p in patterns if p.get("pattern_type") == "positive"]

    neg_patterns.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 1), reverse=True)

    # 선택 대상
    target_positive = pos_patterns[0] if pos_patterns else None
    target_improvement = neg_patterns[0] if neg_patterns else None
    target_examples = neg_patterns[1:2]  # 딱 1개만

    # ---------------------------------------------------------
    # RAG: 전문가 조언 검색 (긍정 / 최악 / 두 번째 패턴 각각)
    # ---------------------------------------------------------
    pos_expert_refs: List[ExpertReference] = _search_refs_for_pattern(target_positive)
    neg_expert_refs: List[ExpertReference] = _search_refs_for_pattern(target_improvement)
    ex_expert_refs: List[ExpertReference] = _search_refs_for_pattern(target_examples[0]) if target_examples else []

    # LLM에 넘길 Expert References 구조화
    expert_refs_payload = {
        "positive": [r.dict() for r in pos_expert_refs],
        "needs_improvement": [r.dict() for r in neg_expert_refs],
        "pattern_examples": [r.dict() for r in ex_expert_refs],
    }
    expert_refs_json = json.dumps(expert_refs_payload, ensure_ascii=False)


    # ---------------------------------------------------------
    # LLM 인풋 컨텍스트 구성
    # ---------------------------------------------------------

    # Positive
    if target_positive:
        pos_ctx = json.dumps({
            "pattern_name": target_positive["pattern_name"],
            "description": target_positive["description"],
            "dialogue": _extract_dialogue(utterances, target_positive["utterance_indices"])
        }, ensure_ascii=False)
    else:
        pos_ctx = "없음"

    # Needs Improvement
    if target_improvement:
        imp_ctx = json.dumps({
            "pattern_name": target_improvement["pattern_name"],
            "description": target_improvement["description"],
            "dialogue": _extract_dialogue(utterances, target_improvement["utterance_indices"])
        }, ensure_ascii=False)
    else:
        imp_ctx = "없음"

    # Pattern Example 후보
    ex_ctx = json.dumps([
        {
            "pattern_name": ex["pattern_name"],
            "description": ex["description"],
            "dialogue": _extract_dialogue(utterances, ex["utterance_indices"])
        }
        for ex in target_examples
    ], ensure_ascii=False)

    # ---------------------------------------------------------
    # LLM 호출 (Structured Output)
    # ---------------------------------------------------------
    llm = get_structured_llm(KeyMomentsResponse)

    result = (_GENERATE_ADVICE_PROMPT | llm).invoke({
        "positive_context": pos_ctx,
        "improvement_context": imp_ctx,
        "examples_context": ex_ctx,
        "expert_references": expert_refs_json
    })

    final_data = result.key_moments

    # ---------------------------------------------------------
    # 후처리: 필드 정제/보정
    # ---------------------------------------------------------

    # Positive 보정
    if target_positive and final_data.positive:
        pm = final_data.positive[0]
        pm.dialogue = _extract_dialogue(utterances, target_positive["utterance_indices"])
        pm.pattern_hint = target_positive["pattern_name"]
        # Positive는 긍정 패턴에 대한 RAG 결과를 기반으로 reference_descriptions 구성
        pm.reference_descriptions = _ref_desc_from_refs(pos_expert_refs)

    # Needs Improvement 보정
    if target_improvement and final_data.needs_improvement:
        ni = final_data.needs_improvement[0]
        ni.dialogue = _extract_dialogue(utterances, target_improvement["utterance_indices"])
        ni.pattern_hint = target_improvement["pattern_name"]
        ni.expert_references = neg_expert_refs
        ni.reference_descriptions = _ref_desc_from_refs(neg_expert_refs)

    # Pattern Examples 보정 (두 번째로 심각한 패턴 1개)
    for i, ex_target in enumerate(target_examples):
        if i < len(final_data.pattern_examples):
            pe = final_data.pattern_examples[i]
            pe.pattern_name = ex_target["pattern_name"]
            pe.dialogue = _extract_dialogue(utterances, ex_target["utterance_indices"])
            pe.occurrences = len(ex_target["utterance_indices"])
            idx = ex_target["utterance_indices"][0]
            pe.occurred_at = f"{idx // 6}분 {idx * 10 % 60}초"

    return {"key_moments": final_data.dict()}