from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm, get_llm
from src.utils.vector_store import search_expert_advice
from src.utils.pattern_manager import extract_pattern_name as extract_pattern_name_from_manager, get_negative_pattern_names_normalized


class DialogueUtterance(BaseModel):
    """대화 발화"""
    speaker: str = Field(description="발화자: 'parent' (부모) 또는 'child' (아이)")
    text: str = Field(description="발화 내용 (한국어 원문)")


class PositiveMoment(BaseModel):
    """긍정적 순간"""
    dialogue: List[DialogueUtterance] = Field(description="대화 발화 리스트 (한국어 원문 포함)")
    reason: str = Field(description="긍정적인 이유 설명 (한국어)")
    pattern_hint: str = Field(description="관련 패턴 힌트 (한국어)")


class NeedsImprovementMoment(BaseModel):
    """개선이 필요한 순간"""
    dialogue: List[DialogueUtterance] = Field(description="대화 발화 리스트 (한국어 원문 포함)")
    reason: str = Field(description="개선이 필요한 이유 설명 (한국어)")
    better_response: str = Field(description="더 나은 응답 예시 (한국어)")
    pattern_hint: str = Field(description="관련 패턴 힌트 (한국어)")


class PatternExample(BaseModel):
    """패턴 예시"""
    pattern_name: str = Field(description="패턴 이름 (한국어)")
    occurrences: int = Field(description="발생 횟수")
    dialogue: List[DialogueUtterance] = Field(description="대화 발화 리스트 (한국어 원문 포함)")
    problem_explanation: str = Field(description="문제 설명 (한국어)")
    suggested_response: str = Field(description="제안된 응답 (한국어)")


class KeyMomentsContent(BaseModel):
    """핵심 순간 내용"""
    positive: List[PositiveMoment] = Field(description="긍정적 순간 리스트", default_factory=list)
    needs_improvement: List[NeedsImprovementMoment] = Field(description="개선이 필요한 순간 리스트", default_factory=list)
    pattern_examples: List[PatternExample] = Field(description="패턴 예시 리스트", default_factory=list)


class KeyMomentsResponse(BaseModel):
    """핵심 순간 결과"""
    key_moments: KeyMomentsContent = Field(description="핵심 순간 객체")


_KEY_MOMENTS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 상호작용에서 핵심 순간을 식별하는 전문가입니다. "
            "핵심 순간을 추출하여 세 가지 카테고리로 분류하세요:\n"
            "1. 'positive': 부모가 잘 대응한 순간들 (감정 코칭, 공감, 인정, 질문형 발화, 선택권 제공 등)\n"
            "2. 'needs_improvement': 부모의 응답을 명확히 개선할 수 있는 부정적 상호작용 순간들\n"
            "3. 'pattern_examples': 감지된 패턴의 구체적인 대화 발췌 예시들\n\n"
            "분류의 핵심 원칙 (반드시 준수):\n"
            "- 각 발화/순간에는 사전에 감지된 패턴 정보(예: 긍정적 패턴, 부정적 패턴)가 함께 제공된다고 가정합니다.\n"
            "- 'positive' 카테고리는 다음 두 조건을 모두 만족하는 순간만 포함합니다:\n"
            "  (1) 감지된 패턴이 '긍정적인 패턴'으로 라벨링된 발화일 것\n"
            "  (2) 당신이 의미적으로 판단했을 때도 아이와의 관계에 긍정적인 영향을 주는 발화일 것\n"
            "- 'needs_improvement' 카테고리는 다음 두 조건을 모두 만족하는 순간만 포함합니다:\n"
            "  (1) 감지된 패턴이 '부정적인 패턴'으로 라벨링된 발화일 것\n"
            "  (2) 당신이 의미적으로 판단했을 때도 아이에게 부정적인 영향을 줄 가능성이 높은 발화일 것\n"
            "- 위 두 조건 중 하나라도 만족하지 않는 순간은 'positive'나 'needs_improvement'에 억지로 넣지 말고 제외합니다.\n\n"
            "추가적인 의미 판단 기준 (semantic 판단 시 활용):\n"
            "- 질문형 발화(\"~할까?\", \"~어떻게 생각해?\", \"~하고 싶어?\", \"~맡아도 될까?\")는 일반적으로 아이의 의견을 존중하는 경향이 있으므로, "
            "패턴이 긍정적일 때 'positive' 후보로 간주합니다.\n"
            "- 선택권을 제공하는 발화(\"~할래?\", \"어떤 게 좋을까?\")는 아이의 자율성을 존중하는 경향이 있으므로, "
            "패턴이 긍정적일 때 'positive' 후보로 간주합니다.\n"
            "- 자녀의 의견을 물어보는 발화(예: \"너는 어떻게 생각해?\", \"어떤 게 좋아?\")는 "
            "대체로 긍정적 상호작용에 해당하므로, 패턴이 긍정적일 때 'positive' 후보로 간주합니다.\n"
            "- 반대로, 직접적인 명령형(\"해라\", \"해야 해\"), 비판적 발화, 아이의 감정을 무시하거나 깎아내리는 발화 등은 "
            "패턴이 부정적일 경우 'needs_improvement' 후보로 간주합니다.\n"
            "- 'needs_improvement'에는 명확히 문제가 있는 경우만 포함합니다. "
            "애매하거나 판단이 어려운 경우에는 해당 순간을 제외하거나, 긍정적 부분이 더 크다고 판단되면 'positive'로만 포함합니다.\n\n"
            "각 순간에 대해 발화에서 실제 대화(발화자와 한국어 원문 텍스트)를 포함하세요. "
            "대화는 핵심 순간을 구성하는 연속된 발화들의 리스트여야 합니다.\n\n"
            "'needs_improvement' 순간의 경우, 반드시 다음을 모두 포함해야 합니다:\n"
            "- 'reason': 왜 이 순간이 문제인지, 어떤 말/행동이 반복되는지, 아이가 어떻게 느꼈을지에 대한 구체적인 설명. "
            "상황을 명확히 묘사하세요 (예: \"아이가 떼를 부릴 때\", \"부모가 비판적으로 반응할 때\" 등).\n"
            "- 'better_response': 위 대화 내용을 바탕으로, 부모가 실제로 어떤 말과 태도로 바꾸어 말해야 하는지에 대한 구체적인 예시. "
            "가능한 한 **구체적인 예시 문장**과 **상황 설명**을 포함하여 작성하세요.\n\n"
            "모든 서술(설명, reason, better_response, problem_explanation 등)은 "
            "반말이 아닌 **존댓말(예: \"~합니다\", \"~합니다.\")** 체로 공손하게 작성하세요.\n\n"
            "'pattern_examples'의 경우, 감지된 패턴 이름, 발생 횟수, 문제/강점 설명, 제안된 응답 예시(있다면)를 포함하세요. "
            "모든 설명과 응답은 한국어로 작성하세요."
        ),
    ),
    (
        "human",
        (
            "라벨링된 발화:\n{utterances_labeled}\n\n"
            "감지된 패턴:\n{patterns}\n\n"
            "상호작용에서 핵심 순간을 추출하고 분류하세요. "
            "각 순간에 대해 발화자와 한국어 원문 텍스트가 포함된 실제 대화 발췌를 포함하세요."
        ),
    ),
])


class ImprovedNeedsImprovement(BaseModel):
    """개선된 needs_improvement"""
    reason: str = Field(description="개선이 필요한 이유 설명 (전문가 조언 반영, 한국어)")
    better_response: str = Field(description="더 나은 응답 예시 (전문가 조언 반영, 한국어)")


class ImprovedPositiveMoment(BaseModel):
    """개선된 positive moment"""
    reason: str = Field(description="긍정적인 이유 설명 (전문가 조언 반영, 한국어)")


_IMPROVE_NEEDS_IMPROVEMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 상호작용 전문가입니다. "
            "주어진 핵심 순간(key_moment)과 전문가 조언(expert_advice)을 바탕으로, "
            "'reason'과 'better_response'를 개선하여 작성하세요.\n\n"
            "요구사항:\n"
            "1. 'reason'에는 반드시 전문가 조언의 핵심 내용을 직접 인용하거나 요약하여 포함해야 합니다.\n"
            "2. 전문가 조언의 구체적인 문장, 원칙, 설명을 그대로 인용하거나 요약하여 'reason'에 반영하세요.\n"
            "3. 'better_response'는 전문가 조언에서 제시한 방법을 실제 대화 상황에 적용한 구체적인 예시로 작성하세요.\n"
            "4. 모든 서술은 반말이 아닌 **존댓말(예: \"~합니다\", \"~합니다.\")** 체로 공손하게 작성하세요.\n"
            "5. 한국어로 작성하세요."
        ),
    ),
    (
        "human",
        (
            "핵심 순간 (key_moment):\n"
            "대화:\n{dialogue}\n\n"
            "패턴 힌트: {pattern_hint}\n\n"
            "초기 분석:\n"
            "reason: {initial_reason}\n"
            "better_response: {initial_better_response}\n\n"
            "전문가 조언 (expert_advice):\n{expert_advice}\n\n"
            "위 전문가 조언을 반영하여 'reason'과 'better_response'를 개선하여 작성하세요. "
            "'reason'에는 반드시 전문가 조언의 핵심 내용을 인용하거나 요약하여 포함하세요."
        ),
    ),
])


_IMPROVE_POSITIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 상호작용에서 긍정적인 순간을 설명하는 전문가입니다. "
            "주어진 긍정적 핵심 순간(key_moment)과 전문가 조언(expert_advice)을 바탕으로, "
            "'reason'(왜 이 순간이 좋은지)을 더 풍부하고 구체적으로 개선해서 작성하세요.\n\n"
            "요구사항:\n"
            "1. 'reason'에는 반드시 전문가 조언의 핵심 내용을 직접 인용하거나 요약하여 포함해야 합니다.\n"
            "2. 전문가 조언의 구체적인 문장, 원칙, 설명을 그대로 인용하거나 요약하여 'reason'에 반영하세요.\n"
            "3. 이 순간이 아이의 정서, 자존감, 부모-자녀 관계에 어떤 긍정적인 영향을 주는지 분명하게 설명하세요.\n"
            "4. 모든 서술은 반말이 아닌 **존댓말(예: \"~합니다\", \"~합니다.\")** 체로 공손하게 작성하세요.\n"
            "5. 한국어로 작성하세요."
        ),
    ),
    (
        "human",
        (
            "긍정적 핵심 순간 (positive key_moment):\n"
            "대화:\n{dialogue}\n\n"
            "패턴 힌트: {pattern_hint}\n\n"
            "초기 reason:\n"
            "{initial_reason}\n\n"
            "전문가 조언 (expert_advice):\n{expert_advice}\n\n"
            "위 정보를 바탕으로 이 순간이 왜 중요한 긍정적 순간인지, "
            "전문가 조언의 내용을 반영하여 더 풍부하고 구체적인 'reason'을 작성해주세요."
        ),
    ),
])


def _build_needs_improvement_query(moment: Dict[str, Any]) -> str:
    """
    needs_improvement 순간에서 검색 쿼리 생성
    """
    pattern_hint = moment.get("pattern_hint", "")
    reason = moment.get("reason", "")
    
    # 패턴 힌트에서 패턴명만 추출 (콜론(:) 이전 부분만 사용)
    if pattern_hint:
        # "명령과제시: 명령만 내림" -> "명령과제시"
        pattern_name = pattern_hint.split(":")[0].strip() if ":" in pattern_hint else pattern_hint.strip()
        query = f"{pattern_name}"
    else:
        # reason에서 키워드 추출
        query = f"{reason}"
    
    return query


def _extract_pattern_name(pattern_hint: str) -> Optional[str]:
    """
    pattern_hint에서 실제 패턴명만 추출
    예: "명령과제시: 명령만 내림" -> "과도한 명령/지시" (정규화된 패턴명)
    """
    return extract_pattern_name_from_manager(pattern_hint)


def _map_dialogue_to_ko(
    dialogue: List[DialogueUtterance], 
    utterances_labeled: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """LLM이 반환한 dialogue를 한국어 원문으로 매핑"""
    mapped = []
    for utt in dialogue:
        matched_text = utt.text
        for orig_utt in utterances_labeled:
            orig_speaker_raw = (orig_utt.get("speaker", "") or "").lower()
            if orig_speaker_raw in ["mom", "mother", "parent", "엄마", "아빠"]:
                orig_speaker = "parent"
            elif orig_speaker_raw in ["chi", "child", "kid", "아이"]:
                orig_speaker = "child"
            else:
                orig_speaker = orig_speaker_raw

            if utt.speaker.lower() != orig_speaker:
                continue

            text_en = orig_utt.get("english", "") or ""
            text_raw = orig_utt.get("text", "") or ""

            if (utt.text in text_en or utt.text in text_raw or
                    text_en in utt.text or text_raw in utt.text):
                matched_text = orig_utt.get(
                    "original_ko",
                    orig_utt.get("korean", utt.text)
                )
                break

        mapped.append({"speaker": utt.speaker, "text": matched_text})
    return mapped


def _create_search_query_from_moment(
    moment: NeedsImprovementMoment,
    patterns: List[Dict[str, Any]]
) -> str:
    """
    needs_improvement moment에서 VectorDB 검색 쿼리 생성
    패턴, 발화 내용, reason(상황 요약)을 키워드로 사용
    """
    query_parts = []
    
    # 1. 패턴명 추가
    if moment.pattern_hint:
        extracted_pattern = _extract_pattern_name(moment.pattern_hint)
        if extracted_pattern:
            query_parts.append(extracted_pattern)
        else:
            # 패턴명 추출 실패 시 pattern_hint에서 콜론 이전 부분만 사용
            pattern_name = moment.pattern_hint.split(":")[0].strip() if ":" in moment.pattern_hint else moment.pattern_hint.strip()
            query_parts.append(pattern_name)
    
    # 2. reason에서 키워드 추출 (상황 요약)
    if moment.reason:
        # reason의 핵심 키워드만 추출 (너무 길면 앞부분만)
        reason_keywords = moment.reason[:100]  # 최대 100자
        query_parts.append(reason_keywords)
    
    # 3. 발화 내용 요약 (문제 상황 묘사)
    if moment.dialogue:
        dialogue_texts = [utt.text for utt in moment.dialogue]
        dialogue_summary = " ".join(dialogue_texts)[:100]  # 최대 100자
        query_parts.append(dialogue_summary)
    
    # 쿼리 조합
    query = " ".join(query_parts)
    return query if query.strip() else "부모 자녀 상호작용 개선"


def key_moments_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑥ key_moments: 핵심 순간 (LLM)
    
    3단계 파이프라인:
    1) LLM으로 key_moments(positive / needs_improvement / pattern_examples) 추출
    2) 각 needs_improvement에 대해 키워드로 VectorDB 검색 (각각 독립적으로)
    3) key_moment + 검색 결과를 LLM에 넣어서 reason, better_response 재생성
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not utterances_labeled:
        return {"key_moments": {"positive": [], "needs_improvement": [], "pattern_examples": []}}
    
    # ========== 1단계: LLM으로 key_moments 추출 ==========
    structured_llm = get_structured_llm(KeyMomentsResponse, mini=False)
    
    utterances_str = "\n".join([
        f"{i}. [{utt.get('speaker', '').lower()}] [{utt.get('label', '')}] "
        f"{utt.get('original_ko', utt.get('korean', utt.get('text', '')))}"
        for i, utt in enumerate(utterances_labeled)
    ])
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    try:
        res = (_KEY_MOMENTS_PROMPT | structured_llm).invoke({
            "utterances_labeled": utterances_str,
            "patterns": patterns_str,
        })
    except Exception as e:
        print(f"Key moments error (LLM): {e}")
        import traceback
        traceback.print_exc()
        return _fallback_key_moments(utterances_labeled, patterns)
    
    if not isinstance(res, KeyMomentsResponse):
        return _fallback_key_moments(utterances_labeled, patterns)
    
    key_moments_content = res.key_moments
    
    # ========== 2단계: 한국어 원문 매핑 ==========
    
    # pattern_examples 변환 (먼저 처리)
    pattern_examples_list: List[Dict[str, Any]] = []
    for ex in key_moments_content.pattern_examples:
        dialogue_with_ko = _map_dialogue_to_ko(ex.dialogue, utterances_labeled)
        pattern_examples_list.append({
            "pattern_name": ex.pattern_name,
            "occurrences": ex.occurrences,
            "dialogue": dialogue_with_ko,
            "problem_explanation": ex.problem_explanation,
            "suggested_response": ex.suggested_response,
        })
    
    # ========== 3단계: positive / needs_improvement 처리 (VectorDB 검색 + LLM 재생성) ==========
    use_vector_db_env = os.getenv("USE_VECTOR_DB", "true").lower()
    use_vector_db = use_vector_db_env != "false"
    
    # positive 개선용 LLM
    improve_positive_llm = get_structured_llm(ImprovedPositiveMoment, mini=False)
    
    # positive 변환 + (선택적) VectorDB + LLM 개선
    positive_list: List[Dict[str, Any]] = []
    for moment in key_moments_content.positive:
        dialogue_with_ko = _map_dialogue_to_ko(moment.dialogue, utterances_labeled)
        
        expert_references: List[Dict[str, Any]] = []
        reference_descriptions: List[str] = []
        improved_reason = moment.reason
        
        if use_vector_db:
            try:
                # positive에서도 동일한 방식으로 검색 쿼리 생성
                query = _create_search_query_from_moment(moment, patterns)
                print(f"[VectorDB] positive 검색 - query: {query}")
                
                expert_advice = search_expert_advice(
                    query=query,
                    top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_POSITIVE", "2")),
                    threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD_POSITIVE", "0.2")),
                )
                
                for advice in expert_advice:
                    content = advice.get("content", "") or ""
                    excerpt = content[:200] + "..." if len(content) > 200 else content
                    expert_references.append({
                        "title": advice.get("title", ""),
                        "source": advice.get("source", ""),
                        "author": advice.get("author", ""),
                        "excerpt": excerpt,
                        "relevance_score": advice.get("relevance_score", 0.0),
                    })
                
                if expert_references:
                    sources = list({r.get("source", "") for r in expert_references if r.get("source")})
                    authors = list({r.get("author", "") for r in expert_references if r.get("author")})
                    reference_descriptions = sources + authors
                    print(f"[VectorDB] positive 검색 성공 - {len(expert_references)}개 레퍼런스 추가됨")
                else:
                    print(f"[VectorDB] positive 검색 결과 없음")
            except Exception as e:
                print(f"[VectorDB] positive 검색 오류: {e}")
                import traceback
                traceback.print_exc()
                expert_references = []
                reference_descriptions = []
        
        # positive reason을 전문가 조언으로 강화
        if expert_references:
            try:
                expert_advice_text = "\n".join([
                    f"- {ref['title']} ({ref.get('source', '')}):\n  {ref['excerpt']}"
                    for ref in expert_references
                ])
                dialogue_text = "\n".join([
                    f"{d['speaker']}: {d['text']}"
                    for d in dialogue_with_ko
                ])
                
                improved_res = (_IMPROVE_POSITIVE_PROMPT | improve_positive_llm).invoke({
                    "dialogue": dialogue_text,
                    "pattern_hint": moment.pattern_hint or "",
                    "initial_reason": moment.reason,
                    "expert_advice": expert_advice_text,
                })
                
                if isinstance(improved_res, ImprovedPositiveMoment):
                    improved_reason = improved_res.reason
                    print("[Key Moments] positive reason 개선 완료")
                else:
                    print("[Key Moments] positive reason 개선 실패 - 기본값 사용")
            except Exception as e:
                print(f"[Key Moments] positive 개선 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 시 초기 reason 사용
        
        positive_list.append({
            "dialogue": dialogue_with_ko,
            "reason": improved_reason,
            "pattern_hint": moment.pattern_hint,
            "reference_descriptions": reference_descriptions,
        })
    
    # needs_improvement 처리 (VectorDB 검색 + LLM 재생성)
    needs_improvement_list: List[Dict[str, Any]] = []
    # needs_improvement 개선용 LLM
    improve_llm = get_structured_llm(ImprovedNeedsImprovement, mini=False)
    
    for moment in key_moments_content.needs_improvement:
        dialogue_with_ko = _map_dialogue_to_ko(moment.dialogue, utterances_labeled)
        
        expert_references: List[Dict[str, Any]] = []
        reference_descriptions: List[str] = []
        improved_reason = moment.reason
        improved_better_response = moment.better_response
        
        # 2-1. VectorDB 검색: 각 needs_improvement마다 독립적으로 검색
        if use_vector_db:
            try:
                query = _create_search_query_from_moment(moment, patterns)
                print(f"[VectorDB] needs_improvement 검색 - query: {query}")
                
                expert_advice = search_expert_advice(
                    query=query,
                    top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_NEEDS_IMPROVEMENT", "2")),
                    threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.3")),
                )
                
                for advice in expert_advice:
                    content = advice.get("content", "") or ""
                    excerpt = content[:200] + "..." if len(content) > 200 else content
                    expert_references.append({
                        "title": advice.get("title", ""),
                        "source": advice.get("source", ""),
                        "author": advice.get("author", ""),
                        "excerpt": excerpt,
                        "relevance_score": advice.get("relevance_score", 0.0),
                    })
                
                if expert_references:
                    sources = list({r.get("source", "") for r in expert_references if r.get("source")})
                    authors = list({r.get("author", "") for r in expert_references if r.get("author")})
                    reference_descriptions = sources + authors
                    print(f"[VectorDB] 검색 성공 - {len(expert_references)}개 레퍼런스 추가됨")
                else:
                    print(f"[VectorDB] 검색 결과 없음")
            except Exception as e:
                print(f"[VectorDB] 검색 오류: {e}")
                import traceback
                traceback.print_exc()
                expert_references = []
                reference_descriptions = []
        
        # 2-2. key_moment + 검색 결과를 LLM에 넣어서 reason, better_response 재생성
        if expert_references:
            try:
                # 전문가 조언 포맷팅
                expert_advice_text = "\n".join([
                    f"- {ref['title']} ({ref.get('source', '')}):\n  {ref['excerpt']}"
                    for ref in expert_references
                ])
                
                # 대화 포맷팅
                dialogue_text = "\n".join([
                    f"{d['speaker']}: {d['text']}"
                    for d in dialogue_with_ko
                ])
                
                # LLM으로 reason, better_response 재생성
                improved_res = (_IMPROVE_NEEDS_IMPROVEMENT_PROMPT | improve_llm).invoke({
                    "dialogue": dialogue_text,
                    "pattern_hint": moment.pattern_hint or "",
                    "initial_reason": moment.reason,
                    "initial_better_response": moment.better_response,
                    "expert_advice": expert_advice_text,
                })
                
                if isinstance(improved_res, ImprovedNeedsImprovement):
                    improved_reason = improved_res.reason
                    improved_better_response = improved_res.better_response
                    print(f"[Key Moments] needs_improvement 개선 완료")
                else:
                    print(f"[Key Moments] needs_improvement 개선 실패 - 기본값 사용")
            except Exception as e:
                print(f"[Key Moments] needs_improvement 개선 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 시 초기값 사용
        
        needs_improvement_list.append({
            "dialogue": dialogue_with_ko,
            "reason": improved_reason,
            "better_response": improved_better_response,
            "pattern_hint": moment.pattern_hint,
            "expert_references": expert_references,
            "reference_descriptions": reference_descriptions,
        })
    
    return {
        "key_moments": {
            "positive": positive_list,
            "needs_improvement": needs_improvement_list,
            "pattern_examples": pattern_examples_list,
        }
    }


def _fallback_key_moments(utterances_labeled: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """폴백: 패턴 기반으로 핵심 순간 생성"""
    positive_list = []
    needs_improvement_list = []
    pattern_examples_list = []
    
    # 패턴 기반으로 pattern_examples 생성
    for pattern in patterns[:5]:
        utterance_indices = pattern.get("utterance_indices", [])
        if not utterance_indices:
            continue
        
        # 발화 추출 (인덱스 범위 내, 한국어 원문 사용)
        dialogue = []
        for idx in utterance_indices[:5]:  # 최대 5개 발화
            if 0 <= idx < len(utterances_labeled):
                utt = utterances_labeled[idx]
                speaker = utt.get("speaker", "").lower()
                if speaker in ["mom", "mother", "parent", "엄마", "아빠"]:
                    speaker = "parent"
                elif speaker in ["chi", "child", "kid", "아이"]:
                    speaker = "child"
                # 한국어 원문 우선 사용
                text = utt.get("original_ko", utt.get("korean", utt.get("text", "")))
                if text:
                    dialogue.append({"speaker": speaker, "text": text})
        
        if dialogue:
            pattern_examples_list.append({
                "pattern_name": pattern.get("pattern_name", "Unknown Pattern"),
                "occurrences": pattern.get("occurrence_count", 1),
                "dialogue": dialogue,
                "problem_explanation": pattern.get("description", "패턴이 감지되었습니다."),
                "suggested_response": pattern.get("suggested_response", "더 나은 응답을 고려해보세요.")
            })
    
    # 긍정적 순간 찾기 (PR 라벨이 있는 발화, 한국어 원문 사용)
    for i, utt in enumerate(utterances_labeled):
        if utt.get("label") == "PR" and i < len(utterances_labeled) - 1:
            speaker = utt.get("speaker", "").lower()
            if speaker in ["mom", "mother", "parent", "엄마", "아빠"]:
                speaker = "parent"
            elif speaker in ["chi", "child", "kid", "아이"]:
                speaker = "child"
            
            # 한국어 원문 우선 사용
            dialogue = [
                {"speaker": speaker, "text": utt.get("original_ko", utt.get("korean", utt.get("text", "")))}
            ]
            # 다음 발화도 포함
            if i + 1 < len(utterances_labeled):
                next_utt = utterances_labeled[i + 1]
                next_speaker = next_utt.get("speaker", "").lower()
                if next_speaker in ["mom", "mother", "parent", "엄마", "아빠"]:
                    next_speaker = "parent"
                elif next_speaker in ["chi", "child", "kid", "아이"]:
                    next_speaker = "child"
                dialogue.append({
                    "speaker": next_speaker,
                    "text": next_utt.get("original_ko", next_utt.get("korean", next_utt.get("text", "")))
                })
            
            if len(positive_list) < 3 and dialogue:
                positive_list.append({
                    "dialogue": dialogue,
                    "reason": "긍정적 상호작용이 감지되었습니다.",
                    "pattern_hint": "긍정적 상호작용"
                })
    
    # 개선이 필요한 순간 찾기 (NEG, CMD 라벨이 있는 발화, 한국어 원문 사용)
    for i, utt in enumerate(utterances_labeled):
        if utt.get("label") in ["NEG", "CMD"] and i < len(utterances_labeled) - 1:
            speaker = utt.get("speaker", "").lower()
            if speaker in ["mom", "mother", "parent", "엄마", "아빠"]:
                speaker = "parent"
            elif speaker in ["chi", "child", "kid", "아이"]:
                speaker = "child"
            
            # 한국어 원문 우선 사용
            dialogue = [
                {"speaker": speaker, "text": utt.get("original_ko", utt.get("korean", utt.get("text", "")))}
            ]
            # 다음 발화도 포함
            if i + 1 < len(utterances_labeled):
                next_utt = utterances_labeled[i + 1]
                next_speaker = next_utt.get("speaker", "").lower()
                if next_speaker in ["mom", "mother", "parent", "엄마", "아빠"]:
                    next_speaker = "parent"
                elif next_speaker in ["chi", "child", "kid", "아이"]:
                    next_speaker = "child"
                dialogue.append({
                    "speaker": next_speaker,
                    "text": next_utt.get("original_ko", next_utt.get("korean", next_utt.get("text", "")))
                })
            
            if len(needs_improvement_list) < 3 and dialogue:
                needs_improvement_list.append({
                    "dialogue": dialogue,
                    "reason": "개선이 필요한 상호작용이 감지되었습니다.",
                    "better_response": "아이의 감정을 먼저 읽어주시고 공감해주세요.",
                    "pattern_hint": "개선 필요"
                })
    
    return {
        "key_moments": {
            "positive": positive_list,
            "needs_improvement": needs_improvement_list,
            "pattern_examples": pattern_examples_list
        }
    }

