from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm
from src.utils.vector_store import search_expert_advice


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
            "1. 'positive': 부모가 잘 대응한 순간들 (감정 코칭, 공감, 인정 등)\n"
            "2. 'needs_improvement': 부모의 응답을 개선할 수 있는 순간들\n"
            "3. 'pattern_examples': 감지된 패턴의 구체적인 대화 발췌 예시들\n\n"
            "각 순간에 대해 발화에서 실제 대화(발화자와 한국어 원문 텍스트)를 포함하세요. "
            "대화는 핵심 순간을 구성하는 연속된 발화들의 리스트여야 합니다. "
            "'needs_improvement' 순간의 경우, 'better_response' 제안을 제공하세요. "
            "전문가 조언이 제공된 경우, 이를 참고하여 더 나은 응답을 생성하세요. "
            "'pattern_examples'의 경우, 패턴 이름, 발생 횟수, 문제 설명, 제안된 응답을 포함하세요. "
            "모든 설명과 응답은 한국어로 작성하세요."
        ),
    ),
    (
        "human",
        (
            "라벨링된 발화:\n{utterances_labeled}\n\n"
            "감지된 패턴:\n{patterns}\n\n"
            "{expert_advice_section}\n\n"
            "상호작용에서 핵심 순간을 추출하고 분류하세요. "
            "각 순간에 대해 발화자와 한국어 원문 텍스트가 포함된 실제 대화 발췌를 포함하세요."
        ),
    ),
])


def _build_needs_improvement_query(moment: Dict[str, Any]) -> str:
    """
    needs_improvement 순간에서 검색 쿼리 생성
    """
    pattern_hint = moment.get("pattern_hint", "")
    reason = moment.get("reason", "")
    
    # 패턴 힌트가 있으면 우선 사용
    if pattern_hint:
        query = f"{pattern_hint} 패턴 개선 방법"
    else:
        # reason에서 키워드 추출
        query = f"{reason} 개선 방법"
    
    return query


def key_moments_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑥ key_moments: 핵심 순간 (LLM)
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not utterances_labeled:
        return {"key_moments": {"positive": [], "needs_improvement": [], "pattern_examples": []}}
    
    # Structured LLM 사용
    structured_llm = get_structured_llm(KeyMomentsResponse, mini=False)
    
    # 포맷팅 - 발화를 인덱스와 함께 표시 (한국어 원문 사용)
    utterances_str = "\n".join([
        f"{i}. [{utt.get('speaker', '').lower()}] [{utt.get('label', '')}] {utt.get('original_ko', utt.get('korean', utt.get('text', '')))}"
        for i, utt in enumerate(utterances_labeled)
    ])
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    # VectorDB에서 전문가 조언 검색 (프롬프트에 포함)
    expert_advice_section = ""
    use_vector_db = os.getenv("USE_VECTOR_DB", "false").lower() == "true"
    
    if use_vector_db and patterns:
        try:
            # 주요 패턴에 대한 조언 검색
            pattern_names = [p.get("pattern_name") for p in patterns[:3] if p.get("pattern_name")]
            if pattern_names:
                # 첫 번째 패턴으로 검색
                query = f"{pattern_names[0]} 패턴 개선 방법"
                expert_advice = search_expert_advice(
                    query=query,
                    top_k=2,
                    threshold=0.7,
                    filters={
                        "advice_type": ["pattern_advice", "coaching"],
                        "pattern_names": [pattern_names[0]]
                    }
                )
                
                if expert_advice:
                    expert_advice_section = "전문가 조언 참고:\n" + "\n".join([
                        f"- {advice['title']}: {advice['content'][:150]}..."
                        for advice in expert_advice
                    ])
        except Exception as e:
            print(f"VectorDB 검색 오류 (프롬프트): {e}")
            expert_advice_section = ""
    
    if not expert_advice_section:
        expert_advice_section = ""
    
    try:
        res = (_KEY_MOMENTS_PROMPT | structured_llm).invoke({
            "utterances_labeled": utterances_str,
            "patterns": patterns_str,
            "expert_advice_section": expert_advice_section,
        })
        
        # Pydantic 모델에서 데이터 추출
        if isinstance(res, KeyMomentsResponse):
            key_moments_content = res.key_moments
            
            # positive 변환 (한국어 원문 사용)
            positive_list = []
            for moment in key_moments_content.positive:
                # dialogue에서 발화자와 텍스트를 매칭하여 한국어 원문 찾기
                dialogue_with_ko = []
                for utt in moment.dialogue:
                    # utterances_labeled에서 매칭되는 발화 찾기
                    matched_text = utt.text
                    for orig_utt in utterances_labeled:
                        # 발화자와 텍스트로 매칭 (한국어 원문 우선)
                        orig_speaker = orig_utt.get('speaker', '').lower()
                        if orig_speaker in ['mom', 'mother', 'parent', '엄마', '아빠']:
                            orig_speaker = 'parent'
                        elif orig_speaker in ['chi', 'child', 'kid', '아이']:
                            orig_speaker = 'child'
                        
                        if (utt.speaker.lower() == orig_speaker and 
                            (utt.text in orig_utt.get('english', '') or 
                             utt.text in orig_utt.get('text', '') or
                             orig_utt.get('english', '') in utt.text or
                             orig_utt.get('text', '') in utt.text)):
                            matched_text = orig_utt.get('original_ko', orig_utt.get('korean', utt.text))
                            break
                    
                    dialogue_with_ko.append({
                        "speaker": utt.speaker,
                        "text": matched_text
                    })
                
                positive_list.append({
                    "dialogue": dialogue_with_ko,
                    "reason": moment.reason,
                    "pattern_hint": moment.pattern_hint
                })
            
            # needs_improvement 변환 (한국어 원문 사용) + VectorDB 검색
            needs_improvement_list = []
            use_vector_db = os.getenv("USE_VECTOR_DB", "false").lower() == "true"
            
            for moment in key_moments_content.needs_improvement:
                dialogue_with_ko = []
                for utt in moment.dialogue:
                    matched_text = utt.text
                    for orig_utt in utterances_labeled:
                        orig_speaker = orig_utt.get('speaker', '').lower()
                        if orig_speaker in ['mom', 'mother', 'parent', '엄마', '아빠']:
                            orig_speaker = 'parent'
                        elif orig_speaker in ['chi', 'child', 'kid', '아이']:
                            orig_speaker = 'child'
                        
                        if (utt.speaker.lower() == orig_speaker and 
                            (utt.text in orig_utt.get('english', '') or 
                             utt.text in orig_utt.get('text', '') or
                             orig_utt.get('english', '') in utt.text or
                             orig_utt.get('text', '') in utt.text)):
                            matched_text = orig_utt.get('original_ko', orig_utt.get('korean', utt.text))
                            break
                    
                    dialogue_with_ko.append({
                        "speaker": utt.speaker,
                        "text": matched_text
                    })
                
                # VectorDB 검색 (needs_improvement용)
                expert_references = []
                if use_vector_db:
                    try:
                        moment_dict = {
                            "pattern_hint": moment.pattern_hint,
                            "reason": moment.reason
                        }
                        query = _build_needs_improvement_query(moment_dict)
                        
                        # VectorDB 검색
                        expert_advice = search_expert_advice(
                            query=query,
                            top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_NEEDS_IMPROVEMENT", "2")),
                            threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.3")),  # 기본값 0.3으로 변경
                            filters={
                                "advice_type": ["pattern_advice", "coaching"],
                                "pattern_names": [moment.pattern_hint] if moment.pattern_hint else None
                            }
                        )
                        
                        # 레퍼런스 정보 구성
                        expert_references = [
                            {
                                "title": advice["title"],
                                "source": advice["source"],
                                "author": advice.get("author", ""),
                                "excerpt": advice["content"][:200] + "..." if len(advice["content"]) > 200 else advice["content"],
                                "relevance_score": advice["relevance_score"]
                            }
                            for advice in expert_advice
                        ]
                    except Exception as e:
                        print(f"VectorDB 검색 오류 (needs_improvement): {e}")
                        expert_references = []
                
                needs_improvement_list.append({
                    "dialogue": dialogue_with_ko,
                    "reason": moment.reason,
                    "better_response": moment.better_response,
                    "pattern_hint": moment.pattern_hint,
                    "expert_references": expert_references if expert_references else None
                })
            
            # pattern_examples 변환 (한국어 원문 사용)
            pattern_examples_list = []
            for example in key_moments_content.pattern_examples:
                dialogue_with_ko = []
                for utt in example.dialogue:
                    matched_text = utt.text
                    for orig_utt in utterances_labeled:
                        orig_speaker = orig_utt.get('speaker', '').lower()
                        if orig_speaker in ['mom', 'mother', 'parent', '엄마', '아빠']:
                            orig_speaker = 'parent'
                        elif orig_speaker in ['chi', 'child', 'kid', '아이']:
                            orig_speaker = 'child'
                        
                        if (utt.speaker.lower() == orig_speaker and 
                            (utt.text in orig_utt.get('english', '') or 
                             utt.text in orig_utt.get('text', '') or
                             orig_utt.get('english', '') in utt.text or
                             orig_utt.get('text', '') in utt.text)):
                            matched_text = orig_utt.get('original_ko', orig_utt.get('korean', utt.text))
                            break
                    
                    dialogue_with_ko.append({
                        "speaker": utt.speaker,
                        "text": matched_text
                    })
                
                pattern_examples_list.append({
                    "pattern_name": example.pattern_name,
                    "occurrences": example.occurrences,
                    "dialogue": dialogue_with_ko,
                    "problem_explanation": example.problem_explanation,
                    "suggested_response": example.suggested_response
                })
            
            return {
                "key_moments": {
                    "positive": positive_list,
                    "needs_improvement": needs_improvement_list,
                    "pattern_examples": pattern_examples_list
                }
            }
        
        # 폴백: 예상치 못한 형식
        return _fallback_key_moments(utterances_labeled, patterns)
        
    except Exception as e:
        print(f"Key moments error: {e}")
        import traceback
        traceback.print_exc()
        # 에러 시 폴백 사용
        return _fallback_key_moments(utterances_labeled, patterns)


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

