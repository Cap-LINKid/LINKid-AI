from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

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
    reference_descriptions: List[str] = Field(default_factory=list)
    pattern_hint: str
    expert_references: List[ExpertReference] = Field(default_factory=list)


class PatternExample(BaseModel):
    pattern_name: str
    occurrences: int
    occurred_at: str
    dialogue: List[DialogueLine]
    problem_explanation: str
    suggested_response: str


class KeyMomentsResult(BaseModel):
    positive: List[PositiveMoment] = Field(default_factory=list)
    needs_improvement: List[NeedsImprovementMoment] = Field(default_factory=list)
    pattern_examples: List[PatternExample] = Field(default_factory=list)


class PositiveMomentResponse(BaseModel):
    positive: List[PositiveMoment] = Field(default_factory=list)


class NeedsImprovementMomentResponse(BaseModel):
    needs_improvement: List[NeedsImprovementMoment] = Field(default_factory=list)


class PatternExampleResponse(BaseModel):
    pattern_examples: List[PatternExample] = Field(default_factory=list)


# -------------------------------------------------------------------------
# 2. LLM 프롬프트 (각 moment 타입별로 분리)
# -------------------------------------------------------------------------

_POSITIVE_MOMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 아동 심리 및 부모 교육 전문가입니다.
입력으로 부모-자녀 대화, 탐지된 패턴 정보, 전문가 조언을 바탕으로
Positive Moment 분석을 생성해야 합니다.

==============================
📌 Positive Moment 규칙 (매우 중요)
==============================
- positive_context.pattern_type 이 'positive'일 때만 positive moment를 생성할 수 있습니다.
- pattern_type이 'positive'가 아니라면, 반드시 빈 배열 []을 반환하세요.

*** [중요] Dialogue 내용 엄격 검증 ***
1. **화자 확인 필수**: dialogue 배열의 각 발화에서 'speaker' 필드를 정확히 확인하세요.
   - 'parent' 또는 'child' 중 누가 말했는지 정확히 파악해야 합니다.
   - reason을 작성할 때 화자를 잘못 해석하지 마세요.
   - 예: dialogue에서 "child"가 말한 내용을 "parent"가 말한 것처럼 설명하면 안 됩니다.

2. **실제 내용 검증 필수**: 
   - dialogue의 실제 텍스트 내용을 정확히 읽고 분석하세요.
   - 패턴 이름이 '구체적 칭찬'이라고 해도, 실제 dialogue에서 부모가 구체적으로 칭찬하고 있는지 확인하세요.
   - dialogue 내용이 불분명하거나, 실제로 positive한 행동이 명확하지 않다면 빈 배열 []을 반환하세요.

3. **억지로 만들지 말 것**:
   - dialogue 내용이 실제로 positive moment를 보여주지 않는다면, 억지로 해석하지 마세요.
   - 패턴이 탐지되었다고 해서 무조건 positive moment를 만들 필요는 없습니다.
   - 실제로 긍정적인 상호작용이 명확하게 드러나는 경우에만 positive moment를 생성하세요.
   - 애매하거나 불분명한 경우에는 반드시 빈 배열 []을 반환하세요.

4. **구체적 칭찬 패턴의 경우**:
   - 부모가 실제로 아이의 구체적인 행동이나 노력을 칭찬하고 있어야 합니다.
   - 단순히 긍정적인 단어만 사용했다고 해서 구체적 칭찬이 아닙니다.
   - 예: "잘했어"만으로는 부족하고, "무엇을" 잘했는지 구체적으로 언급되어야 합니다.

5. **reason 작성 시**:
   - dialogue의 실제 내용을 바탕으로 작성하세요.
   - 화자가 누구인지 정확히 파악한 후 작성하세요.
   - 전문가 excerpt 1개를 자연스럽게 섞어 쓰기
   - reference_descriptions는 최대 2개
   - 전문가 excerpt와 대화의 맥락과 상황을 파악하여 2~4 줄 정도로 길고 전문가가 말하듯이 구체적으로 작성
   - tone은 따뜻하고 전문적이지만, ~~합니다.와 같이 공손하게 말할 수 있도록 한다.

==============================
📌 입력 데이터
==============================
[Positive Context]
{positive_context}

[Expert References]
{expert_references}

위 정보를 바탕으로 positive moment를 생성하십시오.
*** 중요: dialogue의 실제 내용과 화자를 정확히 확인하고, 실제로 positive한 순간이 명확한 경우에만 생성하세요. 애매하거나 불분명하면 빈 배열 []을 반환하세요. ***
"""
    ),
    (
        "human",
        "위 내용을 반영하여 positive moment를 생성하세요. dialogue의 실제 내용과 화자를 정확히 확인하고, 실제로 positive한 순간이 명확한 경우에만 생성하세요. 애매하거나 불분명하면 빈 배열 []을 반환하세요."
    ),
])

_NEEDS_IMPROVEMENT_MOMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 아동 심리 및 부모 교육 전문가입니다.
입력으로 부모-자녀 대화, 탐지된 패턴 정보, 전문가 조언을 바탕으로
Needs Improvement Moment 분석을 생성해야 합니다.

==============================
📌 Needs Improvement 규칙
==============================
- 가장 심각한 부정 패턴 하나만 사용
- reason: 상황 요약 → 문제점 → 아동 발달 영향 → 전문가 조언 인용(1~2개)
- better_response: 실제 사용할 수 있는 구체 대사
- reference_descriptions: 최대 2개
- reason: 전문가 excerpt와 대화의 맥락과 상황을 파악하여 2~4 줄 정도로 길고 구체적으로 작성
- better_response: 부모가 실제 사용할 수 있는 대사 형태와 이런 대안이 나온 이유를 뽑힌 전문가 excerpt를 반영해서 구체적인 행위로 작성
- tone은 따뜻하고 전문적이지만, ~~합니다.와 같이 공손하게 말할 수 있도록 한다.

==============================
📌 입력 데이터
==============================
[Needs Improvement Context]
{improvement_context}

[Expert References]
{expert_references}

위 정보를 바탕으로 needs improvement moment를 생성하십시오.
"""
    ),
    (
        "human",
        "위 내용을 반영하여 needs improvement moment를 생성하세요."
    ),
])

_PATTERN_EXAMPLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 아동 심리 및 부모 교육 전문가입니다.
입력으로 부모-자녀 대화, 탐지된 패턴 정보, 전문가 조언을 바탕으로
Pattern Example 분석을 생성해야 합니다.

==============================
📌 Pattern Examples 규칙
==============================
- Needs Improvement 다음으로 심각한 1개의 패턴을 선택하여 생성
- 이유와 조언은 전문가 excerpt와 대화의 맥락과 상황을 파악하여 구체적으로 작성
- succinct한 problem_explanation & suggested_response 작성하고, 1~2줄 정도로 구체적으로 작성
- tone은 따뜻하고 전문적이지만, ~~합니다.와 같이 공손하게 말할 수 있도록 한다.

==============================
📌 입력 데이터
==============================
[Pattern Examples 후보]
{examples_context}

[Expert References]
{expert_references}

위 정보를 바탕으로 pattern example을 생성하십시오.
"""
    ),
    (
        "human",
        "위 내용을 반영하여 pattern example을 생성하세요."
    ),
])


# -------------------------------------------------------------------------
# 3. Helper 함수
# -------------------------------------------------------------------------

def _extract_dialogue(utterances: List[Dict], indices: List[int], max_items: int = 4) -> List[Dict]:
    """
    주어진 utterance indices 중 앞에서부터 최대 max_items개까지만 dialogue로 추출.
    순서는 원래 순서를 유지하며, 과도한 길이로 인해 LLM이 장문 출력하는 것을 방지.
    """
    dialogue = []
    limited_indices = sorted(indices)[:max_items]
    for idx in limited_indices:
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

async def _key_moments_node_async(state: Dict[str, Any]) -> Dict[str, Any]:
    utterances = state.get("utterances_labeled") or state.get("utterances_ko", [])
    patterns = state.get("patterns", [])

    print(f"[KeyMomentsAgent] 입력 데이터 확인:")
    print(f"  - 발화 수: {len(utterances)}개")
    print(f"  - 패턴 수: {len(patterns)}개")

    if not patterns:
        print("[KeyMomentsAgent] 경고: 분석할 패턴이 없음")
        return {"key_moments": None}

    # Severity 기준 정렬
    print("[KeyMomentsAgent] 패턴 분류 및 정렬 중...")
    severity_order = {"high": 3, "medium": 2, "low": 1}
    neg_patterns = [p for p in patterns if p.get("pattern_type") == "negative"]
    pos_patterns = [p for p in patterns if p.get("pattern_type") == "positive"]

    neg_patterns.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 1), reverse=True)

    print(f"[KeyMomentsAgent] 패턴 분류 완료: 부정 {len(neg_patterns)}개, 긍정 {len(pos_patterns)}개")

    # 선택 대상
    target_positive = pos_patterns[0] if pos_patterns else None
    target_improvement = neg_patterns[0] if neg_patterns else None
    target_examples = neg_patterns[1:2]  # 딱 1개만
    
    if target_positive:
        print(f"[KeyMomentsAgent] 긍정 패턴 선택: {target_positive.get('pattern_name', 'N/A')}")
    if target_improvement:
        print(f"[KeyMomentsAgent] 개선 패턴 선택: {target_improvement.get('pattern_name', 'N/A')}")
    if target_examples:
        print(f"[KeyMomentsAgent] 예시 패턴 선택: {target_examples[0].get('pattern_name', 'N/A')}")

    # ---------------------------------------------------------
    # RAG: 전문가 조언 검색 (긍정 / 최악 / 두 번째 패턴 각각) - 병렬 실행
    # ---------------------------------------------------------
    # 블로킹 호출을 별도 스레드에서 실행하여 이벤트 루프를 블로킹하지 않도록 함
    search_tasks = [
        asyncio.to_thread(_search_refs_for_pattern, target_positive),
        asyncio.to_thread(_search_refs_for_pattern, target_improvement),
    ]
    if target_examples:
        search_tasks.append(asyncio.to_thread(_search_refs_for_pattern, target_examples[0]))
    else:
        search_tasks.append(asyncio.to_thread(lambda: []))
    
    pos_expert_refs, neg_expert_refs, ex_expert_refs = await asyncio.gather(*search_tasks)

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
    # 병렬 LLM 호출 (각 moment 타입별로 분리)
    # ---------------------------------------------------------
    
    async def _generate_positive_moment() -> List[PositiveMoment]:
        """Positive Moment 생성"""

        # 패턴이 positive인지 확인
        if not target_positive or target_positive.get("pattern_type") != "positive":
            return []

        # Dialogue 내용 사전 검증: 실제로 positive한 내용인지 확인
        dialogue_dicts = _extract_dialogue(utterances, target_positive["utterance_indices"])
        
        # 부모 발화가 실제로 positive한 내용인지 간단히 확인
        # 구체적 칭찬 패턴의 경우, 부모가 실제로 칭찬하고 있는지 확인
        pattern_name = target_positive.get("pattern_name", "")
        has_parent_positive_utterance = False
        
        for d in dialogue_dicts:
            speaker = d.get("speaker", "").lower()
            text = d.get("text", "").strip()
            
            # 부모 발화인 경우
            if speaker == "parent":
                # 구체적 칭찬 패턴의 경우, 실제로 칭찬하는 내용이 있는지 확인
                if "칭찬" in pattern_name or "praise" in pattern_name.lower():
                    # 부정적인 단어가 포함되어 있으면 제외
                    negative_keywords = ["화가 나", "몰라", "잘못", "안 그럴게", "예쁘게 말해"]
                    if any(keyword in text for keyword in negative_keywords):
                        continue
                    # 긍정적인 칭찬 표현이 있는지 확인
                    positive_keywords = ["잘했", "훌륭", "좋", "대단", "멋있", "칭찬"]
                    if any(keyword in text for keyword in positive_keywords):
                        has_parent_positive_utterance = True
                        break
                else:
                    # 다른 positive 패턴의 경우, 부모 발화가 있으면 일단 통과
                    has_parent_positive_utterance = True
                    break
        
        # 부모의 positive한 발화가 없거나, dialogue가 비어있으면 빈 배열 반환
        if not dialogue_dicts or (not has_parent_positive_utterance and "칭찬" in pattern_name):
            return []

        try:
            llm = get_structured_llm(PositiveMomentResponse)

            pos_refs_json = json.dumps([r.dict() for r in pos_expert_refs], ensure_ascii=False)

            # pattern_type을 명시적으로 전달
            enriched_ctx = json.dumps({
                "pattern_name": target_positive["pattern_name"],
                "pattern_type": "positive",
                "description": target_positive["description"],
                "dialogue": dialogue_dicts
            }, ensure_ascii=False)

            result = await (_POSITIVE_MOMENT_PROMPT | llm).ainvoke({
                "positive_context": enriched_ctx,
                "expert_references": pos_refs_json
            })

            # LLM 결과도 검증: 빈 배열이거나, dialogue 내용과 맞지 않으면 빈 배열 반환
            if not result.positive or len(result.positive) == 0:
                return []
            
            # 첫 번째 positive moment의 dialogue와 실제 dialogue 비교
            # (LLM이 잘못 해석했을 수 있으므로)
            return result.positive
        except Exception as e:
            print(f"Positive moment LLM 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _generate_needs_improvement_moment() -> List[NeedsImprovementMoment]:
        """Needs Improvement Moment 생성"""
        if not target_improvement:
            return []
        
        try:
            llm = get_structured_llm(NeedsImprovementMomentResponse)
            neg_refs_json = json.dumps([r.dict() for r in neg_expert_refs], ensure_ascii=False)
            
            result = await (_NEEDS_IMPROVEMENT_MOMENT_PROMPT | llm).ainvoke({
                "improvement_context": imp_ctx,
                "expert_references": neg_refs_json
            })
            return result.needs_improvement
        except Exception as e:
            print(f"Needs improvement moment LLM 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _generate_pattern_example() -> List[PatternExample]:
        """Pattern Example 생성"""
        if not target_examples:
            return []
        
        try:
            llm = get_structured_llm(PatternExampleResponse)
            ex_refs_json = json.dumps([r.dict() for r in ex_expert_refs], ensure_ascii=False)
            
            result = await (_PATTERN_EXAMPLE_PROMPT | llm).ainvoke({
                "examples_context": ex_ctx,
                "expert_references": ex_refs_json
            })
            return result.pattern_examples
        except Exception as e:
            print(f"Pattern example LLM 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # 병렬 실행
    try:
        positive_list, needs_improvement_list, pattern_examples_list = await asyncio.gather(
            _generate_positive_moment(),
            _generate_needs_improvement_moment(),
            _generate_pattern_example()
        )
        
        final_data = KeyMomentsResult(
            positive=positive_list,
            needs_improvement=needs_improvement_list,
            pattern_examples=pattern_examples_list
        )
    except Exception as e:
        print(f"Key moments 병렬 LLM 호출 오류: {e}")
        import traceback
        traceback.print_exc()
        # 기본값 반환
        final_data = KeyMomentsResult(
            positive=[],
            needs_improvement=[],
            pattern_examples=[]
        )

    # ---------------------------------------------------------
    # 후처리: 필드 정제/보정
    # ---------------------------------------------------------

    # Positive 보정
    if target_positive and final_data.positive and len(final_data.positive) > 0:
        pm = final_data.positive[0]
        dialogue_dicts = _extract_dialogue(utterances, target_positive["utterance_indices"])
        pm.dialogue = [
            DialogueLine(speaker=d["speaker"], text=d["text"])
            for d in dialogue_dicts
        ]
        pm.pattern_hint = target_positive["pattern_name"]
        # Positive는 긍정 패턴에 대한 RAG 결과를 기반으로 reference_descriptions 구성
        pm.reference_descriptions = _ref_desc_from_refs(pos_expert_refs)

    # Needs Improvement 보정
    if target_improvement and final_data.needs_improvement and len(final_data.needs_improvement) > 0:
        ni = final_data.needs_improvement[0]
        dialogue_dicts = _extract_dialogue(utterances, target_improvement["utterance_indices"])
        ni.dialogue = [
            DialogueLine(speaker=d["speaker"], text=d["text"])
            for d in dialogue_dicts
        ]
        ni.pattern_hint = target_improvement["pattern_name"]
        ni.expert_references = neg_expert_refs
        ni.reference_descriptions = _ref_desc_from_refs(neg_expert_refs)

    # Pattern Examples 보정 (두 번째로 심각한 패턴 1개)
    # LLM이 생성하지 못한 경우 후처리에서 생성
    if target_examples and len(target_examples) > 0:
        if len(final_data.pattern_examples) == 0:
            # LLM이 생성하지 못한 경우 직접 생성
            ex_target = target_examples[0]
            utterance_indices = ex_target.get("utterance_indices", [])
            dialogue_lines = [
                DialogueLine(
                    speaker="parent" if utt.get("speaker") in ["Parent", "Mom", "Dad", "부모", "A"] else "child",
                    text=utt.get("original_ko") or utt.get("korean") or utt.get("text", "")
                )
                for idx in sorted(utterance_indices)
                if 0 <= idx < len(utterances)
                for utt in [utterances[idx]]
            ]
            
            pe = PatternExample(
                pattern_name=ex_target.get("pattern_name", ""),
                occurrences=len(utterance_indices),
                occurred_at=f"{utterance_indices[0] // 6}분 {utterance_indices[0] * 10 % 60}초" if utterance_indices else "0분 0초",
                dialogue=dialogue_lines,
                problem_explanation=ex_target.get("description", "패턴이 발견되었습니다."),
                suggested_response="상황에 맞는 대안적 대응이 필요합니다."
            )
            final_data.pattern_examples.append(pe)
        else:
            # LLM이 생성한 경우 보정
            for i, ex_target in enumerate(target_examples):
                if i < len(final_data.pattern_examples):
                    pe = final_data.pattern_examples[i]
                    pe.pattern_name = ex_target["pattern_name"]
                    utterance_indices = ex_target.get("utterance_indices", [])
                    # Dict를 DialogueLine 리스트로 변환
                    dialogue_dicts = _extract_dialogue(utterances, utterance_indices)
                    pe.dialogue = [
                        DialogueLine(speaker=d["speaker"], text=d["text"])
                        for d in dialogue_dicts
                    ]
                    pe.occurrences = len(utterance_indices)
                    if utterance_indices:
                        idx = utterance_indices[0]
                        pe.occurred_at = f"{idx // 6}분 {idx * 10 % 60}초"
                    else:
                        pe.occurred_at = "0분 0초"

    result = final_data.dict()
    print(f"[KeyMomentsAgent] 핵심 순간 분석 완료:")
    print(f"  - 긍정적 순간: {len(result.get('positive', []))}개")
    print(f"  - 개선 필요 순간: {len(result.get('needs_improvement', []))}개")
    print(f"  - 패턴 예시: {len(result.get('pattern_examples', []))}개")
    
    return {"key_moments": result}


def key_moments_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """동기 래퍼 함수 - async 함수를 실행"""
    print("\n" + "="*60)
    print("[KeyMomentsAgent] 핵심 순간 분석 시작")
    print("="*60)
    try:
        # 이미 실행 중인 이벤트 루프가 있는 경우
        loop = asyncio.get_running_loop()
        # 실행 중인 루프가 있으면 새 스레드에서 실행
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _key_moments_node_async(state))
            result = future.result()
            print("="*60 + "\n")
            return result
    except RuntimeError:
        # 이벤트 루프가 없는 경우
        result = asyncio.run(_key_moments_node_async(state))
        print("="*60 + "\n")
        return result