from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Literal, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_llm, get_structured_llm


def _load_pattern_definitions() -> Dict[str, Any]:
    """패턴 정의 JSON 파일 로드"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pattern_file = os.path.join(base_dir, "data", "expert_advice", "pattern_definitions.json")
    
    try:
        with open(pattern_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load pattern definitions: {e}")
        return {
            "positive_patterns": [],
            "negative_patterns": [],
            "additional_patterns": []
        }


_PATTERN_DEFINITIONS = _load_pattern_definitions()


# -----------------------
#  Pydantic 모델 정의
# -----------------------

class PatternDetection(BaseModel):
    """패턴 탐지 결과"""
    pattern_name: str = Field(description="패턴 이름 (한글)")
    description: str = Field(description="판단 근거 (상황과 맥락을 포함하여 구체적으로 기술)")
    utterance_indices: List[int] = Field(description="관련된 발화 번호들")
    severity: Literal["low", "medium", "high"] = Field(description="심각도")
    pattern_type: Literal["positive", "negative"] = Field(description="패턴 타입")


class PatternDetectionList(BaseModel):
    """패턴 탐지 결과 리스트"""
    patterns: List[PatternDetection] = Field(default_factory=list, description="탐지된 패턴 목록")


class PatternValidation(BaseModel):
    """패턴 검증 결과"""
    pattern_index: int = Field(description="원본 배열 인덱스")
    is_valid: bool = Field(description="유효성 여부")
    reason: str = Field(description="판단 근거")
    corrected_pattern_name: Optional[str] = Field(default=None, description="수정이 필요하면 새 패턴 이름 (없으면 null)")


class PatternValidationList(BaseModel):
    """패턴 검증 결과 리스트"""
    validations: List[PatternValidation] = Field(default_factory=list, description="검증 결과 목록")


def _build_pattern_details_for_prompt(mode: Optional[Literal["positive", "negative"]] = None) -> str:
    """
    LLM 프롬프트용 패턴 상세 정보 생성.
    단순 정의뿐만 아니라 context_clues, keywords, anti_patterns를 포함하여
    LLM이 상황 맥락을 파악하고 오탐을 줄일 수 있도록 함.
    """
    sections: List[str] = []

    use_positive = (mode is None) or (mode == "positive")
    use_negative = (mode is None) or (mode == "negative")
    use_additional = (mode is None) or (mode == "positive")

    target_groups = []
    if use_positive:
        target_groups.append(("긍정적 패턴 (Positive Patterns)", _PATTERN_DEFINITIONS.get("positive_patterns", [])))
    if use_negative:
        target_groups.append(("부정적 패턴 (Negative Patterns)", _PATTERN_DEFINITIONS.get("negative_patterns", [])))
    if use_additional:
        target_groups.append(("추가 패턴 (Additional Patterns)", _PATTERN_DEFINITIONS.get("additional_patterns", [])))

    for group_name, patterns in target_groups:
        if not patterns:
            continue
            
        sections.append(f"\n=== {group_name} ===")
        for p in patterns:
            name = p.get("name", "")
            english = p.get("english_name", "")
            definition = p.get("definition", "")
            dpics_code = p.get("dpics_code", "")
            
            # 패턴 헤더
            pattern_info = f"\n- 패턴명: '{name}' ({english})"
            
            # DPICS 코드 (참고용)
            if dpics_code:
                if isinstance(dpics_code, list):
                    codes = [("RD (RF와 동일)" if c == "RF" else c) for c in dpics_code]
                    pattern_info += f" [참고 라벨: {', '.join(codes)}]"
                else:
                    code_str = "RD (RF와 동일)" if dpics_code == "RF" else dpics_code
                    pattern_info += f" [참고 라벨: {code_str}]"
            
            pattern_info += f"\n  정의: {definition}"

            # [핵심] 맥락적 단서 (Context Clues) - 중괄호가 있다면 이스케이프 처리 필요하지만 보통 텍스트임
            if "context_clues" in p and p["context_clues"]:
                pattern_info += "\n  [탐지 단서(힌트)]: "
                for clue in p["context_clues"]:
                    pattern_info += f"\n    * {clue}"

            # 오답 방지 (Anti-Patterns)
            if "anti_patterns" in p and p["anti_patterns"]:
                pattern_info += "\n  [주의! 이건 해당 패턴이 아님]:"
                for anti in p["anti_patterns"]:
                    pattern_info += f"\n    X {anti}"

            # 키워드 (참고용)
            if "keywords" in p and p["keywords"]:
                pattern_info += f"\n  [관련 키워드]: {', '.join(p['keywords'][:5])}"

            sections.append(pattern_info)

    return "\n".join(sections)


def _get_pattern_prompt(mode: Literal["positive", "negative"]) -> ChatPromptTemplate:
    """
    맥락 인식과 의미론적 매칭에 중점을 둔 프롬프트 생성
    """
    pattern_details = _build_pattern_details_for_prompt(mode)

    if mode == "negative":
        mode_instruction = (
            "현재 '부정적 상호작용(negative)' 모드입니다.\n"
            "아이의 자존감을 낮추거나, 자율성을 침해하거나, 관계를 해치는 패턴(비난, 무시, 과도한 통제 등)을 찾으십시오."
        )
    else:
        mode_instruction = (
            "현재 '긍정적 상호작용(positive)' 모드입니다.\n"
            "아이의 성장을 돕고, 정서를 지지하며, 관계를 강화하는 패턴(칭찬, 경청, 지지 등)을 찾으십시오."
        )

    # 주의: LangChain 템플릿에서 JSON 예시의 중괄호 {}는 {{ }}로 이스케이프 해야 함
    system_prompt = (
        "당신은 아동 심리 및 부모-자녀 상호작용 분석 전문가입니다.\n"
        "제공된 대화 로그를 분석하여 정의된 '상호작용 패턴'을 정밀하게 탐지하십시오.\n"
        "\n"
        "*** [중요] 데이터 특성 및 분석 지침 ***\n"
        "1. **화자 라벨의 불완전성**: 'Parent', 'Child' 라벨은 기술적 한계로 인해 틀릴 수 있습니다.\n"
        "   - 라벨에 맹신하지 말고, **발화의 내용, 말투, 문맥**을 보고 실제 화자가 누구인지, 어떤 의도인지 판단하십시오.\n"
        "2. **DPICS 라벨은 참고용**: 제공된 라벨(RD, PR 등)은 자동 분류된 것입니다.\n"
        "   - 라벨이 패턴과 달라도, **발화 내용이 패턴의 '탐지 단서(Context Clues)'와 일치하면** 패턴으로 인정하십시오.\n"
        "3. **맥락(Context) 최우선**: 단일 발화만 보지 말고, **[앞선 발화 -> 반응 발화]**의 흐름을 보십시오.\n"
        "   - 예: '반영적 경청'은 반드시 상대방의 말이 먼저 있고, 그에 대한 반응이어야 합니다.\n"
        "   - 예: '감정 무시'는 아이가 감정을 표현한 직후에 부모가 반응하는 맥락이어야 합니다.\n"
        "\n"
        "*** 대화 길이 제한 (중요) ***\n"
        "- 각 패턴마다 \"utterance_indices\"에는 **해당 패턴을 가장 잘 보여주는 핵심 발화만 최대 3~4개까지만** 포함하십시오.\n"
        "- 관련된 대화가 훨씬 길더라도, 문제가 되는 순간 전후의 **연속된 3~4개 발화**를 선택하여 인덱스로 반환하십시오.\n"
        "- 같은 패턴이라도 전체 에피소드를 모두 포함하지 말고, **핵심 장면이 되는 짧은 클립만** 선택하십시오.\n"
        "\n"
        f"{mode_instruction}\n"
        "\n"
        "*** 사용할 패턴 정의 ***\n"
        f"{pattern_details}\n"
        "\n"
        "해당되는 패턴이 없으면 빈 리스트를 반환하십시오."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "아래 발화 목록을 분석하고 패턴을 탐지해 주세요.\n"
            "각 줄의 맨 앞 숫자는 발화 번호입니다.\n\n"
            "{utterances_labeled}"
        )),
    ])


_PATTERN_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 상호작용 패턴 탐지 검증관입니다.\n"
            "LLM이 탐지한 패턴이 '오탐(False Positive)'인지, 의미상 적절한지 확인하고 교정해야 합니다.\n"
            "\n"
            "*** 검증 기준 ***\n"
            "1. **반영적 경청(Reflective Listening)**:\n"
            "   - 단순히 앵무새처럼 단어만 반복하는 것은 아닙니다. (감정이나 의도를 읽어야 함)\n"
            "   - 내 의견(훈계, 지시)이 섞여 있으면 안 됩니다.\n"
            "2. **구체적 칭찬(Labeled Praise)**:\n"
            "   - 단순 '잘했어'는 아닙니다. '무엇을' 잘했는지 언급되어야 합니다.\n"
            "3. **감정 무시(Emotion Dismissing)**:\n"
            "   - 아이가 감정을 표현했는지 먼저 확인하세요.\n"
            "   - 부모가 그 감정을 수용하지 않고 차단/전환했는지 확인하세요.\n"
            "4. **일반 원칙**:\n"
            "   - 화자 라벨이 틀렸더라도 발화 내용상 패턴이 맞다면 인정하세요.\n"
            "   - 패턴의 정의와 가장 '의미상으로 가까운' 것이어야 합니다.\n"
        ),
    ),
    (
        "human",
        (
            "탐지된 패턴 목록:\n{patterns}\n\n"
            "관련 발화:\n{utterances}\n\n"
            "각 패턴에 대해 검증 결과를 제공하세요."
        ),
    ),
])


# -----------------------
#  유틸: 에피소드 분할 (Sliding Window 적용)
# -----------------------

def _segment_utterances_into_episodes(
    utterances_labeled: List[Dict[str, Any]],
    window_size: int = 10,
    step: int = 5,
) -> List[List[int]]:
    """
    Sliding Window 방식으로 에피소드를 분할하여 문맥 절단을 방지함.
    - window_size: 한 에피소드의 발화 수
    - step: 다음 윈도우 시작 위치 (겹치는 구간 생성)
    """
    total_len = len(utterances_labeled)
    if total_len == 0:
        return []
    
    episodes: List[List[int]] = []
    
    # 발화가 window_size보다 적을 경우 전체를 하나로
    if total_len <= window_size:
        return [list(range(total_len))]

    for i in range(0, total_len, step):
        # 윈도우 생성
        end = min(i + window_size, total_len)
        indices = list(range(i, end))
        
        # 너무 짧은 자투리(예: 3개 미만)는 무시 (의미 있는 맥락 형성 어려움)
        if len(indices) >= 3: 
            episodes.append(indices)
        
        # 끝에 도달했으면 종료
        if end == total_len:
            break
            
    return episodes


# -----------------------
#  유틸: 패턴 검증
# -----------------------

def _validate_patterns_with_llm(
    patterns: List[Dict[str, Any]],
    utterances_labeled: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """LLM을 사용하여 탐지된 패턴들을 검증하고 잘못된 패턴 제거/수정"""
    if not patterns:
        return patterns

    try:
        structured_llm = get_structured_llm(PatternValidationList, mini=True)

        # 패턴 정보 포맷팅
        patterns_str = "\n".join([
            f"{i}. [{p.get('pattern_name')}] ({p.get('utterance_indices')}) : {p.get('description')}"
            for i, p in enumerate(patterns)
        ])

        # 관련 발화 추출
        all_indices: Set[int] = set()
        for p in patterns:
            all_indices.update(p.get("utterance_indices", []))

        relevant_utterances: List[str] = []
        for idx in sorted(all_indices):
            if 0 <= idx < len(utterances_labeled):
                utt = utterances_labeled[idx]
                text = utt.get("original_ko") or utt.get("korean") or utt.get("text") or ""
                # 라벨 정보도 함께 제공하여 검증 돕기
                relevant_utterances.append(
                    f"{idx}. [{utt.get('speaker')}] ({utt.get('label', 'N/A')}) {text}"
                )

        utterances_str = "\n".join(relevant_utterances) if relevant_utterances else "발화 없음"

        # LLM 검증
        res = (_PATTERN_VALIDATION_PROMPT | structured_llm).invoke({
            "patterns": patterns_str,
            "utterances": utterances_str
        })

        # Pydantic 모델에서 데이터 추출
        if isinstance(res, PatternValidationList):
            validations = [v.dict() for v in res.validations]
        else:
            return patterns

        # 검증 결과 적용
        validated_patterns: List[Dict[str, Any]] = []
        for i, pattern in enumerate(patterns):
            # 해당 패턴의 검증 결과 찾기
            validation = None
            for v in validations:
                if v.get("pattern_index") == i:
                    validation = v
                    break

            if validation is None:
                validated_patterns.append(pattern)
                continue

            is_valid = validation.get("is_valid", True)
            if is_valid:
                validated_patterns.append(pattern)
            else:
                corrected_name = validation.get("corrected_pattern_name")
                reason = validation.get("reason", "Invalid pattern")

                if corrected_name and corrected_name != "null":
                    # 다른 패턴으로 수정
                    pattern["pattern_name"] = corrected_name
                    pattern["description"] = f"[수정됨: {reason}] {pattern.get('description', '')}"
                    
                    # pattern_type 재설정 (긍정/부정 여부 확인)
                    pattern_type = "negative"
                    # 긍정 패턴 목록에 있는지 확인
                    for pos_p in _PATTERN_DEFINITIONS.get("positive_patterns", []):
                        if pos_p.get("name") == corrected_name:
                            pattern_type = "positive"
                            break
                    # 추가 패턴 목록에 있는지 확인
                    for add_p in _PATTERN_DEFINITIONS.get("additional_patterns", []):
                        if add_p.get("name") == corrected_name:
                            pattern_type = "positive"
                            break
                            
                    pattern["pattern_type"] = pattern_type
                    validated_patterns.append(pattern)
                # else: 패턴 제거 (필터링)

        return validated_patterns

    except Exception as e:
        print(f"Pattern validation error: {e}")
        return patterns


# -----------------------
#  유틸: 패턴 병합/중복 제거
# -----------------------

def _merge_overlapping_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sliding Window로 인해 중복 탐지된 패턴들을 병합
    """
    if not patterns:
        return []

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    merged: List[Dict[str, Any]] = []

    for p in patterns:
        name = p.get("pattern_name")
        indices = set(p.get("utterance_indices") or [])
        if not indices:
            continue

        merged_flag = False
        for m in merged:
            if m.get("pattern_name") != name:
                continue
            m_indices = set(m.get("utterance_indices") or [])
            
            # 교집합이 존재하면 병합 시도 (Sliding window로 인해 부분 겹침이 많음)
            if not indices.isdisjoint(m_indices):
                # 인덱스 합치기
                new_indices = sorted(indices | m_indices)
                m["utterance_indices"] = new_indices
                
                # 설명(description)이 더 길거나 구체적인 것 유지
                if len(p.get("description", "")) > len(m.get("description", "")):
                    m["description"] = p["description"]

                # severity는 더 높은 쪽
                s1 = m.get("severity", "medium")
                s2 = p.get("severity", "medium")
                if severity_rank.get(s2, 1) > severity_rank.get(s1, 1):
                    m["severity"] = s2

                merged_flag = True
                break

        if not merged_flag:
            merged.append(p)

    return merged


# -----------------------
#  유틸: LLM 호출 (에피소드 단위)
# -----------------------

def _run_pattern_llm_on_episode(
    utterances_labeled: List[Dict[str, Any]],
    episode_indices: List[int],
    mode: Literal["positive", "negative"],
) -> List[Dict[str, Any]]:
    """
    하나의 에피소드에 대해 LLM 패턴 탐지 실행
    """
    if not episode_indices:
        return []

    try:
        structured_llm = get_structured_llm(PatternDetectionList, mini=False)
        prompt = _get_pattern_prompt(mode)

        lines: List[str] = []
        for idx in episode_indices:
            if 0 <= idx < len(utterances_labeled):
                utt = utterances_labeled[idx]
                speaker = utt.get("speaker", "Unknown")
                label = utt.get("label", "N/A")
                text = utt.get("original_ko") or utt.get("korean") or utt.get("text") or ""
                lines.append(f"{idx}. [{speaker}] ({label}) {text}")

        utterances_str = "\n".join(lines)

        res = (prompt | structured_llm).invoke({"utterances_labeled": utterances_str})

        # Pydantic 모델에서 데이터 추출
        if isinstance(res, PatternDetectionList):
            llm_patterns = [p.dict() for p in res.patterns]
        else:
            return []

        # 기본 속성 보정
        for p in llm_patterns:
            p["pattern_type"] = mode if mode in ["positive", "negative"] else p.get("pattern_type", mode)
            if not p.get("severity"):
                p["severity"] = "medium" if mode == "negative" else "low"

        return llm_patterns

    except Exception as e:
        print(f"LLM pattern detection error ({mode}): {e}")
        return []


# -----------------------
#  메인 엔트리: detect_patterns_node
# -----------------------

def detect_patterns_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 패턴 탐지 노드
    1. 전체 대화에 대해 부정 패턴 -> 긍정 패턴 순차 탐지 (모드별 1회 호출)
    2. 중복 병합 및 검증
    """
    utterances_labeled = state.get("utterances_labeled") or []

    if not utterances_labeled:
        return {"patterns": []}

    # 1. 부정 패턴 탐지 (전체 대화 기준, 1회 호출)
    negative_patterns: List[Dict[str, Any]] = _run_pattern_llm_on_episode(
        utterances_labeled=utterances_labeled,
        episode_indices=list(range(len(utterances_labeled))),
        mode="negative",
    )
    # 부정 패턴 병합 (혹시 중복이 있더라도 정리)
    negative_patterns = _merge_overlapping_patterns(negative_patterns)
    
    # 2. 긍정 패턴 탐지 (전체 대화 기준, 1회 호출)
    positive_patterns: List[Dict[str, Any]] = _run_pattern_llm_on_episode(
        utterances_labeled=utterances_labeled,
        episode_indices=list(range(len(utterances_labeled))),
        mode="positive",
    )
    # 긍정 패턴 병합
    positive_patterns = _merge_overlapping_patterns(positive_patterns)

    # 4. 검증 (LLM Cross-Check)
    negative_patterns = _validate_patterns_with_llm(negative_patterns, utterances_labeled)
    positive_patterns = _validate_patterns_with_llm(positive_patterns, utterances_labeled)

    # 5. 최종 병합
    all_patterns = negative_patterns + positive_patterns

    return {"patterns": all_patterns}