from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm


class SpeakerMapping(BaseModel):
    """스피커 매핑"""
    A: str = Field(description="A가 부모인지 아이인지: 'MOM' 또는 'CHI'")
    B: str = Field(description="B가 부모인지 아이인지: 'MOM' 또는 'CHI'")


class CorrectionItem(BaseModel):
    """보정 항목"""
    index: int = Field(description="발화의 인덱스 (0부터 시작)")
    original_speaker: str = Field(description="원래 분류된 화자 (MOM 또는 CHI)")
    corrected_speaker: str = Field(description="올바른 화자 (MOM 또는 CHI)")


class SpeakerCorrection(BaseModel):
    """스피커 분리 보정"""
    needs_correction: bool = Field(description="A/B 분리가 잘못되었는지 여부")
    correction_reason: str = Field(description="보정이 필요한 이유 (보정이 필요없으면 빈 문자열)")
    corrections: List[CorrectionItem] = Field(
        description="보정이 필요한 경우, 각 발화의 인덱스와 올바른 화자 정보",
        default_factory=list
    )


class STTCorrectionItem(BaseModel):
    """STT 보정 항목"""
    index: int = Field(description="발화의 인덱스 (0부터 시작)")
    original_text: str = Field(description="원본 텍스트 (STT 오류가 있는 텍스트)")
    corrected_text: str = Field(description="보정된 텍스트 (문법적으로 올바르고 자연스러운 텍스트)")


class STTCorrection(BaseModel):
    """STT 텍스트 보정"""
    needs_correction: bool = Field(description="STT 오류가 있는지 여부")
    correction_reason: str = Field(description="보정이 필요한 이유 (보정이 필요없으면 빈 문자열)")
    corrections: List[STTCorrectionItem] = Field(
        description="보정이 필요한 경우, 각 발화의 인덱스와 보정된 텍스트",
        default_factory=list
    )


_SPEAKER_IDENTIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 대화를 분석하는 전문가입니다. "
            "대화 내용을 보고 A와 B 중 어느 것이 부모(엄마/아빠)이고 어느 것이 아이(자녀)인지 판단하세요.\n\n"
            "판단 기준:\n"
            "- 부모(엄마/아빠): 질문을 많이 하거나, 지시하거나, 설명하는 역할. 어른스러운 언어 사용. "
            "예: '제트레인이 뭐야?', '왜 엄마가 만들고 있는 거 뺏어가', '우리 같이 이렇게 블록 놀이 하면 좋잖아'\n"
            "- 아이(자녀): 질문에 답하거나, 자신의 감정이나 생각을 표현. 아이다운 언어 사용. "
            "예: '제피에는 아주 오래전에 살았던', '아니 나 혼자 만들고 싶어', '같이 하면은 재미가 없어'\n\n"
            "대화의 톤, 언어 사용, 역할, 대화 흐름을 종합적으로 고려하여 정확히 판단하세요. "
            "질문을 하는 쪽이 부모일 가능성이 높고, 답변하거나 감정을 표현하는 쪽이 아이일 가능성이 높습니다."
        ),
    ),
    (
        "human",
        (
            "다음 대화를 분석하여 A와 B 중 어느 것이 부모이고 어느 것이 아이인지 판단해주세요:\n\n"
            "{dialogue}\n\n"
            "주의사항:\n"
            "- 각 발화의 내용, 톤, 역할을 신중히 분석하세요\n"
            "- 질문을 하는 쪽이 부모일 가능성이 높습니다\n"
            "- 답변하거나 감정을 표현하는 쪽이 아이일 가능성이 높습니다\n"
            "- 대화의 전체적인 흐름을 고려하세요\n\n"
            "A와 B 각각이 'MOM'(부모)인지 'CHI'(아이)인지 정확히 판단해주세요."
        ),
    ),
])


_STT_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 STT(Speech-to-Text)로 생성된 텍스트의 오류를 보정하는 전문가입니다. "
            "STT 텍스트는 발음이 부정확하거나 잘 안들린 경우 문장이나 단어가 이상하게 변환될 수 있습니다.\n\n"
            "일반적인 STT 오류 유형:\n"
            "- 비슷한 발음의 단어로 잘못 인식 (예: '샤프' -> '샤푸')\n"
            "- 문장이 중간에 끊기거나 단어가 합쳐짐\n"
            "- 문맥상 맞지 않는 단어나 표현\n"
            "- 문법적으로 어색한 문장 구조\n"
            "- 발음이 비슷한 다른 단어로 대체됨\n\n"
            "보정 원칙:\n"
            "- 원본 텍스트의 의미와 의도를 최대한 보존\n"
            "- 문법적으로 올바르고 자연스러운 한국어로 보정\n"
            "- 부모-자녀 대화의 맥락을 고려하여 보정\n"
            "- 확실하게 오류라고 판단되는 경우만 보정 (애매한 경우는 보정하지 않음)"
        ),
    ),
        (
            "human",
            (
                "다음 STT로 생성된 대화 텍스트를 분석하여 오류가 있는 발화들을 보정해주세요:\n\n"
                "{stt_dialogue}\n\n"
                "주의사항:\n"
                "- 각 발화의 인덱스는 0부터 시작합니다\n"
                "- 발화의 내용을 신중히 분석하여 STT 오류를 찾으세요\n"
                "- 확실하게 오류라고 판단되는 경우만 보정하세요 (애매한 경우는 보정하지 마세요)\n"
                "- 보정이 필요없으면 needs_correction을 false로 설정하세요\n"
                "- 보정이 필요한 경우, corrections 배열에 각 발화의 인덱스, 원본 텍스트, 보정된 텍스트를 명시하세요\n"
                "- 보정할 때는 반드시 정확한 인덱스를 사용하세요\n\n"
                "분석 결과를 제공해주세요."
            ),
        ),
])


_SPEAKER_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 부모-자녀 대화의 화자 분리를 검증하고 보정하는 전문가입니다. "
            "정규화된 대화를 보고 각 발화가 올바른 화자(MOM 또는 CHI)로 분류되었는지 확인하세요.\n\n"
            "부모(MOM) 발화의 특징:\n"
            "- 질문을 많이 하거나, 지시하거나, 설명하는 역할\n"
            "- 예: '제트레인이 뭐야?', '왜 엄마가 만들고 있는 거 뺏어가', '우리 같이 이렇게 블록 놀이 하면 좋잖아'\n\n"
            "아이(CHI) 발화의 특징:\n"
            "- 질문에 답하거나, 자신의 감정이나 생각을 표현\n"
            "- 아이다운 언어 사용\n"
            "- 예: '제피에는 아주 오래전에 살았던', '아니 나 혼자 만들고 싶어', '같이 하면은 재미가 없어'\n\n"
            "잘못된 분류의 예시:\n"
            "- 부모 발화인데 CHI로 분류된 경우 (예: '왜 엄마가 만들고 있는 거 뺏어가'가 CHI로 분류됨)\n"
            "- 아이 발화인데 MOM으로 분류된 경우 (예: '아니 나 혼자 만들고 싶어'가 MOM으로 분류됨)\n\n"
            "주의사항:\n"
            "- 거의 확실하게 잘못된 것들만 보정하세요\n"
            "- 애매한 경우는 보정하지 마세요\n"
            "- 발화의 내용, 톤, 역할을 종합적으로 고려하세요"
        ),
    ),
    (
        "human",
        (
            "다음 정규화된 대화를 분석하여 각 발화가 올바른 화자로 분류되었는지 확인하고, "
            "거의 확실하게 잘못 분류된 발화들을 올바른 화자로 보정해주세요:\n\n"
            "{normalized_dialogue}\n\n"
            "주의사항:\n"
            "- 각 발화의 인덱스는 0부터 시작합니다\n"
            "- 발화의 내용, 톤, 역할을 신중히 분석하세요\n"
            "- 부모 발화인데 CHI로 분류되었거나, 아이 발화인데 MOM으로 분류된 것들을 찾으세요\n"
            "- 거의 확실하게 잘못된 것들만 보정하세요 (애매한 경우는 보정하지 마세요)\n"
            "- 보정이 필요없으면 needs_correction을 false로 설정하세요\n"
            "- 보정이 필요한 경우, corrections 배열에 각 발화의 정보를 명시하세요:\n"
            "  * index: 발화의 인덱스 (0부터 시작)\n"
            "  * original_speaker: 현재 분류된 화자 (MOM 또는 CHI)\n"
            "  * corrected_speaker: 올바른 화자 (MOM 또는 CHI)\n"
            "- 보정할 때는 반드시 정확한 인덱스를 사용하세요. 인덱스는 대화에서 표시된 [숫자]와 정확히 일치해야 합니다.\n\n"
            "분석 결과를 제공해주세요."
        ),
    ),
])


# 스피커 라벨 상수 정의
_PARENT_LABELS = {"PARENT", "MOM", "MOTHER", "FATHER", "DAD", "부모", "엄마", "아빠", "어머니", "아버지"}
_CHILD_LABELS = {"CHILD", "CHI", "KID", "SON", "DAUGHTER", "아이", "자녀", "아들", "딸"}
_PARENT_LABELS_LOWER = {label.lower() for label in _PARENT_LABELS} | {"parent", "mom", "dad", "mother", "father"}
_CHILD_LABELS_LOWER = {label.lower() for label in _CHILD_LABELS} | {"child", "kid", "son", "daughter"}


def _normalize_speaker_label(speaker_label: str, ab_mapping: Dict[str, str]) -> Optional[str]:
    """스피커 라벨을 'MOM' 또는 'CHI'로 정규화"""
    speaker_label = speaker_label.strip().upper()
    
    # A/B 패턴 처리
    if speaker_label in ["A", "B"]:
        return ab_mapping.get(speaker_label, ("MOM" if speaker_label == "A" else "CHI"))
    
    # 부모 패턴
    if speaker_label in _PARENT_LABELS:
        return "MOM"
    
    # 아이 패턴
    if speaker_label in _CHILD_LABELS:
        return "CHI"
    
    return None


def _normalize_speaker_label_from_string(speaker_label: str, ab_mapping: Dict[str, str]) -> Optional[str]:
    """문자열에서 추출한 스피커 라벨을 정규화 (대소문자 구분 없음)"""
    speaker_label_lower = speaker_label.strip().lower()
    
    # A/B 패턴 처리
    if speaker_label_lower in ["a", "b"]:
        ab_upper = speaker_label_lower.upper()
        return ab_mapping.get(ab_upper, ("MOM" if ab_upper == "A" else "CHI"))
    
    # 부모 패턴
    if speaker_label_lower in _PARENT_LABELS_LOWER:
        return "MOM"
    
    # 아이 패턴
    if speaker_label_lower in _CHILD_LABELS_LOWER:
        return "CHI"
    
    return None


def _identify_speakers_with_llm(utterances_ko: List[Dict[str, Any]]) -> Dict[str, str]:
    """LLM을 사용하여 A/B가 부모인지 아이인지 판단"""
    try:
        # 대화 내용 포맷팅 (speaker와 text만 사용)
        dialogue_lines = []
        for utt in utterances_ko:
            if isinstance(utt, dict):
                speaker = utt.get("speaker", "")
                text = utt.get("text", "")
                if speaker and text:
                    dialogue_lines.append(f"{speaker}: {text}")
        
        if not dialogue_lines:
            print("Warning: No dialogue lines found for speaker identification")
            return {"A": "MOM", "B": "CHI"}
        
        dialogue_str = "\n".join(dialogue_lines)
        
        # 디버깅: 대화 내용 출력
        print(f"DEBUG: Identifying speakers from dialogue:\n{dialogue_str[:200]}...")
        
        # Structured LLM 사용
        structured_llm = get_structured_llm(SpeakerMapping, mini=True)
        res = (_SPEAKER_IDENTIFY_PROMPT | structured_llm).invoke({"dialogue": dialogue_str})
        
        # Pydantic 모델에서 데이터 추출
        if isinstance(res, SpeakerMapping):
            result = {}
            # A와 B의 값을 정규화
            a_value = res.A.upper() if res.A else "MOM"
            b_value = res.B.upper() if res.B else "CHI"
            
            result["A"] = "MOM" if a_value == "MOM" else "CHI"
            result["B"] = "MOM" if b_value == "MOM" else "CHI"
            
            # 디버깅: 매핑 결과 출력
            print(f"DEBUG: Speaker mapping result: A={result['A']}, B={result['B']}")
            
            return result
        else:
            print(f"Warning: Unexpected response type from LLM: {type(res)}")
    except Exception as e:
        print(f"LLM speaker identification error: {e}")
        import traceback
        traceback.print_exc()
    
    # 폴백: A는 부모, B는 아이
    print("Warning: Using fallback mapping: A=MOM, B=CHI")
    return {"A": "MOM", "B": "CHI"}


def _detect_patterns(utterances_ko: List[Any], is_object_list: bool) -> Tuple[bool, bool]:
    """A/B 패턴과 명시적 스피커 패턴 감지"""
    has_ab_pattern = False
    has_explicit_speakers = False
    
    for utt in utterances_ko:
        if is_object_list and isinstance(utt, dict):
            speaker = str(utt.get("speaker", "")).strip().upper()
            if speaker in ["A", "B"]:
                has_ab_pattern = True
            elif speaker in _PARENT_LABELS | _CHILD_LABELS:
                has_explicit_speakers = True
                break
        else:
            # 문자열 리스트인 경우
            utt_str = str(utt).strip()
            if re.match(r'^([AB])[:\s]+', utt_str, flags=re.IGNORECASE):
                has_ab_pattern = True
            elif re.match(r'^\[?(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father|아이|자녀|아들|딸|Child|Kid|Son|Daughter)\]?', 
                         utt_str, flags=re.IGNORECASE):
                has_explicit_speakers = True
                break
    
    return has_ab_pattern, has_explicit_speakers


def _parse_speaker_from_string(utt_str: str, ab_mapping: Dict[str, str]) -> Tuple[Optional[str], str]:
    """문자열에서 스피커와 발화 내용 추출"""
    utt_str = utt_str.strip()
    speaker = None
    발화내용_ko = utt_str
    
    # 대괄호 패턴 매칭 (우선 처리)
    bracket_match = re.match(
        r'^\[(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father|아이|자녀|아들|딸|Child|Kid|Son|Daughter)\]\s*(.+)$',
        utt_str, flags=re.IGNORECASE
    )
    if bracket_match:
        speaker_label = bracket_match.group(1)
        발화내용_ko = bracket_match.group(2).strip()
        speaker = _normalize_speaker_label_from_string(speaker_label, ab_mapping)
    
    # A/B 알파벳 패턴 매칭
    if speaker is None:
        ab_match = re.match(r'^([AB])[:\s]+(.+)$', utt_str, flags=re.IGNORECASE)
        if ab_match:
            ab_label = ab_match.group(1).upper()
            발화내용_ko = ab_match.group(2).strip()
            speaker = ab_mapping.get(ab_label, ("MOM" if ab_label == "A" else "CHI"))
    
    # 부모/아이 패턴 매칭
    if speaker is None:
        parent_match = re.match(
            r'^(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father)[:\s]+(.+)$',
            utt_str, flags=re.IGNORECASE
        )
        if parent_match:
            speaker = "MOM"
            발화내용_ko = parent_match.group(2).strip()
        else:
            child_match = re.match(
                r'^(아이|자녀|아들|딸|Child|Kid|Son|Daughter)[:\s]+(.+)$',
                utt_str, flags=re.IGNORECASE
            )
            if child_match:
                speaker = "CHI"
                발화내용_ko = child_match.group(2).strip()
    
    # A/B prefix 제거 (스피커를 찾지 못한 경우)
    if speaker is None:
        발화내용_ko = re.sub(r'^([AB])[:\s]+', '', 발화내용_ko, flags=re.IGNORECASE).strip()
    
    return speaker, 발화내용_ko


def _normalize_utterance_object(utt: Dict[str, Any], ab_mapping: Dict[str, str], 
                                last_speaker: Optional[str]) -> Optional[Dict[str, Any]]:
    """객체 형태의 발화를 정규화"""
    speaker_label = str(utt.get("speaker", "")).strip().upper()
    발화내용_ko = str(utt.get("text", "")).strip()
    
    if not 발화내용_ko:
        return None
    
    # 원본 A/B 라벨 저장 (검증을 위해)
    original_ab_label = speaker_label if speaker_label in ["A", "B"] else None
    
    speaker = _normalize_speaker_label(speaker_label, ab_mapping)
    
    # 스피커를 알 수 없으면 이전 스피커 추론
    if speaker is None:
        if last_speaker:
            speaker = "CHI" if last_speaker == "MOM" else "MOM"
        else:
            speaker = "MOM"  # 기본값
    
    # timestamp 추출
    timestamp = utt.get("timestamp") or utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
    
    result = {
        "speaker": speaker,
        "발화내용_ko": 발화내용_ko,
        "timestamp": timestamp
    }
    
    # 원본 A/B 라벨이 있으면 저장
    if original_ab_label:
        result["original_ab_label"] = original_ab_label
    
    return result


def _normalize_utterance_string(utt: Any, ab_mapping: Dict[str, str], 
                                last_speaker: Optional[str]) -> Optional[Dict[str, Any]]:
    """문자열 형태의 발화를 정규화"""
    utt_str = str(utt).strip()
    if not utt_str:
        return None
    
    # 원본 A/B 라벨 추출 (검증을 위해)
    original_ab_label = None
    ab_match = re.match(r'^([AB])[:\s]+', utt_str, flags=re.IGNORECASE)
    if ab_match:
        original_ab_label = ab_match.group(1).upper()
    
    speaker, 발화내용_ko = _parse_speaker_from_string(utt_str, ab_mapping)
    
    # 스피커를 알 수 없으면 이전 스피커 추론
    if speaker is None:
        if last_speaker:
            speaker = "CHI" if last_speaker == "MOM" else "MOM"
        else:
            speaker = "MOM"  # 기본값
    
    # 최종적으로 A/B prefix가 남아있는지 확인하고 제거
    발화내용_ko = re.sub(r'^([AB])[:\s]+', '', 발화내용_ko, flags=re.IGNORECASE).strip()
    
    result = {
        "speaker": speaker,
        "발화내용_ko": 발화내용_ko,
        "timestamp": None
    }
    
    # 원본 A/B 라벨이 있으면 저장
    if original_ab_label:
        result["original_ab_label"] = original_ab_label
    
    return result


def _verify_and_correct_speakers(normalized: List[Dict[str, Any]], ab_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """정규화된 발화들을 검증하고 잘못 분류된 경우 보정 (발화 내용 기반)"""
    if not normalized or len(normalized) < 2:
        return normalized
    
    print(f"DEBUG: [Speaker Correction] Starting verification for {len(normalized)} utterances")
    
    # 부모와 아이 발화 비율 확인
    mom_count = sum(1 for utt in normalized if utt.get("speaker") == "MOM")
    chi_count = sum(1 for utt in normalized if utt.get("speaker") == "CHI")
    total_count = len(normalized)
    
    print(f"DEBUG: [Speaker Correction] Current distribution: MOM={mom_count}, CHI={chi_count}, Total={total_count}")
    
    # 한쪽 화자만 있거나 극단적으로 불균형한 경우 검증하지 않음
    if mom_count == 0 or chi_count == 0:
        print(f"DEBUG: [Speaker Correction] Skipping verification: only one speaker type detected")
        return normalized
    
    # 극단적으로 불균형한 경우 (95% 이상이 한쪽 화자) 검증하지 않음
    if mom_count / total_count > 0.95 or chi_count / total_count > 0.95:
        print(f"DEBUG: [Speaker Correction] Skipping verification: extremely imbalanced distribution")
        return normalized
    
    try:
        # 대화 내용 포맷팅 (현재 분류된 화자와 발화 내용만 포함)
        # 디버깅을 위해 인덱스와 함께 발화 앞글자 5자도 포함
        dialogue_lines = []
        for idx, utt in enumerate(normalized):
            current_speaker = utt.get("speaker", "UNKNOWN")
            text = utt.get("발화내용_ko", "")
            if text:
                # 앞글자 5자 추출 (공백 제거 후)
                text_preview = text.strip()[:5] if text.strip() else ""
                dialogue_lines.append(f"[{idx}] {current_speaker}: {text} (앞글자5자: '{text_preview}')")
        
        if not dialogue_lines:
            return normalized
        
        dialogue_str = "\n".join(dialogue_lines)
        print(f"DEBUG: [Speaker Correction] Sending {len(dialogue_lines)} utterances to LLM for verification")
        
        # Structured LLM 사용하여 검증
        structured_llm = get_structured_llm(SpeakerCorrection, mini=True)
        res = (_SPEAKER_CORRECTION_PROMPT | structured_llm).invoke({"normalized_dialogue": dialogue_str})
        
        use_speaker_correction = True
        # 보정 적용
        if isinstance(res, SpeakerCorrection) and res.needs_correction and use_speaker_correction:
            print(f"DEBUG: [Speaker Correction] ===== CORRECTION NEEDED =====")
            print(f"DEBUG: [Speaker Correction] Reason: {res.correction_reason}")
            print(f"DEBUG: [Speaker Correction] Number of corrections: {len(res.corrections)}")
            print(f"DEBUG: [Speaker Correction] Corrections detail: {res.corrections}")
            
            # 보정 사전 생성 (인덱스 -> 올바른 화자)
            correction_map = {}
            for correction in res.corrections:
                idx = correction.index
                original_speaker_from_llm = correction.original_speaker.upper()
                corrected_speaker = correction.corrected_speaker.upper()
                
                # 인덱스 범위 확인
                if idx < 0 or idx >= len(normalized):
                    utt_text_preview = ""
                    if idx >= 0 and idx < len(normalized):
                        utt_text_preview = normalized[idx].get("발화내용_ko", "").strip()[:5]
                    print(f"DEBUG: [Speaker Correction] WARNING: Invalid index {idx} (out of range, total={len(normalized)})")
                    continue
                
                # 실제 발화의 현재 화자 확인
                actual_speaker = normalized[idx].get("speaker", "").upper()
                utt_text_preview = normalized[idx].get("발화내용_ko", "").strip()[:5]
                
                # LLM이 보고한 original_speaker와 실제 발화의 speaker가 일치하는지 확인
                if original_speaker_from_llm != actual_speaker:
                    print(f"DEBUG: [Speaker Correction] WARNING: Index {idx} - LLM reported original_speaker={original_speaker_from_llm}, but actual speaker={actual_speaker}. Text preview: '{utt_text_preview}'. Skipping.")
                    continue
                
                if corrected_speaker in ["MOM", "CHI"]:
                    correction_map[idx] = corrected_speaker
                    print(f"DEBUG: [Speaker Correction] Valid correction: Index {idx}, {original_speaker_from_llm} -> {corrected_speaker}, Text preview: '{utt_text_preview}'")
                else:
                    print(f"DEBUG: [Speaker Correction] WARNING: Invalid corrected_speaker: {corrected_speaker}")
            
            # 보정 적용
            if correction_map:
                print(f"DEBUG: [Speaker Correction] Applying {len(correction_map)} corrections:")
                corrected = []
                correction_count = 0
                for idx, utt in enumerate(normalized):
                    if idx in correction_map:
                        original_speaker = utt.get("speaker")
                        corrected_speaker = correction_map[idx]
                        corrected_utt = utt.copy()
                        corrected_utt["speaker"] = corrected_speaker
                        corrected.append(corrected_utt)
                        correction_count += 1
                        print(f"DEBUG: [Speaker Correction]   [{idx}] {original_speaker} -> {corrected_speaker}")
                        print(f"DEBUG: [Speaker Correction]      Text: {utt.get('발화내용_ko', '')[:100]}")
                    else:
                        corrected.append(utt)
                
                # 보정 후 통계
                new_mom_count = sum(1 for utt in corrected if utt.get("speaker") == "MOM")
                new_chi_count = sum(1 for utt in corrected if utt.get("speaker") == "CHI")
                print(f"DEBUG: [Speaker Correction] ===== CORRECTION COMPLETE =====")
                print(f"DEBUG: [Speaker Correction] Total corrections applied: {correction_count}")
                print(f"DEBUG: [Speaker Correction] New distribution: MOM={new_mom_count}, CHI={new_chi_count}")
                return corrected
            else:
                print(f"DEBUG: [Speaker Correction] WARNING: No valid corrections to apply")
        else:
            print(f"DEBUG: [Speaker Correction] Verification complete - no correction needed")
    except Exception as e:
        print(f"DEBUG: [Speaker Correction] ERROR: LLM speaker correction error: {e}")
        import traceback
        traceback.print_exc()
        # 오류 발생 시 원본 반환
        return normalized
    
    return normalized


def _correct_stt_errors(utterances_ko: List[Any], is_object_list: bool) -> List[Any]:
    """STT 텍스트 오류를 보정"""
    if not utterances_ko:
        return utterances_ko
    
    try:
        # 대화 내용 포맷팅
        dialogue_lines = []
        for idx, utt in enumerate(utterances_ko):
            if is_object_list and isinstance(utt, dict):
                text = str(utt.get("text", "")).strip()
                speaker = str(utt.get("speaker", "")).strip()
                if text:
                    dialogue_lines.append(f"[{idx}] {speaker}: {text}")
            else:
                text = str(utt).strip()
                if text:
                    dialogue_lines.append(f"[{idx}]: {text}")
        
        if not dialogue_lines:
            return utterances_ko
        
        dialogue_str = "\n".join(dialogue_lines)
        
        print(f"DEBUG: [STT Correction] Sending {len(dialogue_lines)} utterances to LLM for STT error correction")
        
        # Structured LLM 사용하여 STT 오류 보정
        structured_llm = get_structured_llm(STTCorrection, mini=True)
        res = (_STT_CORRECTION_PROMPT | structured_llm).invoke({
            "stt_dialogue": dialogue_str
        })
        
        # 보정 적용
        if isinstance(res, STTCorrection) and res.needs_correction:
            print(f"DEBUG: [STT Correction] ===== STT CORRECTION NEEDED =====")
            print(f"DEBUG: [STT Correction] Reason: {res.correction_reason}")
            print(f"DEBUG: [STT Correction] Number of corrections: {len(res.corrections)}")
            
            # 보정 사전 생성 (인덱스 -> 보정된 텍스트)
            correction_map = {}
            for correction in res.corrections:
                idx = correction.index
                
                # 인덱스 범위 확인
                if idx < 0 or idx >= len(utterances_ko):
                    print(f"DEBUG: [STT Correction] WARNING: Invalid index {idx} (out of range, total={len(utterances_ko)})")
                    continue
                
                # 원본 텍스트 확인
                if is_object_list and isinstance(utterances_ko[idx], dict):
                    original_text = str(utterances_ko[idx].get("text", "")).strip()
                else:
                    original_text = str(utterances_ko[idx]).strip()
                
                # LLM이 보고한 original_text와 실제 텍스트가 일치하는지 확인
                if correction.original_text.strip() != original_text:
                    print(f"DEBUG: [STT Correction] WARNING: Index {idx} - LLM reported original_text='{correction.original_text[:50]}', but actual text='{original_text[:50]}'. Skipping.")
                    continue
                
                correction_map[idx] = correction.corrected_text
                print(f"DEBUG: [STT Correction] Valid correction: Index {idx}")
                print(f"DEBUG: [STT Correction]   Original: {correction.original_text[:100]}")
                print(f"DEBUG: [STT Correction]   Corrected: {correction.corrected_text[:100]}")
            
            # 보정 적용
            if correction_map:
                corrected = []
                correction_count = 0
                for idx, utt in enumerate(utterances_ko):
                    if idx in correction_map:
                        if is_object_list and isinstance(utt, dict):
                            corrected_utt = utt.copy()
                            corrected_utt["text"] = correction_map[idx]
                            corrected.append(corrected_utt)
                        else:
                            corrected.append(correction_map[idx])
                        correction_count += 1
                    else:
                        corrected.append(utt)
                
                print(f"DEBUG: [STT Correction] ===== STT CORRECTION COMPLETE =====")
                print(f"DEBUG: [STT Correction] Total corrections applied: {correction_count}")
                return corrected
            else:
                print(f"DEBUG: [STT Correction] WARNING: No valid corrections to apply")
        else:
            print(f"DEBUG: [STT Correction] Verification complete - no STT errors found")
    except Exception as e:
        print(f"DEBUG: [STT Correction] ERROR: LLM STT correction error: {e}")
        import traceback
        traceback.print_exc()
        # 오류 발생 시 원본 반환
        return utterances_ko
    
    return utterances_ko


def preprocess_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ① preprocess: 스피커 정규화
    utterances_ko를 받아서 스피커를 정규화한 utterances_normalized 반환
    반환 형식: [{speaker: "MOM" | "CHI", 발화내용_ko: str}, ...]
    """
    utterances_ko = state.get("utterances_ko") or []
    
    if not utterances_ko:
        # 기존 message/dialogue에서 파싱 시도
        dialogue = state.get("message") or state.get("dialogue") or ""
        if dialogue:
            utterances_ko = [ln.strip() for ln in str(dialogue).splitlines() if ln.strip()]
    
    if not utterances_ko:
        return {"utterances_normalized": []}
    
    # utterances_ko가 객체 리스트인지 문자열 리스트인지 확인
    is_object_list = bool(utterances_ko and isinstance(utterances_ko[0], dict))
    
    # STT 텍스트 오류 보정 (정규화 전에 수행)
    print(f"DEBUG: [STT Correction] Starting STT error correction for {len(utterances_ko)} utterances")
    utterances_ko = _correct_stt_errors(utterances_ko, is_object_list)
    
    # A/B 패턴과 명시적 스피커 패턴 감지
    has_ab_pattern, has_explicit_speakers = _detect_patterns(utterances_ko, is_object_list)
    
    # A/B 패턴이 있고, 명시적 스피커가 없을 때만 LLM으로 스피커 식별
    ab_mapping = {}
    if has_ab_pattern and not has_explicit_speakers:
        if is_object_list:
            ab_mapping = _identify_speakers_with_llm(utterances_ko)
        else:
            # 문자열 리스트를 객체 리스트로 변환
            obj_list = []
            for utt in utterances_ko:
                utt_str = str(utt).strip()
                ab_match = re.match(r'^([AB])[:\s]+(.+)$', utt_str, flags=re.IGNORECASE)
                if ab_match:
                    obj_list.append({
                        "speaker": ab_match.group(1).upper(),
                        "text": ab_match.group(2).strip()
                    })
            if obj_list:
                ab_mapping = _identify_speakers_with_llm(obj_list)
    
    # 발화 정규화
    normalized: List[Dict[str, Any]] = []
    last_speaker: Optional[str] = None
    
    for utt in utterances_ko:
        if is_object_list and isinstance(utt, dict):
            normalized_utt = _normalize_utterance_object(utt, ab_mapping, last_speaker)
        else:
            normalized_utt = _normalize_utterance_string(utt, ab_mapping, last_speaker)
        
        if normalized_utt:
            normalized.append(normalized_utt)
            last_speaker = normalized_utt["speaker"]
    
    # A/B 패턴이 있었던 경우에만 검증 및 보정 수행
    # (발화 내용을 분석하여 잘못 분류된 것들을 보정)
    if has_ab_pattern and not has_explicit_speakers and normalized:
        normalized = _verify_and_correct_speakers(normalized, ab_mapping)
    
    # 검증 후 original_ab_label 필드 제거 (최종 결과에는 불필요)
    for utt in normalized:
        utt.pop("original_ab_label", None)
    
    return {"utterances_normalized": normalized}

