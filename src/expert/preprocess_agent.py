from __future__ import annotations

import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm


class SpeakerMapping(BaseModel):
    """스피커 매핑"""
    A: str = Field(description="A가 부모인지 아이인지: 'MOM' 또는 'CHI'")
    B: str = Field(description="B가 부모인지 아이인지: 'MOM' 또는 'CHI'")


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
    is_object_list = False
    if utterances_ko and isinstance(utterances_ko[0], dict):
        is_object_list = True
    
    # A/B 패턴이 있는지 먼저 확인 (객체 리스트인 경우)
    # parent/child가 이미 명시되어 있는지도 확인
    has_ab_pattern = False
    has_explicit_speakers = False
    if is_object_list:
        for utt in utterances_ko:
            if isinstance(utt, dict):
                speaker = str(utt.get("speaker", "")).strip().upper()
                if speaker in ["A", "B"]:
                    has_ab_pattern = True
                elif speaker in ["PARENT", "CHILD", "MOM", "CHI", "부모", "아이", "엄마", "아빠"]:
                    has_explicit_speakers = True
                    break
    else:
        # 문자열 리스트인 경우 (하위 호환성)
        for utt in utterances_ko:
            utt_str = str(utt).strip()
            ab_match = re.match(r'^([AB])[:\s]+', utt_str, flags=re.IGNORECASE)
            if ab_match:
                has_ab_pattern = True
                break
    
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
                    obj_list.append({"speaker": ab_match.group(1).upper(), "text": ab_match.group(2).strip()})
            if obj_list:
                ab_mapping = _identify_speakers_with_llm(obj_list)
    
    normalized: List[Dict[str, str]] = []
    
    for utt in utterances_ko:
        # 객체 리스트인 경우
        if is_object_list and isinstance(utt, dict):
            speaker_label = str(utt.get("speaker", "")).strip().upper()
            발화내용_ko = str(utt.get("text", "")).strip()
            
            if not 발화내용_ko:
                continue
            
            speaker = None
            
            # speaker가 "A" 또는 "B"인 경우
            if speaker_label in ["A", "B"]:
                # LLM으로 판단한 매핑 사용
                if speaker_label in ab_mapping:
                    speaker = ab_mapping[speaker_label]
                else:
                    # 매핑이 없으면 기본값 (A는 부모, B는 아이)
                    speaker = "MOM" if speaker_label == 'A' else "CHI"
            # speaker가 "parent" 또는 "child"인 경우
            elif speaker_label in ["PARENT", "MOM", "MOTHER", "FATHER", "DAD"]:
                speaker = "MOM"
            elif speaker_label in ["CHILD", "CHI", "KID", "SON", "DAUGHTER"]:
                speaker = "CHI"
            # 한국어 패턴
            elif speaker_label in ["부모", "엄마", "아빠", "어머니", "아버지"]:
                speaker = "MOM"
            elif speaker_label in ["아이", "자녀", "아들", "딸"]:
                speaker = "CHI"
            else:
                # 스피커를 알 수 없으면 이전 스피커 추론
                if normalized:
                    last_speaker = normalized[-1].get("speaker")
                    speaker = "CHI" if last_speaker == "MOM" else "MOM"
                else:
                    speaker = "MOM"  # 기본값
            
            # timestamp 추출 (여러 가능한 필드명 확인)
            timestamp = utt.get("timestamp") or utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
            
            normalized.append({
                "speaker": speaker,
                "발화내용_ko": 발화내용_ko,
                "timestamp": timestamp  # timestamp 포함
            })
        
        # 문자열 리스트인 경우 (하위 호환성)
        else:
            utt_str = str(utt).strip()
            if not utt_str:
                continue
            
            # 스피커 추출 및 정규화
            speaker = None
            발화내용_ko = utt_str
            
            # 대괄호 패턴 매칭 (우선 처리)
            bracket_match = re.match(r'^\[(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father|아이|자녀|아들|딸|Child|Kid|Son|Daughter)\]\s*(.+)$', 
                                    utt_str, flags=re.IGNORECASE)
            if bracket_match:
                speaker_label = bracket_match.group(1).lower()
                발화내용_ko = bracket_match.group(2).strip()
                # 부모 패턴
                if speaker_label in ['부모', '엄마', '아빠', '어머니', '아버지', 'parent', 'mom', 'dad', 'mother', 'father']:
                    speaker = "MOM"
                # 아이 패턴
                elif speaker_label in ['아이', '자녀', '아들', '딸', 'child', 'kid', 'son', 'daughter']:
                    speaker = "CHI"
            
            # 대괄호 패턴이 없으면 기존 패턴 매칭
            if speaker is None:
                # A/B 알파벳 패턴 매칭 (LLM으로 판단한 결과 사용)
                ab_match = re.match(r'^([AB])[:\s]+(.+)$', utt_str, flags=re.IGNORECASE)
                if ab_match:
                    ab_label = ab_match.group(1).upper()
                    발화내용_ko = ab_match.group(2).strip()  # A: 또는 B: prefix 제거
                    # LLM으로 판단한 매핑 사용
                    if ab_label in ab_mapping:
                        speaker = ab_mapping[ab_label]
                    else:
                        # 매핑이 없으면 기본값 (A는 부모, B는 아이)
                        speaker = "MOM" if ab_label == 'A' else "CHI"
                
                # A/B 패턴이 없으면 부모 패턴 매칭
                if speaker is None:
                    parent_match = re.match(r'^(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father)[:\s]+(.+)$', 
                                           utt_str, flags=re.IGNORECASE)
                    if parent_match:
                        speaker = "MOM"
                        발화내용_ko = parent_match.group(2).strip()
                    else:
                        # 아이 패턴 매칭
                        child_match = re.match(r'^(아이|자녀|아들|딸|Child|Kid|Son|Daughter)[:\s]+(.+)$', 
                                              utt_str, flags=re.IGNORECASE)
                        if child_match:
                            speaker = "CHI"
                            발화내용_ko = child_match.group(2).strip()
            
            # 스피커가 없으면 이전 스피커 추론 (간단한 휴리스틱)
            if speaker is None:
                # 발화내용에서 A/B prefix 제거
                발화내용_ko = re.sub(r'^([AB])[:\s]+', '', 발화내용_ko, flags=re.IGNORECASE).strip()
                
                if normalized:
                    last_speaker = normalized[-1].get("speaker")
                    # 이전 발화와 다른 스피커로 가정 (대화는 주로 교대로 진행)
                    if last_speaker == "MOM":
                        speaker = "CHI"
                    elif last_speaker == "CHI":
                        speaker = "MOM"
                    else:
                        speaker = "MOM"  # 기본값
                else:
                    speaker = "MOM"  # 첫 발화는 기본적으로 부모
            
            # 최종적으로 A/B prefix가 남아있는지 확인하고 제거 (모든 경우에 대해)
            if 발화내용_ko:
                발화내용_ko = re.sub(r'^([AB])[:\s]+', '', 발화내용_ko, flags=re.IGNORECASE).strip()
            
            # 문자열 리스트인 경우 timestamp는 없음
            normalized.append({
                "speaker": speaker,
                "발화내용_ko": 발화내용_ko,
                "timestamp": None  # 문자열 형식에는 timestamp 정보 없음
            })
    
    return {"utterances_normalized": normalized}

