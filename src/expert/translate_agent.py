from __future__ import annotations

from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.common import get_structured_llm


class TranslationItem(BaseModel):
    """번역된 발화 항목"""
    speaker: str = Field(description="Speaker label: MOM (부모/엄마/아빠) or CHI (아이/자녀)")
    korean: str = Field(description="한국어 원문 발화")
    english: str = Field(description="영어 번역된 발화")


class TranslationResponse(BaseModel):
    """번역 결과"""
    translations: List[TranslationItem] = Field(description="번역된 발화 리스트")


_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a professional translator specializing in parent-child dialogue. "
            "Translate Korean utterances to English while preserving the speaker labels and emotional tone. "
            "For each utterance, identify the speaker: use 'MOM' for parent/mother/father (부모/엄마/아빠/어머니/아버지) "
            "and 'CHI' for child (아이/자녀/아들/딸). "
            "Return a structured response with speaker label, Korean original, and English translation for each utterance."
        ),
    ),
    (
        "human",
        (
            "Korean utterances (one per line):\n{utterances_ko}\n\n"
            "Translate each utterance and return structured data with speaker (MOM/CHI), Korean original, and English translation."
        ),
    ),
])


def translate_ko_to_en_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ② translate_ko_to_en: 한국어 → 영어 번역
    utterances_normalized를 받아서 영어로 번역한 utterances_en 반환
    utterances_normalized 형식: [{speaker: "MOM"|"CHI", 발화내용_ko: str}, ...]
    """
    print("\n" + "="*60)
    print("[TranslateAgent] 한국어→영어 번역 시작")
    print("="*60)
    
    utterances_normalized = state.get("utterances_normalized") or []
    
    if not utterances_normalized:
        print("[TranslateAgent] 경고: 번역할 발화가 없음")
        print("="*60 + "\n")
        return {"utterances_en": []}
    
    print(f"[TranslateAgent] 번역할 발화 수: {len(utterances_normalized)}개")
    
    # 구조화된 형식인지 확인 (딕셔너리 리스트)
    if utterances_normalized and isinstance(utterances_normalized[0], dict):
        # 새로운 형식: 딕셔너리 리스트
        utterances_text = "\n".join([
            f"{utt.get('speaker', 'MOM')}: {utt.get('발화내용_ko', '')}"
            for utt in utterances_normalized
        ])
    else:
        # 기존 형식: 문자열 리스트 (하위 호환성)
        utterances_text = "\n".join(utterances_normalized)
    
    # Structured LLM로 번역
    print("[TranslateAgent] LLM으로 번역 중...")
    structured_llm = get_structured_llm(TranslationResponse, mini=True)
    
    try:
        res = (_TRANSLATE_PROMPT | structured_llm).invoke({"utterances_ko": utterances_text})
        print("[TranslateAgent] LLM 번역 완료")
        
        # Pydantic 모델에서 데이터 추출
        if isinstance(res, TranslationResponse):
            translations = res.translations
            
            # 구조화된 형식으로 반환: 한국어 원문 보존
            utterances_en = []
            for idx, item in enumerate(translations):
                # speaker label을 Parent/Child로 변환
                speaker_label = "Parent" if item.speaker.upper() == "MOM" else "Child"
                # timestamp 추출 (utterances_normalized에서)
                timestamp = None
                if idx < len(utterances_normalized) and isinstance(utterances_normalized[idx], dict):
                    utt_norm = utterances_normalized[idx]
                    timestamp = utt_norm.get("timestamp") or utt_norm.get("timestamp_ms") or utt_norm.get("time") or utt_norm.get("ts")
                # 구조화된 형식: 한국어 원문과 영어 번역 모두 포함
                utterances_en.append({
                    "speaker": speaker_label,
                    "korean": item.korean,
                    "english": item.english,
                    "text": item.english,  # 하위 호환성을 위한 필드
                    "original_ko": item.korean,  # 한국어 원문 명시적 보존
                    "timestamp": timestamp  # timestamp 포함
                })
            
            print(f"[TranslateAgent] 번역 완료: {len(utterances_en)}개 발화")
            print("="*60 + "\n")
            return {"utterances_en": utterances_en}
        
        # 폴백: 예상치 못한 형식
        # 구조화된 형식이면 발화내용_ko를 보존하여 반환
        if utterances_normalized and isinstance(utterances_normalized[0], dict):
            utterances_en = []
            for utt in utterances_normalized:
                speaker = utt.get('speaker', 'MOM')
                korean = utt.get('발화내용_ko', '')
                speaker_label = "Parent" if speaker.upper() == "MOM" else "Child"
                # timestamp 추출
                timestamp = utt.get("timestamp") or utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
                utterances_en.append({
                    "speaker": speaker_label,
                    "korean": korean,
                    "english": korean,  # 번역 실패 시 한국어 그대로
                    "text": korean,
                    "original_ko": korean,
                    "timestamp": timestamp  # timestamp 포함
                })
        else:
            # 기존 문자열 형식 (하위 호환성)
            utterances_en = utterances_normalized
        return {"utterances_en": utterances_en}
        
    except Exception as e:
        print(f"Translation error: {e}")
        # 에러 시 원문 반환 (한국어 보존)
        if utterances_normalized and isinstance(utterances_normalized[0], dict):
            utterances_en = []
            for utt in utterances_normalized:
                speaker = utt.get('speaker', 'MOM')
                korean = utt.get('발화내용_ko', '')
                speaker_label = "Parent" if speaker.upper() == "MOM" else "Child"
                # timestamp 추출
                timestamp = utt.get("timestamp") or utt.get("timestamp_ms") or utt.get("time") or utt.get("ts")
                utterances_en.append({
                    "speaker": speaker_label,
                    "korean": korean,
                    "english": korean,  # 번역 실패 시 한국어 그대로
                    "text": korean,
                    "original_ko": korean,
                    "timestamp": timestamp  # timestamp 포함
                })
        else:
            utterances_en = utterances_normalized
        return {"utterances_en": utterances_en}

