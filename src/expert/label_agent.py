from __future__ import annotations

import os
from typing import Dict, Any, List

from src.utils.dpics import label_lines_dpics_llm

# ELECTRA 모델 사용 여부 (환경 변수로 제어 가능)
USE_ELECTRA = os.getenv("USE_DPICS_ELECTRA", "true").lower() == "true"

try:
    from src.utils.dpics_electra import label_lines_dpics_electra
    ELECTRA_AVAILABLE = True
except ImportError:
    ELECTRA_AVAILABLE = False
    print("경고: dpics_electra를 사용할 수 없습니다. LLM 기반 라벨링을 사용합니다.")


def label_utterances_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ③ label_utterances: 영어 발화에 라벨 달기 (파인튜닝 ELECTRA 또는 LLM)
    utterances_en을 받아서 라벨링된 utterances_labeled 반환
    utterances_en 형식: [{speaker, korean, english, text, original_ko}, ...] 또는 ["Parent: ...", ...]
    """
    utterances_en = state.get("utterances_en") or []
    
    if not utterances_en:
        return {"utterances_labeled": []}
    
    # 구조화된 형식인지 확인
    is_structured = isinstance(utterances_en[0], dict) if utterances_en else False
    
    # 영어 발화를 텍스트로 합치기 (DPICS 라벨링용)
    if is_structured:
        utterances_text = "\n".join([
            f"{utt.get('speaker', 'Parent')}: {utt.get('english', utt.get('text', ''))}"
            for utt in utterances_en
        ])
        # 한국어 발화를 텍스트로 합치기 (DPICS LLM 라벨링용)
        utterances_text_ko = "\n".join([
            f"{utt.get('speaker', 'Parent')}: {utt.get('korean', utt.get('original_ko', ''))}"
            for utt in utterances_en
            if utt.get('korean') or utt.get('original_ko')
        ])
    else:
        utterances_text = "\n".join(utterances_en)
        utterances_text_ko = ""  # 기존 형식에는 한국어 정보 없음
    
    # DPICS 라벨링 (ELECTRA 모델 또는 LLM 기반)
    if USE_ELECTRA and ELECTRA_AVAILABLE:
        try:
            labeled_pairs = label_lines_dpics_electra(utterances_text)
        except Exception as e:
            print(f"ELECTRA 모델 라벨링 실패, LLM으로 폴백: {e}")
            # LLM은 한국어 발화를 대상으로 라벨링
            labeled_pairs = label_lines_dpics_llm(utterances_text_ko if utterances_text_ko else utterances_text)
    else:
        # LLM은 한국어 발화를 대상으로 라벨링
        labeled_pairs = label_lines_dpics_llm(utterances_text_ko if utterances_text_ko else utterances_text)
    
    # 발화와 라벨을 딕셔너리 리스트로 변환
    # LLM이 반환한 line이 원본과 정확히 일치하지 않을 수 있으므로,
    # 원본 utterances_en을 기준으로 스피커를 추출하고 LLM 결과와 매칭
    utterances_labeled: List[Dict[str, Any]] = []
    
    # 원본 발화에서 스피커와 텍스트 추출 (인덱스별로 미리 파싱)
    original_parsed = []
    for orig_utt in utterances_en:
        if is_structured:
            # 구조화된 형식
            speaker = orig_utt.get("speaker", "Unknown")
            text = orig_utt.get("english", orig_utt.get("text", ""))
            korean = orig_utt.get("korean", orig_utt.get("original_ko", ""))
            timestamp = orig_utt.get("timestamp")  # timestamp 추출
            english_text = f"{speaker}: {text}"
            original_parsed.append({
                "speaker": speaker,
                "text": text,
                "korean": korean,
                "english": text,
                "original": english_text,  # DPICS 라벨링용 원본 (영어)
                "original_ko": korean,  # 한국어 원문
                "timestamp": timestamp,  # timestamp 포함
                "matched": False
            })
        else:
            # 기존 문자열 형식 (하위 호환성)
            speaker = "Unknown"
            text = orig_utt
            if orig_utt.startswith("Parent:"):
                speaker = "Parent"
                text = orig_utt.replace("Parent:", "").strip()
            elif orig_utt.startswith("Child:"):
                speaker = "Child"
                text = orig_utt.replace("Child:", "").strip()
            original_parsed.append({
                "speaker": speaker,
                "text": text,
                "korean": "",  # 기존 형식에는 한국어 정보 없음
                "english": text,
                "original": orig_utt,
                "original_ko": "",  # 기존 형식에는 한국어 정보 없음
                "timestamp": None,  # 기존 형식에는 timestamp 정보 없음
                "matched": False
            })
    
    # LLM이 반환한 결과와 원본을 매칭
    for llm_line, label in labeled_pairs:
        matched_idx = None
        
        # 1. 정확한 매칭 시도 (원본 전체 문자열)
        for idx, orig in enumerate(original_parsed):
            if not orig["matched"] and (orig["original"] == llm_line or orig["original"].strip() == llm_line.strip()):
                matched_idx = idx
                break
        
        # 2. 부분 매칭 시도 (텍스트 내용만 비교, 스피커 라벨 무시)
        if matched_idx is None:
            llm_text_clean = llm_line.replace("Parent:", "").replace("Child:", "").strip().lower()
            for idx, orig in enumerate(original_parsed):
                if not orig["matched"]:
                    orig_text_clean = orig["text"].strip().lower()
                    if orig_text_clean and (llm_text_clean == orig_text_clean or 
                                           llm_text_clean in orig_text_clean or 
                                           orig_text_clean in llm_text_clean):
                        matched_idx = idx
                        break
        
        # 3. 매칭 실패 시 순서대로 첫 번째 미사용 항목 사용
        if matched_idx is None:
            for idx, orig in enumerate(original_parsed):
                if not orig["matched"]:
                    matched_idx = idx
                    break
        
        # 매칭된 항목 사용
        # 순환 참조 방지를 위해 필요한 필드만 명시적으로 추출하여 새로운 딕셔너리 생성
        if matched_idx is not None:
            orig = original_parsed[matched_idx]
            utterances_labeled.append({
                "speaker": str(orig.get("speaker", "")),
                "text": str(orig.get("text", "")),
                "label": str(label),  # DPICS 코드 (PR, RD, BD, NT, Q, CMD, NEG, IGN, OTH)
                "original": str(orig.get("original_ko", orig.get("original", ""))),  # 한국어 원문 우선, 없으면 영어
                "original_ko": str(orig.get("original_ko", "")),  # 한국어 원문 명시적 포함
                "english": str(orig.get("english", orig.get("text", ""))),  # 영어 번역 포함
                "timestamp": orig.get("timestamp")  # timestamp 포함 (int 또는 None)
            })
            original_parsed[matched_idx]["matched"] = True
    
    # LLM이 반환하지 않은 원본 발화들도 추가 (기본 라벨)
    # 순환 참조 방지를 위해 필요한 필드만 명시적으로 추출하여 새로운 딕셔너리 생성
    for orig in original_parsed:
        if not orig["matched"]:
            utterances_labeled.append({
                "speaker": str(orig.get("speaker", "")),
                "text": str(orig.get("text", "")),
                "label": "OTH",  # 기본 라벨
                "original": str(orig.get("original_ko", orig.get("original", ""))),  # 한국어 원문 우선
                "original_ko": str(orig.get("original_ko", "")),  # 한국어 원문 명시적 포함
                "english": str(orig.get("english", orig.get("text", ""))),  # 영어 번역 포함
                "timestamp": orig.get("timestamp")  # timestamp 포함 (int 또는 None)
            })
    
    return {"utterances_labeled": utterances_labeled}

