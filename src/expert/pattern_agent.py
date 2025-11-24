from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


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


def _build_pattern_list_for_prompt() -> str:
    """LLM 프롬프트용 패턴 목록 생성"""
    patterns = []
    
    # Positive patterns
    for p in _PATTERN_DEFINITIONS.get("positive_patterns", []):
        name = p.get("name", "")
        english = p.get("english_name", "")
        patterns.append(f"'{name}' ({english})")
    
    # Negative patterns
    for p in _PATTERN_DEFINITIONS.get("negative_patterns", []):
        name = p.get("name", "")
        english = p.get("english_name", "")
        patterns.append(f"'{name}' ({english})")
    
    # Additional patterns
    for p in _PATTERN_DEFINITIONS.get("additional_patterns", []):
        name = p.get("name", "")
        english = p.get("english_name", "")
        patterns.append(f"'{name}' ({english})")
    
    return ", ".join(patterns)


_PATTERN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert in analyzing parent-child interaction patterns. "
            "Detect specific interaction patterns from labeled utterances. "
            f"Common patterns include: {_build_pattern_list_for_prompt()}. "
            "Return ONLY a JSON array of objects with: {pattern_name, description, utterance_indices, severity}. "
            "severity: 'low', 'medium', 'high'. "
            "pattern_name should match one of the Korean pattern names from the list above. "
            "No extra text."
        ),
    ),
    (
        "human",
        (
            "Labeled utterances:\n{utterances_labeled}\n\n"
            "Detect interaction patterns and return JSON array only."
        ),
    ),
])

_PATTERN_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert in validating parent-child interaction pattern detections. "
            "Your task is to verify if detected patterns are correctly identified. "
            "Pay special attention to false positives:\n"
            "- '행동 묘사' (BD) should NOT include negative, critical, or judgmental statements\n"
            "- '구체적 칭찬' (PR) should NOT include sarcasm, criticism, or negative comments\n"
            "- '반영적 경청' (RF/RD) should NOT include dismissive or invalidating responses\n"
            "- Negative patterns like '비판적 반응' should be correctly identified even if labeled as positive\n"
            "Return ONLY a JSON array with validation results. "
            "Each object should have: {pattern_index, is_valid, reason, corrected_pattern_name (if invalid)}. "
            "is_valid: true if the pattern is correctly identified, false otherwise. "
            "If invalid, provide corrected_pattern_name or null if it's not a pattern at all. "
            "No extra text."
        ),
    ),
    (
        "human",
        (
            "Detected patterns:\n{patterns}\n\n"
            "Relevant utterances:\n{utterances}\n\n"
            "Validate each pattern and return JSON array only."
        ),
    ),
])


def _detect_positive_patterns(utterances_labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """긍정적 패턴 탐지"""
    patterns = []
    positive_patterns = _PATTERN_DEFINITIONS.get("positive_patterns", [])
    
    for i, utt in enumerate(utterances_labeled):
        speaker = utt.get("speaker", "").lower()
        label = utt.get("label", "")
        text = utt.get("text", "").lower()
        
        if speaker not in ["parent", "mom", "mother", "dad", "father", "부모", "엄마", "아빠"]:
            continue
        
        # 패턴 1: 구체적 칭찬 (PR 라벨) - 비꼬는 말이나 부정적 내용 제외
        if label == "PR":
            # 비꼬는 말이나 부정적 키워드 체크
            sarcasm_keywords = [
                "fool", "stupid", "idiot", "really?", "great job (sarcastic)",
                "바보", "멍청", "정말 잘했네", "참 좋네", "훌륭하네 (비꼬는)"
            ]
            text_lower = text.lower()
            
            # 비꼬는 말이 아닐 때만 긍정적 패턴으로 인식
            if not any(kw in text_lower for kw in sarcasm_keywords):
                for pattern_def in positive_patterns:
                    if pattern_def.get("dpics_code") == "PR":
                        patterns.append({
                            "pattern_name": pattern_def.get("name", "구체적 칭찬"),
                            "description": f"구체적 칭찬을 사용했습니다: {utt.get('text', '')[:50]}",
                            "utterance_indices": [i],
                            "severity": "low",  # 긍정적 패턴은 낮은 severity
                            "pattern_type": "positive"
                        })
                        break
        
        # 패턴 2: 반영적 경청 (RF 또는 RD 라벨)
        if label in ["RF", "RD"]:
            for pattern_def in positive_patterns:
                if pattern_def.get("dpics_code") == "RF" or pattern_def.get("name") == "반영적 경청":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "반영적 경청"),
                        "description": f"반영적 경청을 사용했습니다: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "low",
                        "pattern_type": "positive"
                    })
                    break
        
        # 패턴 3: 행동 묘사 (BD 라벨) - 부정적 내용 제외
        if label == "BD" and speaker in ["parent", "mom", "mother", "dad", "father", "부모", "엄마", "아빠"]:
            # 부정적 키워드 체크 (행동 묘사는 중립적이거나 긍정적이어야 함)
            negative_keywords = [
                "fool", "stupid", "bad", "wrong", "idiot", "dumb", "loser",
                "바보", "멍청", "나쁜", "잘못", "못된", "싫어", "미워", "혼내"
            ]
            text_lower = text.lower()
            
            # 부정적 키워드가 없을 때만 긍정적 패턴으로 인식
            if not any(kw in text_lower for kw in negative_keywords):
                for pattern_def in positive_patterns:
                    if pattern_def.get("dpics_code") == "BD":
                        patterns.append({
                            "pattern_name": pattern_def.get("name", "행동 묘사"),
                            "description": f"행동을 묘사했습니다: {utt.get('text', '')[:50]}",
                            "utterance_indices": [i],
                            "severity": "low",
                            "pattern_type": "positive"
                        })
                        break
        
        # 패턴 4: 즉각적 긍정 강화 - 아이의 긍정적 행동 직후 칭찬
        if i > 0:
            prev_utt = utterances_labeled[i - 1]
            prev_speaker = prev_utt.get("speaker", "").lower()
            prev_label = prev_utt.get("label", "")
            
            # 이전 발화가 아이의 긍정적 행동이고, 현재가 부모의 칭찬인 경우
            if (prev_speaker in ["child", "chi", "kid", "아이"] and 
                prev_label in ["BD"] and 
                label == "PR"):
                for pattern_def in positive_patterns:
                    if pattern_def.get("name") == "즉각적 긍정 강화":
                        patterns.append({
                            "pattern_name": pattern_def.get("name", "즉각적 긍정 강화"),
                            "description": "아이의 긍정적 행동 직후 칭찬을 제공했습니다",
                            "utterance_indices": [i - 1, i],
                            "severity": "low",
                            "pattern_type": "positive"
                        })
                        break
        
        # 패턴 5: 감정 코칭 - 감정 관련 키워드와 함께 반영/공감
        emotion_keywords = ["화", "슬", "무서", "기쁘", "행복", "속상", "힘들", "짜증"]
        if label in ["RF", "RD"] and any(kw in text for kw in emotion_keywords):
            for pattern_def in positive_patterns:
                if pattern_def.get("name") == "감정 코칭":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "감정 코칭"),
                        "description": f"아이의 감정을 인정하고 코칭했습니다: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "low",
                        "pattern_type": "positive"
                    })
                    break
        
        # 패턴 6: 개방형 질문 (Q 라벨 + 개방형 질문 키워드)
        open_question_keywords = ["어떻게", "왜", "무엇", "어떤", "어디서", "언제", "어떠한"]
        if label == "Q" and any(kw in text for kw in open_question_keywords):
            for pattern_def in positive_patterns:
                if pattern_def.get("name") == "개방형 질문":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "개방형 질문"),
                        "description": f"개방형 질문을 사용했습니다: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "low",
                        "pattern_type": "positive"
                    })
                    break
    
    return patterns


def _detect_negative_patterns(utterances_labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """부정적 패턴 탐지"""
    patterns = []
    negative_patterns = _PATTERN_DEFINITIONS.get("negative_patterns", [])
    
    # 명령 카운트 (과도한 명령 탐지용)
    command_count = 0
    command_indices = []
    
    # 질문 카운트 (과도한 질문 탐지용)
    question_count = 0
    question_indices = []
    
    for i, utt in enumerate(utterances_labeled):
        speaker = utt.get("speaker", "").lower()
        label = utt.get("label", "")
        text = utt.get("text", "").lower()
        
        if speaker not in ["parent", "mom", "mother", "dad", "father", "부모", "엄마", "아빠"]:
            continue
        
        # 패턴 1: 비판적 반응 (NEG 라벨)
        if label == "NEG":
            for pattern_def in negative_patterns:
                if pattern_def.get("dpics_code") == "NEG" or pattern_def.get("name") == "비판적 반응":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "비판적 반응"),
                        "description": f"비판적 반응: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "high",
                        "pattern_type": "negative"
                    })
                    break
        
        # 패턴 2: 과도한 명령/지시 (CMD, IND 라벨)
        if label in ["CMD", "IND"]:
            command_count += 1
            command_indices.append(i)
            
            # 연속된 명령 탐지
            if i > 0:
                prev_label = utterances_labeled[i - 1].get("label", "")
                if prev_label in ["CMD", "IND"]:
                    for pattern_def in negative_patterns:
                        if pattern_def.get("name") == "과도한 명령/지시":
                            patterns.append({
                                "pattern_name": pattern_def.get("name", "과도한 명령/지시"),
                                "description": "연속된 명령을 사용했습니다",
                                "utterance_indices": [i - 1, i],
                                "severity": "medium",
                                "pattern_type": "negative"
                            })
                            break
        
        # 패턴 3: 반영/공감 부족 - 아이의 감정 표현 후 반영 없음
        if i > 0:
            prev_utt = utterances_labeled[i - 1]
            prev_speaker = prev_utt.get("speaker", "").lower()
            prev_text = prev_utt.get("text", "").lower()
            
            # 이전 발화가 아이의 감정 표현이고, 현재가 반영/공감이 아닌 경우
            emotion_keywords = ["화", "슬", "무서", "속상", "힘들", "짜증", "기쁘", "행복"]
            if (prev_speaker in ["child", "chi", "kid", "아이"] and 
                any(kw in prev_text for kw in emotion_keywords) and
                label not in ["RF", "RD", "PR"]):
                for pattern_def in negative_patterns:
                    if pattern_def.get("name") == "반영/공감 부족":
                        patterns.append({
                            "pattern_name": pattern_def.get("name", "반영/공감 부족"),
                            "description": "아이의 감정 표현에 반영/공감하지 않았습니다",
                            "utterance_indices": [i - 1, i],
                            "severity": "medium",
                            "pattern_type": "negative"
                        })
                        break
        
        # 패턴 4: 감정 무시/감정 기각
        dismiss_keywords = ["울지 마", "별거 아니", "남자가 왜", "왜 그런", "그만해"]
        if any(kw in text for kw in dismiss_keywords):
            for pattern_def in negative_patterns:
                if pattern_def.get("name") == "감정 무시/감정 기각":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "감정 무시/감정 기각"),
                        "description": f"아이의 감정을 무시하거나 기각했습니다: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "high",
                        "pattern_type": "negative"
                    })
                    break
        
        # 패턴 5: 심리적 통제
        control_keywords = ["안 좋아해", "아프잖아", "너 때문에", "네가 그러니까"]
        if any(kw in text for kw in control_keywords):
            for pattern_def in negative_patterns:
                if pattern_def.get("name") == "심리적 통제":
                    patterns.append({
                        "pattern_name": pattern_def.get("name", "심리적 통제"),
                        "description": f"심리적 통제를 사용했습니다: {utt.get('text', '')[:50]}",
                        "utterance_indices": [i],
                        "severity": "high",
                        "pattern_type": "negative"
                    })
                    break
        
        # 패턴 6: 과도한 질문 (Q 라벨 카운트)
        if label == "Q":
            question_count += 1
            question_indices.append(i)
            
            # 연속된 질문 탐지
            if i > 0:
                prev_label = utterances_labeled[i - 1].get("label", "")
                if prev_label == "Q":
                    for pattern_def in negative_patterns:
                        if pattern_def.get("name") == "과도한 질문":
                            patterns.append({
                                "pattern_name": pattern_def.get("name", "과도한 질문"),
                                "description": "연속된 질문을 사용했습니다",
                                "utterance_indices": [i - 1, i],
                                "severity": "medium",
                                "pattern_type": "negative"
                            })
                            break
        
        # 패턴 7: 긍정기회놓치기 - 아이의 긍정적 행동에 칭찬 없음
        if i > 0:
            prev_utt = utterances_labeled[i - 1]
            prev_speaker = prev_utt.get("speaker", "").lower()
            prev_label = prev_utt.get("label", "")
            
            if (prev_speaker in ["child", "chi", "kid", "아이"] and 
                prev_label in ["BD"] and 
                label != "PR"):
                patterns.append({
                    "pattern_name": "긍정기회놓치기",
                    "description": "아이의 긍정적 행동에 칭찬하지 않았습니다",
                    "utterance_indices": [i - 1, i],
                    "severity": "medium",
                    "pattern_type": "negative"
                })
    
    # 패턴 8: 과도한 명령/지시 (전체 카운트 기반)
    if command_count >= 5:  # 5개 이상이면 과도함
        for pattern_def in negative_patterns:
            if pattern_def.get("name") == "과도한 명령/지시":
                patterns.append({
                    "pattern_name": pattern_def.get("name", "과도한 명령/지시"),
                    "description": f"과도한 명령을 사용했습니다 (총 {command_count}개)",
                    "utterance_indices": command_indices[:10],  # 최대 10개
                    "severity": "high" if command_count >= 10 else "medium",
                    "pattern_type": "negative"
                })
                break
    
    # 패턴 9: 과도한 질문 (전체 카운트 기반)
    if question_count >= 5:  # 5개 이상이면 과도함
        for pattern_def in negative_patterns:
            if pattern_def.get("name") == "과도한 질문":
                patterns.append({
                    "pattern_name": pattern_def.get("name", "과도한 질문"),
                    "description": f"과도한 질문을 사용했습니다 (총 {question_count}개)",
                    "utterance_indices": question_indices[:10],  # 최대 10개
                    "severity": "high" if question_count >= 10 else "medium",
                    "pattern_type": "negative"
                })
                break
    
    return patterns


def _validate_patterns_with_llm(
    patterns: List[Dict[str, Any]], 
    utterances_labeled: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """LLM을 사용하여 탐지된 패턴들을 검증하고 잘못된 패턴 제거/수정"""
    if not patterns:
        return patterns
    
    try:
        llm = get_llm(mini=True)
        
        # 패턴 정보 포맷팅
        patterns_str = "\n".join([
            f"{i}. Pattern: {p.get('pattern_name')}, "
            f"Indices: {p.get('utterance_indices')}, "
            f"Description: {p.get('description')}"
            for i, p in enumerate(patterns)
        ])
        
        # 관련 발화 추출
        all_indices = set()
        for p in patterns:
            all_indices.update(p.get("utterance_indices", []))
        
        relevant_utterances = []
        for idx in sorted(all_indices):
            if 0 <= idx < len(utterances_labeled):
                utt = utterances_labeled[idx]
                relevant_utterances.append(
                    f"{idx}. [{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
                )
        
        utterances_str = "\n".join(relevant_utterances) if relevant_utterances else "No utterances"
        
        # LLM 검증
        res = (_PATTERN_VALIDATION_PROMPT | llm).invoke({
            "patterns": patterns_str,
            "utterances": utterances_str
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 배열 파싱
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            print("Pattern validation: No JSON found in LLM response")
            return patterns
        
        validations = json.loads(json_match.group(0))
        if not isinstance(validations, list):
            print("Pattern validation: Invalid response format")
            return patterns
        
        # 검증 결과 적용
        validated_patterns = []
        for i, pattern in enumerate(patterns):
            # 해당 패턴의 검증 결과 찾기
            validation = None
            for v in validations:
                if v.get("pattern_index") == i:
                    validation = v
                    break
            
            if validation is None:
                # 검증 결과가 없으면 유지 (안전한 선택)
                validated_patterns.append(pattern)
                continue
            
            is_valid = validation.get("is_valid", True)
            if is_valid:
                validated_patterns.append(pattern)
            else:
                # 잘못 탐지된 패턴 처리
                corrected_name = validation.get("corrected_pattern_name")
                reason = validation.get("reason", "Invalid pattern")
                
                if corrected_name and corrected_name != "null":
                    # 다른 패턴으로 수정
                    pattern["pattern_name"] = corrected_name
                    pattern["description"] = f"{corrected_name}: {reason}"
                    # pattern_type 재설정
                    pattern_type = "negative"
                    for pos_p in _PATTERN_DEFINITIONS.get("positive_patterns", []):
                        if pos_p.get("name") == corrected_name:
                            pattern_type = "positive"
                            break
                    pattern["pattern_type"] = pattern_type
                    validated_patterns.append(pattern)
                # corrected_name이 null이거나 없으면 패턴 제거 (필터링)
                print(f"Pattern filtered out: {pattern.get('pattern_name')} - {reason}")
        
        return validated_patterns
        
    except Exception as e:
        print(f"Pattern validation error: {e}")
        # 에러 발생 시 원본 패턴 반환
        return patterns


def detect_patterns_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ④ detect_patterns: 규칙/LLM로 패턴 찾기
    utterances_labeled를 받아서 탐지된 패턴들 반환
    """
    utterances_labeled = state.get("utterances_labeled") or []
    
    if not utterances_labeled:
        return {"patterns": []}
    
    # 규칙 기반 패턴 탐지
    patterns: List[Dict[str, Any]] = []
    
    # 긍정적 패턴 탐지
    positive_patterns = _detect_positive_patterns(utterances_labeled)
    patterns.extend(positive_patterns)
    
    # 부정적 패턴 탐지
    negative_patterns = _detect_negative_patterns(utterances_labeled)
    patterns.extend(negative_patterns)
    
    # LLM 기반 추가 패턴 탐지 (규칙 기반으로 탐지되지 않은 패턴 찾기)
    try:
        llm = get_llm(mini=True)
        
        # 발화를 포맷팅
        utterances_str = "\n".join([
            f"{i}. [{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
            for i, utt in enumerate(utterances_labeled)
        ])
        
        res = (_PATTERN_PROMPT | llm).invoke({"utterances_labeled": utterances_str})
        content = getattr(res, "content", "") or str(res)
        
        # JSON 배열 파싱
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            llm_patterns = json.loads(json_match.group(0))
            if isinstance(llm_patterns, list):
                # LLM이 탐지한 패턴 중 규칙 기반으로 탐지되지 않은 것만 추가
                existing_pattern_names = {p.get("pattern_name") for p in patterns}
                for llm_p in llm_patterns:
                    pattern_name = llm_p.get("pattern_name", "")
                    if pattern_name and pattern_name not in existing_pattern_names:
                        # pattern_type 설정
                        pattern_type = "negative"
                        for pos_p in _PATTERN_DEFINITIONS.get("positive_patterns", []):
                            if pos_p.get("name") == pattern_name:
                                pattern_type = "positive"
                                break
                        llm_p["pattern_type"] = pattern_type
                        patterns.append(llm_p)
    except Exception as e:
        print(f"LLM pattern detection error: {e}")
    
    # 중복 제거 (패턴명과 발화 인덱스 기반)
    seen = set()
    unique_patterns = []
    for p in patterns:
        # 패턴명과 관련 발화 인덱스로 중복 판단
        indices = tuple(sorted(p.get("utterance_indices", [])))
        key = (p.get("pattern_name"), indices)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # LLM 기반 패턴 검증 (잘못 탐지된 패턴 필터링)
    validated_patterns = _validate_patterns_with_llm(unique_patterns, utterances_labeled)
    
    return {"patterns": validated_patterns}

