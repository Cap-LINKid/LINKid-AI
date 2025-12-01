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


def _build_pattern_details_for_prompt() -> str:
    """LLM 프롬프트용 패턴 상세 정보 생성 (정의, 예시 포함)"""
    sections = []
    
    # 긍정적 패턴 섹션
    positive_patterns = _PATTERN_DEFINITIONS.get("positive_patterns", [])
    if positive_patterns:
        sections.append("=== 긍정적 패턴 (Positive Patterns) ===")
        for p in positive_patterns:
            name = p.get("name", "")
            english = p.get("english_name", "")
            definition = p.get("definition", "")
            dpics_code = p.get("dpics_code", "")
            examples = p.get("examples", [])
            
            pattern_info = f"\n- '{name}' ({english})"
            if dpics_code:
                # RF는 RD로 통일 (dpics_electra.py 기준)
                if dpics_code == "RF":
                    pattern_info += f" [DPICS: RD (RF와 동일)]"
                else:
                    pattern_info += f" [DPICS: {dpics_code}]"
            if definition:
                pattern_info += f"\n  정의: {definition}"
            if examples:
                pattern_info += f"\n  예시: {', '.join(examples[:2])}"  # 최대 2개 예시
            sections.append(pattern_info)
    
    # 부정적 패턴 섹션
    negative_patterns = _PATTERN_DEFINITIONS.get("negative_patterns", [])
    if negative_patterns:
        sections.append("\n=== 부정적 패턴 (Negative Patterns) ===")
        for p in negative_patterns:
            name = p.get("name", "")
            english = p.get("english_name", "")
            definition = p.get("definition", "")
            dpics_code = p.get("dpics_code", "")
            examples = p.get("examples", [])
            
            pattern_info = f"\n- '{name}' ({english})"
            if dpics_code:
                if isinstance(dpics_code, list):
                    # 리스트인 경우 각 코드 처리
                    codes = []
                    for code in dpics_code:
                        if code == "RF":
                            codes.append("RD (RF와 동일)")
                        else:
                            codes.append(code)
                    pattern_info += f" [DPICS: {', '.join(codes)}]"
                else:
                    if dpics_code == "RF":
                        pattern_info += f" [DPICS: RD (RF와 동일)]"
                    else:
                        pattern_info += f" [DPICS: {dpics_code}]"
            if definition:
                pattern_info += f"\n  정의: {definition}"
            if examples:
                pattern_info += f"\n  예시: {', '.join(examples[:2])}"  # 최대 2개 예시
            sections.append(pattern_info)
    
    # 추가 패턴 섹션
    additional_patterns = _PATTERN_DEFINITIONS.get("additional_patterns", [])
    if additional_patterns:
        sections.append("\n=== 추가 패턴 (Additional Patterns) ===")
        for p in additional_patterns:
            name = p.get("name", "")
            english = p.get("english_name", "")
            definition = p.get("definition", "")
            
            pattern_info = f"\n- '{name}' ({english})"
            if definition:
                pattern_info += f"\n  정의: {definition}"
            sections.append(pattern_info)
    
    return "\n".join(sections)


_PATTERN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert in analyzing parent-child interaction patterns. "
            "Detect specific interaction patterns from labeled utterances using the pattern definitions provided below. "
            "\n"
            f"{_build_pattern_details_for_prompt()}"
            "\n\n"
            "DPICS label meanings (dpics_electra.py의 라벨 매핑 기준):\n"
            "- RD: Reflective Statement (반영적 경청) - ALWAYS positive, shows understanding and empathy\n"
            "  Note: RF and RD are the same (both refer to Reflective Statement)\n"
            "- PR: Praise (구체적 칭찬) - positive, specific praise (Labeled/Unlabeled/Prosocial Talk)\n"
            "- BD: Behavior Description (행동 묘사) - positive, neutral description of behavior\n"
            "- NEG: Negative Talk (비판적 반응) - negative, critical response\n"
            "- Q: Question (질문) - context dependent\n"
            "- CMD: Command (명령) - can be negative if excessive\n"
            "- NT: Neutral Talk (중립적 대화)\n"
            "\n"
            "Important: When pattern definitions mention 'RF' as dpics_code, it should be treated as 'RD' (Reflective Statement).\n"
            "\n"
            "Instructions:\n"
            "1. Analyze the labeled utterances carefully and match them to the patterns defined above.\n"
            "2. Use the pattern definitions, examples, and DPICS codes to identify patterns accurately.\n"
            "3. Consider the context and sequence of utterances when detecting patterns.\n"
            "4. For each detected pattern, provide: pattern_name (must match Korean name from definitions), description, utterance_indices, severity (low/medium/high), and pattern_type (positive/negative).\n"
            "5. pattern_name must exactly match one of the Korean pattern names from the definitions above.\n"
            "\n"
            "Return ONLY a JSON array of objects with: {{pattern_name, description, utterance_indices, severity, pattern_type}}. "
            "No extra text."
        ),
    ),
    (
        "human",
        (
            "Labeled utterances:\n{utterances_labeled}\n\n"
            "Detect interaction patterns based on the pattern definitions provided and return JSON array only."
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
            "- '반영적 경청' (RD) should NOT include dismissive or invalidating responses\n"
            "- IMPORTANT: RD labels indicate reflective listening and are ALWAYS positive patterns. "
            "Do NOT reclassify RD labeled utterances as negative patterns like '비판적 반응'.\n"
            "- Only reclassify as negative if the utterance clearly contains criticism, sarcasm, or invalidation "
            "AND the label is clearly wrong (e.g., PR labeled but contains sarcasm)\n"
            "Return ONLY a JSON array with validation results. "
            "Each object should have: {{pattern_index, is_valid, reason, corrected_pattern_name (if invalid)}}. "
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


# 규칙 기반 패턴 탐지 함수들은 제거됨
# 이제 LLM이 중앙화된 패턴 정의를 사용하여 패턴을 탐지함


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
    ④ detect_patterns: LLM 기반 패턴 탐지 (중앙화된 패턴 정의 사용)
    utterances_labeled를 받아서 탐지된 패턴들 반환
    """
    utterances_labeled = state.get("utterances_labeled") or []
    
    if not utterances_labeled:
        return {"patterns": []}
    
    # LLM 기반 패턴 탐지 (중앙화된 패턴 정의 사용)
    patterns: List[Dict[str, Any]] = []
    
    try:
        llm = get_llm(mini=False)  # 패턴 탐지는 더 정확한 모델 사용
        
        # 발화를 포맷팅 (라벨 정보 명시, 한국어 원문 포함)
        utterances_str = "\n".join([
            f"{i}. [{utt.get('speaker', 'Unknown')}] [Label: {utt.get('label', 'N/A')}] {utt.get('original_ko', utt.get('text', ''))}"
            for i, utt in enumerate(utterances_labeled)
        ])
        
        res = (_PATTERN_PROMPT | llm).invoke({"utterances_labeled": utterances_str})
        content = getattr(res, "content", "") or str(res)
        
        # JSON 배열 파싱
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            llm_patterns = json.loads(json_match.group(0))
            if isinstance(llm_patterns, list):
                # 패턴 타입이 없으면 패턴 정의에서 확인하여 추가
                for llm_p in llm_patterns:
                    pattern_name = llm_p.get("pattern_name", "")
                    
                    # pattern_type이 없으면 패턴 정의에서 확인
                    if "pattern_type" not in llm_p:
                        pattern_type = "negative"  # 기본값
                        # 긍정적 패턴 확인
                        for pos_p in _PATTERN_DEFINITIONS.get("positive_patterns", []):
                            if pos_p.get("name") == pattern_name:
                                pattern_type = "positive"
                                break
                        # 부정적 패턴 확인
                        if pattern_type == "negative":
                            for neg_p in _PATTERN_DEFINITIONS.get("negative_patterns", []):
                                if neg_p.get("name") == pattern_name:
                                    pattern_type = "negative"
                                    break
                        llm_p["pattern_type"] = pattern_type
                    
                    patterns.append(llm_p)
        else:
            print(f"LLM pattern detection: No JSON array found in response")
    except Exception as e:
        print(f"LLM pattern detection error: {e}")
        import traceback
        traceback.print_exc()
    
    # 중복 제거 (패턴명과 발화 인덱스 기반)
    seen = set()
    unique_patterns = []
    for p in patterns:
        indices = tuple(sorted(p.get("utterance_indices", [])))
        key = (p.get("pattern_name"), indices)
        
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # LLM 기반 패턴 검증
    validated_patterns = _validate_patterns_with_llm(unique_patterns, utterances_labeled)
    
    return {"patterns": validated_patterns}

