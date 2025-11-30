"""
패턴 중앙 관리 유틸리티

pattern_definitions.json에서 패턴을 로드하고,
긍정적/부정적 패턴을 분리하여 제공하는 중앙화된 관리 모듈
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Set, Optional


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


# 패턴 정의 캐시
_PATTERN_DEFINITIONS = _load_pattern_definitions()


def get_positive_pattern_names() -> List[str]:
    """
    긍정적 패턴 이름 목록 반환
    
    Returns:
        List[str]: 긍정적 패턴 이름 리스트
    """
    patterns = _PATTERN_DEFINITIONS.get("positive_patterns", [])
    return [p.get("name", "") for p in patterns if p.get("name")]


def get_negative_pattern_names() -> List[str]:
    """
    부정적 패턴 이름 목록 반환
    
    Returns:
        List[str]: 부정적 패턴 이름 리스트
    """
    patterns = _PATTERN_DEFINITIONS.get("negative_patterns", [])
    return [p.get("name", "") for p in patterns if p.get("name")]


def get_all_pattern_names() -> List[str]:
    """
    모든 패턴 이름 목록 반환 (긍정적 + 부정적 + 추가 패턴)
    
    Returns:
        List[str]: 모든 패턴 이름 리스트
    """
    positive = get_positive_pattern_names()
    negative = get_negative_pattern_names()
    additional = [p.get("name", "") for p in _PATTERN_DEFINITIONS.get("additional_patterns", []) if p.get("name")]
    return positive + negative + additional


def normalize_pattern_name(pattern_name: str) -> str:
    """
    패턴 이름 정규화 (공백 제거)
    
    Args:
        pattern_name: 원본 패턴 이름
        
    Returns:
        str: 공백이 제거된 패턴 이름
    """
    if not pattern_name:
        return ""
    return pattern_name.replace(" ", "")


def is_positive_pattern(pattern_name: str) -> bool:
    """
    패턴이 긍정적인지 확인
    
    Args:
        pattern_name: 패턴 이름 (공백 포함/미포함 모두 가능)
        
    Returns:
        bool: 긍정적 패턴이면 True
    """
    if not pattern_name:
        return False
    
    normalized = normalize_pattern_name(pattern_name)
    positive_patterns = [normalize_pattern_name(p) for p in get_positive_pattern_names()]
    
    return normalized in positive_patterns


def is_negative_pattern(pattern_name: str) -> bool:
    """
    패턴이 부정적인지 확인
    
    Args:
        pattern_name: 패턴 이름 (공백 포함/미포함 모두 가능)
        
    Returns:
        bool: 부정적 패턴이면 True
    """
    if not pattern_name:
        return False
    
    normalized = normalize_pattern_name(pattern_name)
    negative_patterns = [normalize_pattern_name(p) for p in get_negative_pattern_names()]
    
    return normalized in negative_patterns


def get_negative_pattern_names_normalized() -> Set[str]:
    """
    정규화된 부정적 패턴 이름 집합 반환 (빠른 검색용)
    
    Returns:
        Set[str]: 공백이 제거된 부정적 패턴 이름 집합
    """
    return {normalize_pattern_name(p) for p in get_negative_pattern_names()}


def get_positive_pattern_names_normalized() -> Set[str]:
    """
    정규화된 긍정적 패턴 이름 집합 반환 (빠른 검색용)
    
    Returns:
        Set[str]: 공백이 제거된 긍정적 패턴 이름 집합
    """
    return {normalize_pattern_name(p) for p in get_positive_pattern_names()}


def extract_pattern_name(pattern_hint: str) -> Optional[str]:
    """
    pattern_hint에서 실제 패턴명만 추출하고 정규화된 패턴명과 매칭
    
    예: "명령과제시: 명령만 내림" -> "과도한 명령/지시" (정규화된 이름)
    
    Args:
        pattern_hint: 패턴 힌트 문자열
        
    Returns:
        Optional[str]: 매칭된 패턴 이름 (정규화된 버전), 매칭되지 않으면 None
    """
    if not pattern_hint:
        return None
    
    # 콜론(:) 이전 부분만 추출
    pattern_name = pattern_hint.split(":")[0].strip() if ":" in pattern_hint else pattern_hint.strip()
    if not pattern_name:
        return None
    
    # 정규화
    normalized_hint = normalize_pattern_name(pattern_name)
    
    # 모든 패턴 이름과 비교 (정규화된 버전으로)
    all_patterns = get_all_pattern_names()
    for known_pattern in all_patterns:
        normalized_known = normalize_pattern_name(known_pattern)
        
        # 정확히 일치하거나 포함 관계 확인
        if normalized_hint == normalized_known or normalized_hint in normalized_known or normalized_known in normalized_hint:
            return known_pattern
    
    # 매칭되지 않으면 정규화된 원본 반환
    return pattern_name


def is_pattern_negative(pattern_name: str, pattern_type: Optional[str] = None, severity: Optional[str] = None) -> bool:
    """
    패턴이 부정적인지 종합적으로 판단
    (pattern_name, pattern_type, severity를 모두 고려)
    
    Args:
        pattern_name: 패턴 이름
        pattern_type: 패턴 타입 ("positive", "negative" 등)
        severity: 심각도 ("low", "medium", "high")
        
    Returns:
        bool: 부정적 패턴이면 True
    """
    # pattern_type이 명시적으로 "negative"이면 부정적
    if pattern_type == "negative":
        return True
    
    # severity가 "medium" 또는 "high"이면 부정적일 가능성 높음
    if severity in ["medium", "high"]:
        # 단, pattern_type이 "positive"로 명시되어 있으면 제외
        if pattern_type == "positive":
            return False
        # pattern_name으로도 확인
        if pattern_name and is_negative_pattern(pattern_name):
            return True
    
    # pattern_name으로 확인
    if pattern_name and is_negative_pattern(pattern_name):
        return True
    
    return False

