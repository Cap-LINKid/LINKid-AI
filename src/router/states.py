from __future__ import annotations

from typing import Dict, Any, TypedDict, List, Optional


class RouterState(TypedDict, total=False):
    # 입력
    utterances_ko: List[str]  # 한국어 대화 발화 리스트
    challenge_spec: Dict[str, Any]  # 이번 주 챌린지 스펙
    meta: Dict[str, Any]  # 메타데이터
    
    # 중간 처리 결과
    utterances_normalized: List[Dict[str, str]]  # ① preprocess 결과 (스피커 정규화) - [{speaker: "MOM"|"CHI", 발화내용_ko: str}, ...]
    utterances_en: List[Dict[str, Any]]  # ② translate 결과 (영어 번역) - [{speaker, korean, english, text, original_ko}, ...]
    utterances_labeled: List[Dict[str, Any]]  # ③ label 결과 (라벨링된 발화)
    patterns: List[Dict[str, Any]]  # ④ detect_patterns 결과 (탐지된 패턴)
    
    # 병렬 분석 결과
    summary: str  # ⑤ summarize 결과
    key_moments: Dict[str, Any]  # ⑥ key_moments 결과 - {positive: [], needs_improvement: [], pattern_examples: []}
    style_analysis: Dict[str, Any]  # ⑦ analyze_style 결과
    coaching_plan: Dict[str, Any]  # ⑧ coaching_plan 결과
    challenge_eval: Dict[str, Any]  # ⑨ challenge_eval 결과
    summary_diagnosis: Dict[str, Any]  # summary_diagnosis 결과 - {stage_name, positive_ratio, negative_ratio}
    
    # 최종 결과
    result: Dict[str, Any]  # ⑩ aggregate_result 최종 JSON
    
    # 기존 필드 (하위 호환성)
    message: str
    dialogue: str
    context: str
    tdl: Dict[str, Any]
    annotated: str
    highlights: List[str]
    advice: str
