from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm
from src.utils.vector_store import search_expert_advice


_COACHING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a professional parenting coach. "
            "Create a personalized coaching plan based on the dialogue analysis. "
            "Return ONLY a JSON object with the following structure:\n"
            "{{\n"
            '  "summary": "요약 텍스트 (2-3문장)",\n'
            '  "challenge": {{\n'
            '    "title": "챌린지 제목 (패턴명 + 횟수 + 도전)",\n'
            '    "goal": "챌린지 목표 (1문장)",\n'
            '    "actions": ["액션1", "액션2", "액션3"],\n'
            '    "rationale": "챌린지 생성 이유 (패턴 발생 상황, 개선 필요성, 전문가 조언 참고)"\n'
            "  }},\n"
            '  "qa_tips": [\n'
            '    {{"question": "질문1", "answer": "답변1"}},\n'
            '    {{"question": "질문2", "answer": "답변2"}}\n'
            "  ]\n"
            "}}\n"
            "The challenge should focus on the most frequent pattern. "
            "Actions should be specific and actionable, and should incorporate the expert advice provided. "
            "The rationale field should explain why this challenge was created, including the pattern occurrence context and references to expert advice (e.g., '참고: [제목] ([저자], [출처])'). "
            "QA tips should address common questions about the pattern. "
            "All text in Korean. No extra text, only JSON."
        ),
    ),
    (
        "human",
        (
            "대화 요약:\n{summary}\n\n"
            "스타일 분석:\n{style_analysis}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "핵심 순간:\n{key_moments}\n\n"
            "{expert_advice_section}\n\n"
            "위 정보를 바탕으로 코칭 계획을 JSON 형식으로 작성해주세요. "
            "전문가 조언을 참고하여 챌린지의 actions와 goal을 구체적으로 작성하세요."
        ),
    ),
])


def _find_most_frequent_pattern(patterns: List[Dict[str, Any]], key_moments: Dict[str, Any]) -> Optional[str]:
    """
    가장 빈번한 패턴 찾기
    """
    pattern_counts = {}
    
    # key_moments의 pattern_examples에서 패턴별 발생 횟수 계산
    if isinstance(key_moments, dict):
        pattern_examples = key_moments.get("pattern_examples", [])
        for p in pattern_examples:
            pattern_name = p.get("pattern_name", "")
            occurrences = p.get("occurrences", 1)
            if pattern_name:
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + occurrences
    
    # patterns에서도 계산
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        if pattern_name:
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
    
    if not pattern_counts:
        return None
    
    # 가장 빈번한 패턴 반환
    return max(pattern_counts.items(), key=lambda x: x[1])[0]


def _build_challenge_query(patterns: List[Dict[str, Any]], key_moments: Dict[str, Any]) -> str:
    """
    챌린지 생성 시 검색 쿼리 생성
    """
    most_frequent = _find_most_frequent_pattern(patterns, key_moments)
    
    if most_frequent:
        query = f"{most_frequent} 패턴 개선 챌린지 가이드"
    else:
        query = "부모-자녀 상호작용 개선 챌린지 가이드"
    
    return query


def coaching_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑧ coaching_plan: 코칭/실천 계획 (LLM)
    """
    summary = state.get("summary") or ""
    style_analysis = state.get("style_analysis") or {}
    patterns = state.get("patterns") or []
    key_moments = state.get("key_moments") or {}
    
    if not summary and not patterns:
        return {
            "coaching_plan": {
                "summary": "분석할 데이터가 없습니다.",
                "challenge": {
                    "title": "",
                    "goal": "",
                    "period_days": 7,
                    "suggested_period": {
                        "start": _get_today_str(),
                        "end": _get_date_after_days(7)
                    },
                    "actions": [],
                    "rationale": ""
                },
                "qa_tips": []
            }
        }
    
    llm = get_llm(mini=False)
    
    # 포맷팅
    style_str = json.dumps(style_analysis, ensure_ascii=False, indent=2) if style_analysis else "(없음)"
    
    # 패턴 정보 포맷팅 (패턴명과 횟수 포함)
    pattern_examples = key_moments.get("pattern_examples", []) if isinstance(key_moments, dict) else []
    patterns_str = "\n".join([
        f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')} (발생 횟수: {p.get('occurrences', 0)})"
        for p in pattern_examples
    ]) if pattern_examples else (
        "\n".join([
            f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')}"
            for p in patterns
        ]) if patterns else "(없음)"
    )
    
    key_moments_str = ""
    if isinstance(key_moments, dict):
        positive = key_moments.get("positive", [])
        needs_improvement = key_moments.get("needs_improvement", [])
        if positive:
            key_moments_str += "긍정적 순간:\n" + "\n".join([
                f"- {m.get('reason', '')}"
                for m in positive
            ]) + "\n\n"
        if needs_improvement:
            key_moments_str += "개선이 필요한 순간:\n" + "\n".join([
                f"- {m.get('reason', '')}"
                for m in needs_improvement
            ])
    else:
        key_moments_str = str(key_moments) if key_moments else "(없음)"
    
    # VectorDB 검색 (챌린지 생성용)
    expert_advice_section = ""
    expert_references_list = []
    # USE_VECTOR_DB가 명시적으로 false가 아니면 검색 시도 (기본값: true)
    use_vector_db_env = os.getenv("USE_VECTOR_DB", "true").lower()
    use_vector_db = use_vector_db_env != "false"
    
    if use_vector_db:
        try:
            # 가장 빈번한 패턴 찾기
            most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
            
            # 검색 쿼리 생성
            query = _build_challenge_query(patterns, key_moments)
            
            # VectorDB 검색
            expert_advice = search_expert_advice(
                query=query,
                top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_CHALLENGE", "5")),
                threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.3")),  # 기본값 0.3으로 변경
                filters={
                    "advice_type": ["challenge_guide", "pattern_advice"],
                    "pattern_names": [most_frequent_pattern] if most_frequent_pattern else None
                }
            )
            
            if expert_advice:
                # 프롬프트용 전문가 조언 섹션
                expert_advice_section = "전문가 조언 및 챌린지 가이드:\n" + "\n".join([
                    f"[{advice['advice_type']}] {advice['title']}\n{advice['content'][:300]}..."
                    for advice in expert_advice
                ])
                
                # 레퍼런스 리스트 구성
                expert_references_list = [
                    {
                        "title": advice["title"],
                        "source": advice["source"],
                        "author": advice.get("author", ""),
                        "type": advice["advice_type"]
                    }
                    for advice in expert_advice
                ]
        except Exception as e:
            print(f"VectorDB 검색 오류 (coaching_plan): {e}")
            expert_advice_section = ""
            expert_references_list = []
    
    if not expert_advice_section:
        expert_advice_section = ""
    
    try:
        res = (_COACHING_PROMPT | llm).invoke({
            "summary": summary,
            "style_analysis": style_str,
            "patterns": patterns_str,
            "key_moments": key_moments_str,
            "expert_advice_section": expert_advice_section,
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 객체 파싱
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            coaching_data = json.loads(json_match.group(0))
            if isinstance(coaching_data, dict):
                # 챌린지 정보에 날짜 정보 추가
                if "challenge" in coaching_data and isinstance(coaching_data["challenge"], dict):
                    coaching_data["challenge"]["period_days"] = 7
                    coaching_data["challenge"]["suggested_period"] = {
                        "start": _get_today_str(),
                        "end": _get_date_after_days(7)
                    }
                    
                    # rationale 필드에 레퍼런스 정보 포함하여 생성
                    if expert_references_list:
                        # 기존 rationale이 있으면 유지, 없으면 생성
                        existing_rationale = coaching_data["challenge"].get("rationale", "")
                        
                        # 레퍼런스 텍스트 생성
                        reference_texts = []
                        for ref in expert_references_list:
                            author = ref.get("author", "")
                            source = ref.get("source", "")
                            title = ref.get("title", "")
                            
                            # "참고: [제목] ([저자], [출처])" 형식
                            if author and source:
                                ref_text = f"참고: '{title}' ({author}, {source})"
                            elif author:
                                ref_text = f"참고: '{title}' ({author})"
                            elif source:
                                ref_text = f"참고: '{title}' ({source})"
                            else:
                                ref_text = f"참고: '{title}'"
                            
                            reference_texts.append(ref_text)
                        
                        # rationale에 레퍼런스 추가
                        if reference_texts:
                            references_section = " " + " ".join(reference_texts)
                            if existing_rationale:
                                # 기존 rationale 끝에 레퍼런스 추가
                                coaching_data["challenge"]["rationale"] = existing_rationale + references_section
                            else:
                                # rationale이 없으면 패턴 정보와 함께 생성
                                most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                                pattern_info = f"이 챌린지는 '{most_frequent_pattern}' 패턴이 감지되어 생성되었습니다." if most_frequent_pattern else "이 챌린지는 분석 결과를 바탕으로 생성되었습니다."
                                coaching_data["challenge"]["rationale"] = pattern_info + references_section
                    else:
                        # 레퍼런스가 없어도 rationale이 없으면 기본 텍스트 생성
                        if "rationale" not in coaching_data["challenge"] or not coaching_data["challenge"].get("rationale"):
                            most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                            if most_frequent_pattern:
                                coaching_data["challenge"]["rationale"] = f"이 챌린지는 '{most_frequent_pattern}' 패턴이 감지되어 생성되었습니다."
                            else:
                                coaching_data["challenge"]["rationale"] = "이 챌린지는 분석 결과를 바탕으로 생성되었습니다."
                
                return {"coaching_plan": coaching_data}
    except Exception as e:
        print(f"Coaching plan error: {e}")
    
    # 폴백: 기본 구조 반환
    return {
        "coaching_plan": {
            "summary": "코칭 계획 생성 중 오류가 발생했습니다.",
            "challenge": {
                "title": "",
                "goal": "",
                "period_days": 7,
                "suggested_period": {
                    "start": _get_today_str(),
                    "end": _get_date_after_days(7)
                },
                "actions": [],
                "rationale": ""
            },
            "qa_tips": []
        }
    }


def _get_today_str() -> str:
    """오늘 날짜를 YYYY-MM-DD 형식으로 반환"""
    return datetime.now().strftime("%Y-%m-%d")


def _get_date_after_days(days: int) -> str:
    """오늘부터 days일 후 날짜를 YYYY-MM-DD 형식으로 반환"""
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

