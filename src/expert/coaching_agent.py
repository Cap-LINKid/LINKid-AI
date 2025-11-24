from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

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
            "IMPORTANT: The rationale field should ONLY be included if expert advice references are provided in the expert_advice_section. "
            "If expert_advice_section is empty or no references are available, do NOT create a rationale field (leave it empty or omit it). "
            "When expert advice is available, include references in the format: '참고: [제목] ([저자], [출처])'. "
            "Do NOT make up or hallucinate references. Only use references that are explicitly provided in the expert_advice_section. "
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
            "전문가 조언을 참고하여 챌린지의 actions와 goal을 구체적으로 작성하세요.\n"
            "중요: expert_advice_section이 비어있거나 '(없음)'인 경우, rationale 필드를 생성하지 마세요. "
            "rationale은 반드시 expert_advice_section에 제공된 레퍼런스가 있을 때만 포함하세요."
        ),
    ),
])


def _find_most_frequent_pattern(patterns: List[Dict[str, Any]], key_moments: Dict[str, Any]) -> Optional[str]:
    """
    가장 빈번한 부정적 패턴 찾기 (챌린지 생성용)
    """
    pattern_counts = {}
    
    # key_moments의 pattern_examples에서 부정적 패턴만 필터링하여 계산
    if isinstance(key_moments, dict):
        pattern_examples = key_moments.get("pattern_examples", [])
        for p in pattern_examples:
            pattern_name = p.get("pattern_name", "")
            pattern_type = p.get("pattern_type", "")
            severity = p.get("severity", "")
            occurrences = p.get("occurrences", 1)
            
            # 부정적 패턴만 필터링 (pattern_type이 "negative"이거나 severity가 "medium"/"high")
            is_negative = (
                pattern_type == "negative" or 
                severity in ["medium", "high"] or
                # 패턴명으로도 판단 (부정적 패턴명 목록)
                pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                                "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
            )
            
            if pattern_name and is_negative:
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + occurrences
    
    # patterns에서도 부정적 패턴만 계산
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        
        # 부정적 패턴만 필터링
        is_negative = (
            pattern_type == "negative" or 
            severity in ["medium", "high"] or
            # 패턴명으로도 판단
            pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                            "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
        )
        
        if pattern_name and is_negative:
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
    
    if not pattern_counts:
        return None
    
    # 가장 빈번한 부정적 패턴 반환
    most_frequent = max(pattern_counts.items(), key=lambda x: x[1])[0]
    return most_frequent


def _build_challenge_query(patterns: List[Dict[str, Any]], key_moments: Dict[str, Any]) -> str:
    """
    챌린지 생성 시 검색 쿼리 생성 (부정적 패턴만 고려)
    """
    most_frequent = _find_most_frequent_pattern(patterns, key_moments)
    
    # 패턴명 추출 (부정적 패턴만, 공백 제거)
    pattern_names = []
    if most_frequent:
        # 공백 제거하여 정규화
        normalized_pattern = most_frequent.replace(" ", "")
        pattern_names.append(normalized_pattern)
    
    # key_moments에서 추가 부정적 패턴 찾기
    if isinstance(key_moments, dict):
        pattern_examples = key_moments.get("pattern_examples", [])
        for p in pattern_examples:
            pattern_name = p.get("pattern_name", "")
            pattern_type = p.get("pattern_type", "")
            severity = p.get("severity", "")
            
            # 부정적 패턴만 필터링
            is_negative = (
                pattern_type == "negative" or 
                severity in ["medium", "high"] or
                pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                                "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
            )
            
            if pattern_name and is_negative:
                normalized = pattern_name.replace(" ", "")
                if normalized not in pattern_names:
                    pattern_names.append(normalized)
    
    # patterns에서도 추가 부정적 패턴 찾기
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        
        # 부정적 패턴만 필터링
        is_negative = (
            pattern_type == "negative" or 
            severity in ["medium", "high"] or
            pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                            "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
        )
        
        if pattern_name and is_negative:
            normalized = pattern_name.replace(" ", "")
            if normalized not in pattern_names:
                pattern_names.append(normalized)
    
    # 쿼리 생성 (여러 부정적 패턴 고려)
    if pattern_names:
        if len(pattern_names) == 1:
            query = f"{pattern_names[0]} 패턴 개선 챌린지 가이드"
        else:
            # 여러 패턴이 있으면 모두 포함
            query = f"{', '.join(pattern_names[:3])} 패턴 개선 챌린지 가이드"
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
    
    # 패턴 정보 포맷팅 (부정적 패턴만 포함, 패턴명과 횟수 포함)
    pattern_examples = key_moments.get("pattern_examples", []) if isinstance(key_moments, dict) else []
    
    # 부정적 패턴만 필터링
    negative_pattern_examples = []
    for p in pattern_examples:
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        pattern_name = p.get("pattern_name", "")
        is_negative = (
            pattern_type == "negative" or 
            severity in ["medium", "high"] or
            pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                            "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
        )
        if is_negative:
            negative_pattern_examples.append(p)
    
    negative_patterns = []
    for p in patterns:
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        pattern_name = p.get("pattern_name", "")
        is_negative = (
            pattern_type == "negative" or 
            severity in ["medium", "high"] or
            pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                            "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
        )
        if is_negative:
            negative_patterns.append(p)
    
    patterns_str = "\n".join([
        f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')} (발생 횟수: {p.get('occurrences', 0)})"
        for p in negative_pattern_examples
    ]) if negative_pattern_examples else (
        "\n".join([
            f"- {p.get('pattern_name', '알 수 없음')}: {p.get('description', '')}"
            for p in negative_patterns
        ]) if negative_patterns else "(없음)"
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
            print(f"[VectorDB] 챌린지 생성 검색 시작 - most_frequent_pattern: {most_frequent_pattern}")
            
            # 모든 관련 부정적 패턴 수집 (공백 제거하여 정규화)
            all_patterns = []
            if most_frequent_pattern:
                normalized = most_frequent_pattern.replace(" ", "")
                if normalized not in all_patterns:
                    all_patterns.append(normalized)
            
            # key_moments에서 추가 부정적 패턴 찾기
            if isinstance(key_moments, dict):
                pattern_examples = key_moments.get("pattern_examples", [])
                for p in pattern_examples:
                    pattern_name = p.get("pattern_name", "")
                    pattern_type = p.get("pattern_type", "")
                    severity = p.get("severity", "")
                    
                    # 부정적 패턴만 필터링
                    is_negative = (
                        pattern_type == "negative" or 
                        severity in ["medium", "high"] or
                        pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                                        "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
                    )
                    
                    if pattern_name and is_negative:
                        normalized = pattern_name.replace(" ", "")
                        if normalized not in all_patterns:
                            all_patterns.append(normalized)
            
            # patterns에서도 추가 부정적 패턴 찾기
            for p in patterns:
                pattern_name = p.get("pattern_name", "")
                pattern_type = p.get("pattern_type", "")
                severity = p.get("severity", "")
                
                # 부정적 패턴만 필터링
                is_negative = (
                    pattern_type == "negative" or 
                    severity in ["medium", "high"] or
                    pattern_name in ["비판적 반응", "비판적반응", "명령과제시", "과도한 질문", 
                                    "감정 무시", "감정 기각", "심리적 통제", "긍정기회놓치기"]
                )
                
                if pattern_name and is_negative:
                    normalized = pattern_name.replace(" ", "")
                    if normalized not in all_patterns:
                        all_patterns.append(normalized)
            
            print(f"[VectorDB] 수집된 패턴 목록: {all_patterns}")
            
            # 검색 쿼리 생성
            query = _build_challenge_query(patterns, key_moments)
            print(f"[VectorDB] 검색 쿼리: {query}")
            
            # 필터 조건: 여러 패턴을 OR 조건으로 검색
            filter_patterns = all_patterns if all_patterns else None
            print(f"[VectorDB] 필터 패턴: {filter_patterns}")
            
            # VectorDB 검색
            expert_advice = search_expert_advice(
                query=query,
                top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_CHALLENGE", "5")),
                threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.3")),  # 기본값 0.3으로 변경
                filters={
                    "advice_type": ["challenge_guide", "pattern_advice", "coaching"],  # coaching도 포함
                    "pattern_names": filter_patterns  # 여러 패턴을 OR 조건으로 검색
                }
            )
            
            print(f"[VectorDB] 검색 결과 개수: {len(expert_advice)}")
            if expert_advice:
                print(f"[VectorDB] 검색 결과 제목들:")
                for advice in expert_advice:
                    print(f"  - [{advice['advice_type']}] {advice['title']} (유사도: {advice.get('relevance_score', 0):.3f})")
            
            if expert_advice:
                # 프롬프트용 전문가 조언 섹션
                expert_advice_section = "전문가 조언 및 챌린지 가이드:\n" + "\n".join([
                    f"[{advice['advice_type']}] {advice['title']}\n{advice['content'][:300]}..."
                    for advice in expert_advice
                ])
                
                # 레퍼런스 리스트 구성 (중복 제거)
                seen_titles = set()
                expert_references_list = []
                for advice in expert_advice:
                    title = advice["title"]
                    # 제목으로 중복 체크
                    if title not in seen_titles:
                        seen_titles.add(title)
                        expert_references_list.append({
                            "title": title,
                            "source": advice["source"],
                            "author": advice.get("author", ""),
                            "type": advice["advice_type"]
                        })
                print(f"[VectorDB] 전문가 조언 섹션 생성 완료 - {len(expert_references_list)}개 레퍼런스 (중복 제거 후)")
            else:
                print(f"[VectorDB] 검색 결과 없음 (threshold 미달 또는 필터 조건 불일치)")
        except Exception as e:
            print(f"[VectorDB] 검색 오류 (coaching_plan): {e}")
            import traceback
            traceback.print_exc()
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
                    
                    # rationale 필드 처리: 레퍼런스가 있을 때만 포함
                    if expert_references_list:
                        # 레퍼런스 텍스트 생성 (중복 제거)
                        seen_refs = set()
                        reference_texts = []
                        for ref in expert_references_list:
                            author = ref.get("author", "")
                            source = ref.get("source", "")
                            title = ref.get("title", "")
                            
                            # 중복 체크용 키 생성
                            ref_key = f"{title}|{author}|{source}"
                            if ref_key in seen_refs:
                                continue
                            seen_refs.add(ref_key)
                            
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
                        
                        # rationale에 레퍼런스 추가 (레퍼런스가 있을 때만)
                        if reference_texts:
                            # 레퍼런스를 줄바꿈으로 구분하여 가독성 향상
                            references_section = " " + " ".join(reference_texts)
                            existing_rationale = coaching_data["challenge"].get("rationale", "")
                            
                            # 기존 rationale에서 이미 "참고:"가 포함되어 있는지 확인
                            if existing_rationale and "참고:" in existing_rationale:
                                # 이미 레퍼런스가 있으면 추가하지 않음 (LLM이 이미 생성했을 수 있음)
                                print(f"[VectorDB] rationale에 이미 레퍼런스가 포함되어 있음, 추가하지 않음")
                            elif existing_rationale:
                                # 기존 rationale 끝에 레퍼런스 추가
                                coaching_data["challenge"]["rationale"] = existing_rationale + references_section
                            else:
                                # rationale이 없으면 패턴 정보와 함께 생성 (레퍼런스와 함께)
                                most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                                # 패턴명 정규화 (공백 제거)
                                if most_frequent_pattern:
                                    normalized_pattern = most_frequent_pattern.replace(" ", "")
                                    pattern_info = f"이 챌린지는 '{normalized_pattern}' 패턴이 감지되어 생성되었습니다."
                                else:
                                    pattern_info = "이 챌린지는 분석 결과를 바탕으로 생성되었습니다."
                                coaching_data["challenge"]["rationale"] = pattern_info + references_section
                                print(f"[VectorDB] rationale 생성 완료 - 패턴: {most_frequent_pattern}, 레퍼런스: {len(reference_texts)}개")
                    else:
                        # 레퍼런스가 없을 때: rationale이 LLM이 생성했다면 유지, 없으면 비워두기
                        # LLM이 프롬프트 지시에 따라 rationale을 생성하지 않았을 가능성이 높음
                        # 만약 LLM이 rationale을 생성했다면 (레퍼런스 없이), 제거하여 할루시네이션 방지
                        if "rationale" in coaching_data["challenge"]:
                            rationale = coaching_data["challenge"].get("rationale", "")
                            # 레퍼런스 없이 생성된 rationale은 제거 (할루시네이션 방지)
                            # 단, 패턴 정보만 포함된 간단한 rationale은 유지 가능
                            if rationale and "참고:" not in rationale:
                                # 레퍼런스가 없는데 rationale이 있으면, 패턴 정보만 포함된 경우만 유지
                                most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                                if most_frequent_pattern and f"'{most_frequent_pattern}' 패턴이 감지되어" in rationale:
                                    # 패턴 정보만 포함된 간단한 rationale은 유지
                                    pass
                                else:
                                    # 복잡한 rationale은 제거 (할루시네이션 가능성)
                                    coaching_data["challenge"]["rationale"] = ""
                            elif not rationale:
                                # 빈 rationale은 그대로 유지
                                pass
                
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

