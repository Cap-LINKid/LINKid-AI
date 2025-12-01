from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm
from src.utils.vector_store import search_expert_advice
from src.utils.pattern_manager import (
    get_negative_pattern_names_normalized,
    normalize_pattern_name,
    is_pattern_negative
)


_KEYWORD_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert at generating search keywords for finding relevant parenting advice. "
            "Based on the dialogue summary, detected patterns, and key moments, generate 3-5 search keywords "
            "that would help find relevant expert advice for creating a coaching challenge. "
            "Return ONLY a JSON array of strings, each string being a search keyword. "
            "Keywords should be specific, relevant to the patterns and dialogue context, and suitable for semantic search. "
            "Example: [\"긍정기회놓치기 패턴 개선\", \"아동과의 긍정적 상호작용\", \"부모 코칭 기법\"]\n"
            "All keywords in Korean. No extra text, only JSON array."
        ),
    ),
    (
        "human",
        (
            "대화 요약:\n{summary}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "핵심 순간:\n{key_moments}\n\n"
            "위 정보를 바탕으로 전문가 조언 검색에 적합한 키워드 3-5개를 생성해주세요. "
            "각 키워드는 패턴, 대화 맥락, 개선 방향을 고려하여 구체적으로 작성해주세요."
        ),
    ),
])

_COACHING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a professional parenting coach. "
            "Create a personalized coaching plan based on:\n"
            "- 분석된 패턴 정보 (patterns)\n"
            "- 핵심 순간 및 이유 (key_moments)\n"
            "- 실제 대화 예시 (dialogue_examples)\n"
            "- VectorDB에서 가져온 전문가 레퍼런스와 조언 (expert_advice_section)\n\n"
            "Return ONLY a JSON object with the following structure:\n"
            "{{\n"
            '  \"summary\": \"요약 텍스트 (2-4문장)\",\n'
            '  \"challenge\": {{\n'
            '    \"title\": \"챌린지 제목 (가장 중요한 부정적 패턴 + 맥락을 드러내는 짧은 문구)\",\n'
            '    \"goal\": \"챌린지 목표 (1문장, 어떤 대화/행동을 어떻게 바꾸는지 명확히)\",\n'
            '    \"actions\": [\"액션1\", \"액션2\", \"액션3\"],\n'
            '    \"rationale\": \"챌린지 생성 이유 (필요한 경우에만, 패턴·대화·레퍼런스 근거 요약)\"\n'
            "  }},\n"
            '  \"qa_tips\": [\n'
            '    {{\"question\": \"질문1\", \"answer\": \"답변1\"}},\n'
            '    {{\"question\": \"질문2\", \"answer\": \"답변2\"}}\n'
            "  ]\n"
            "}}\n\n"
            "요구사항:\n"
            "1) 챌린지는 **가장 빈번하고 중요한 부정적 패턴**을 중심으로 만들 것.\n"
            "2) 각 action은 반드시 다음 세 가지를 동시에 반영해야 함:\n"
            "   - 패턴 정보 (예: '비판적 반응', '과도한 지시' 등)\n"
            "   - 실제 대화에서 반복된 구체 장면 (예: 용돈, 물건 뺏기, 고집 부림, 협박 발화 등)\n"
            "   - 전문가 레퍼런스에서 제시한 구체적 방법 (말문장 예시, 태도 변화, 훈육 원칙 등)\n"
            "3) summary에는 반드시 다음이 포함되어야 함:\n"
            "   - 왜 이 챌린지를 선택했는지 (어떤 패턴과 상황이 반복되었는지)\n"
            "   - 최소 1개 이상의 **구체적인 예시**:\n"
            "     - 실제 대화 문장 예시 1개 이상 (부모/아이 발화)\n"
            "       또는\n"
            "     - 전문가 레퍼런스 내용 요약 1개 이상 (어떤 조언/원칙을 따르는지)\n"
            "   예) \"부모가 아이에게 '너 때문에 힘들어'라고 반복해서 말하는 비판적 화법이 관찰되어, "
            "전문가 조언에서 제안한 '행동만 분리해서 지적하기' 원칙을 적용한 챌린지를 설계했습니다.\"\n"
            "4) rationale은 다음 조건에서만 포함:\n"
            "   - expert_advice_section에 실제 레퍼런스가 존재할 때\n"
            "   - 패턴·대화·레퍼런스 간 연결 근거를 1-2문장으로 정리할 때\n"
            "   레퍼런스 인용 시에는 LLM이 임의로 만들지 말고, expert_advice_section 안에 있는 **실제 제목/내용**만 사용할 것.\n"
            "5) QA tips는 다음을 다루도록 구성:\n"
            "   - 이 패턴을 가진 부모가 자주 할 법한 질문 2개 이상\n"
            "   - 각 질문에 대해, 대화 예시와 전문가 레퍼런스를 바탕으로 한 현실적인 답변\n"
            "6) 절대 새로운 이론·연구·저자를 상상해서 만들지 말 것. "
            "expert_advice_section에 없는 정보는 인용하지 말 것.\n"
            "7) 모든 텍스트는 한국어로 작성하고, JSON 이외의 추가 설명은 포함하지 말 것."
        ),
    ),
    (
        "human",
        (
            "대화 요약:\n{summary}\n\n"
            "스타일 분석:\n{style_analysis}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "핵심 순간:\n{key_moments}\n\n"
            "대표 대화 예시:\n{dialogue_examples}\n\n"
            "전문가 레퍼런스 및 조언 (VectorDB 검색 결과):\n{expert_advice_section}\n\n"
            "위 정보를 바탕으로 코칭 계획을 JSON 형식으로 작성해주세요.\n"
            "- 챌린지는 반드시 위 **부정적 패턴**과 **구체적인 대화 장면**에 직접 대응해야 합니다.\n"
            "- actions와 goal은 전문가 조언을 실제 부모-자녀 대화에 적용하는 형태로, "
            "구체적인 말문장/행동 수준까지 내려가서 작성해주세요.\n"
            "- summary에는 왜 이 챌린지를 생성했는지와, 대화 내용 또는 레퍼런스 내용 중 "
            "최소 한 가지 이상의 구체적인 예시를 반드시 포함하세요."
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
            
            # 부정적 패턴만 필터링 (중앙화된 함수 사용)
            if pattern_name and is_pattern_negative(pattern_name, pattern_type, severity):
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + occurrences
    
    # patterns에서도 부정적 패턴만 계산
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        
        # 부정적 패턴만 필터링 (중앙화된 함수 사용)
        if pattern_name and is_pattern_negative(pattern_name, pattern_type, severity):
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
    
    if not pattern_counts:
        return None
    
    # 가장 빈번한 부정적 패턴 반환
    most_frequent = max(pattern_counts.items(), key=lambda x: x[1])[0]
    return most_frequent


def _get_negative_patterns(patterns: List[Dict[str, Any]], key_moments: Dict[str, Any]) -> List[str]:
    """
    부정적 패턴 이름 목록 반환 (정규화된 이름)
    """
    pattern_names = []
    
    # key_moments에서 부정적 패턴 찾기
    if isinstance(key_moments, dict):
        pattern_examples = key_moments.get("pattern_examples", [])
        for p in pattern_examples:
            pattern_name = p.get("pattern_name", "")
            pattern_type = p.get("pattern_type", "")
            severity = p.get("severity", "")
            
            # 부정적 패턴만 필터링 (중앙화된 함수 사용)
            if pattern_name and is_pattern_negative(pattern_name, pattern_type, severity):
                normalized = normalize_pattern_name(pattern_name)
                if normalized not in pattern_names:
                    pattern_names.append(normalized)
    
    # patterns에서도 부정적 패턴 찾기
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        
        # 부정적 패턴만 필터링 (중앙화된 함수 사용)
        if pattern_name and is_pattern_negative(pattern_name, pattern_type, severity):
            normalized = normalize_pattern_name(pattern_name)
            if normalized not in pattern_names:
                pattern_names.append(normalized)
    
    return pattern_names


def _generate_search_keywords(
    summary: str,
    patterns_str: str,
    key_moments_str: str,
    llm
) -> List[str]:
    """
    LLM을 사용하여 검색 키워드 생성
    
    Args:
        summary: 대화 요약
        patterns_str: 패턴 정보 문자열
        key_moments_str: 핵심 순간 문자열
        llm: LLM 인스턴스
    
    Returns:
        검색 키워드 리스트
    """
    try:
        res = (_KEYWORD_GENERATION_PROMPT | llm).invoke({
            "summary": summary,
            "patterns": patterns_str,
            "key_moments": key_moments_str,
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 배열 파싱
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            keywords = json.loads(json_match.group(0))
            if isinstance(keywords, list):
                # 문자열 리스트로 변환 및 필터링
                keywords = [str(k).strip() for k in keywords if k and str(k).strip()]
                print(f"[VectorDB] 생성된 검색 키워드: {keywords}")
                return keywords
        
        print(f"[VectorDB] 키워드 생성 실패, 기본 패턴명 사용")
        return []
    except Exception as e:
        print(f"[VectorDB] 키워드 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


def coaching_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑧ coaching_plan: 코칭/실천 계획 (LLM)
    """
    summary = state.get("summary") or ""
    style_analysis = state.get("style_analysis") or {}
    patterns = state.get("patterns") or []
    key_moments = state.get("key_moments") or {}
    utterances = state.get("utterances_ko") or []
    
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
    
    # 부정적 패턴만 필터링 (중앙화된 함수 사용)
    negative_pattern_examples = []
    for p in pattern_examples:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        if is_pattern_negative(pattern_name, pattern_type, severity):
            negative_pattern_examples.append(p)
    
    negative_patterns = []
    for p in patterns:
        pattern_name = p.get("pattern_name", "")
        pattern_type = p.get("pattern_type", "")
        severity = p.get("severity", "")
        if is_pattern_negative(pattern_name, pattern_type, severity):
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

    # 대표 대화 예시 구성 (실제 발화를 최대 N개까지 포함)
    dialogue_examples_str = ""
    if isinstance(utterances, list) and utterances:
        max_utterances = 8
        formatted_utterances = []
        for u in utterances[:max_utterances]:
            speaker = u.get("speaker", "")
            text = u.get("text", "")
            if text:
                formatted_utterances.append(f"{speaker}: {text}")
        dialogue_examples_str = "\n".join(formatted_utterances)
    else:
        dialogue_examples_str = "(없음)"
    
    # VectorDB 검색 (챌린지 생성용) - LLM으로 키워드 생성 후 검색
    expert_advice_section = ""
    expert_references_list = []
    all_expert_advice = []  # 모든 검색 결과 수집
    
    # USE_VECTOR_DB가 명시적으로 false가 아니면 검색 시도 (기본값: true)
    use_vector_db_env = os.getenv("USE_VECTOR_DB", "true").lower()
    use_vector_db = use_vector_db_env != "false"
    
    if use_vector_db:
        try:
            # 부정적 패턴 목록 가져오기
            negative_patterns = _get_negative_patterns(patterns, key_moments)
            print(f"[VectorDB] 챌린지 생성 검색 시작 - 패턴 목록: {negative_patterns}")
            
            # LLM으로 검색 키워드 생성
            search_keywords = _generate_search_keywords(
                summary=summary,
                patterns_str=patterns_str,
                key_moments_str=key_moments_str,
                llm=llm
            )
            
            # 패턴명도 키워드에 추가 (키워드가 없거나 부족한 경우를 대비)
            if negative_patterns:
                for pattern_name in negative_patterns:
                    normalized_pattern = normalize_pattern_name(pattern_name)
                    if normalized_pattern not in search_keywords:
                        search_keywords.append(normalized_pattern)
            
            if search_keywords:
                print(f"[VectorDB] 총 {len(search_keywords)}개 키워드로 검색 시작")
                
                # 각 키워드로 검색 수행
                seen_advice_ids = set()  # 중복 제거용 (ID 기준)
                for keyword in search_keywords:
                    print(f"[VectorDB] 키워드 '{keyword}' 검색 중...")
                    
                    # 부정적 패턴이 있으면 pattern_names 필터 추가
                    # filters = {
                    #     "advice_type": ["pattern_advice", "coaching", "challenge_guide"]
                    # }
                    # if negative_patterns:
                    #     filters["pattern_names"] = negative_patterns
                    
                    expert_advice = search_expert_advice(
                        query=keyword,
                        top_k=int(os.getenv("VECTOR_SEARCH_TOP_K_CHALLENGE", "3")),  # 키워드당 3개
                        threshold=float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.1"))
                        # filters=filters
                    )
                    
                    if expert_advice:
                        print(f"[VectorDB] 키워드 '{keyword}' 검색 결과: {len(expert_advice)}개")
                        for advice in expert_advice:
                            # ID로 중복 제거 (ID가 없으면 title로)
                            advice_id = advice.get("id") or advice.get("title", "")
                            if advice_id and advice_id not in seen_advice_ids:
                                seen_advice_ids.add(advice_id)
                                all_expert_advice.append(advice)
                                title = advice.get('title', '')
                                author = advice.get('author', '')
                                source = advice.get('source', '')
                                print(f"  - [{advice['advice_type']}] {title} (유사도: {advice.get('relevance_score', 0):.3f})")
                    else:
                        print(f"[VectorDB] 키워드 '{keyword}' 검색 결과 없음")
                
                # 모든 검색 결과를 프롬프트에 포함
                if all_expert_advice:
                    # 유사도 점수 기준으로 정렬 (내림차순)
                    all_expert_advice.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                    
                    # 상위 결과만 선택 (너무 많으면 제한)
                    max_results = int(os.getenv("VECTOR_SEARCH_MAX_RESULTS", "15"))
                    selected_advice = all_expert_advice[:max_results]
                    
                    print(f"[VectorDB] 총 {len(all_expert_advice)}개 결과 중 상위 {len(selected_advice)}개 선택")
                    
                    # 전문가 조언 섹션 구성
                    expert_advice_sections = []
                    for advice in selected_advice:
                        advice_section = f"- [{advice['advice_type']}] {advice['title']}\n  {advice['content'][:400]}..."
                        expert_advice_sections.append(advice_section)
                    
                    expert_advice_section = "전문가 조언 및 챌린지 가이드:\n" + "\n".join(expert_advice_sections)
                    
                    # 레퍼런스 리스트 구성 (중복 제거)
                    seen_titles = set()
                    expert_references_list = []
                    for advice in selected_advice:
                        # 안전하게 데이터 추출 (None 체크 포함)
                        # 주의: VectorDB 스키마에서는 DB 컬럼명이 reference 이지만,
                        # search_expert_advice 결과에서는 source 필드에 매핑되어 들어온다.
                        title = advice.get("title") or ""
                        reference = advice.get("source") or ""
                        
                        # 제목으로 중복 체크
                        if title and title.strip() and title not in seen_titles:
                            seen_titles.add(title)
                            ref_data = {
                                "title": title.strip(),
                                "reference": reference.strip() if reference else "",
                                "type": advice.get("advice_type", "")
                            }
                            expert_references_list.append(ref_data)
                            print(f"[VectorDB] 레퍼런스 추가: title='{ref_data['title']}', reference='{ref_data['reference']}'")
                    
                    print(f"[VectorDB] 전문가 조언 섹션 생성 완료 - {len(expert_references_list)}개 레퍼런스 (중복 제거 후)")
                else:
                    print(f"[VectorDB] 모든 키워드 검색 결과 없음")
            else:
                print(f"[VectorDB] 검색 키워드가 생성되지 않아 검색하지 않음")
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
            "dialogue_examples": dialogue_examples_str,
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
                    
                    # rationale 필드 처리: 레퍼런스가 있을 때만 간단하게 추가
                    if expert_references_list:
                        # 중복 제거 후 최대 2개의 레퍼런스만 사용
                        seen_refs = set()
                        selected_refs = []
                        for ref in expert_references_list:
                            title = (ref.get("title") or "").strip()
                            reference = (ref.get("reference") or "").strip()
                            if not title:
                                continue
                            ref_key = f"{title}|{reference}"
                            if ref_key in seen_refs:
                                continue
                            seen_refs.add(ref_key)
                            selected_refs.append({"title": title, "reference": reference})
                            if len(selected_refs) >= 2:
                                break

                        reference_texts = []
                        for ref in selected_refs:
                            title = ref["title"]
                            reference = ref["reference"]
                            if reference:
                                ref_text = f"'{title}' ({reference})"
                            else:
                                ref_text = f"'{title}'"

                            reference_texts.append(ref_text)

                        if reference_texts:
                            # [참고] 뒤에 최대 2개까지 붙이기
                            references_section = " [참고] " + "; ".join(reference_texts)
                            existing_rationale = coaching_data["challenge"].get("rationale", "") or ""

                            if existing_rationale.strip():
                                coaching_data["challenge"]["rationale"] = existing_rationale.strip() + references_section
                            else:
                                most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                                if most_frequent_pattern:
                                    normalized_pattern = normalize_pattern_name(most_frequent_pattern)
                                    pattern_info = f"이 챌린지는 '{normalized_pattern}' 패턴이 감지되어 생성되었습니다."
                                else:
                                    pattern_info = "이 챌린지는 분석 결과를 바탕으로 생성되었습니다."
                                coaching_data["challenge"]["rationale"] = pattern_info + references_section
                    else:
                        # 레퍼런스가 없으면 기존 rationale이 있더라도 그대로 두되,
                        # LLM이 임의로 생성했을 수 있으므로 추가 가공은 하지 않는다.
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

