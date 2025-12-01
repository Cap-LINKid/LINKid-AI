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
            "IMPORTANT: When expert_advice_section is provided, you MUST use the specific advice and strategies from it to create the challenge. "
            "The challenge goal, actions, and rationale should be based on the expert advice provided. "
            "If expert_advice_section contains pattern-specific advice, prioritize using that advice for the corresponding pattern. "
            "The rationale field should ONLY be included if expert advice references are provided in the expert_advice_section. "
            "If expert_advice_section is empty or no references are available, do NOT create a rationale field (leave it empty or omit it). "
            "When expert advice is available, include references in the format: '참고: [제목] ([저자], [출처])'. "
            "IMPORTANT: Replace [제목], [저자], [출처] with actual values from expert_advice_section. "
            "Do NOT use literal text '[저자]' or '[출처]' - use the actual author name and source. "
            "If author or source information is not available, omit that part (e.g., '참고: [제목]' or '참고: [제목] ([저자])'). "
            "Do NOT make up or hallucinate references. Only use references that are explicitly provided in the expert_advice_section. "
            "QA tips should address common questions about the pattern based on the expert advice. "
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
                        title = advice.get("title") or ""
                        source = advice.get("source") or ""
                        author = advice.get("author") or ""
                        
                        # 제목으로 중복 체크
                        if title and title.strip() and title not in seen_titles:
                            seen_titles.add(title)
                            ref_data = {
                                "title": title.strip(),
                                "source": source.strip() if source else "",
                                "author": author.strip() if author else "",
                                "type": advice.get("advice_type", "")
                            }
                            expert_references_list.append(ref_data)
                            print(f"[VectorDB] 레퍼런스 추가: title='{ref_data['title']}', author='{ref_data['author']}', source='{ref_data['source']}'")
                    
                    print(f"[VectorDB] 전문가 조언 섹션 생성 완료 - {len(expert_references_list)}개 레퍼런스 (중복 제거 후)")
                    if expert_references_list:
                        print(f"[VectorDB] 레퍼런스 상세 정보:")
                        for i, ref in enumerate(expert_references_list, 1):
                            print(f"  {i}. title='{ref['title']}', author='{ref['author']}', source='{ref['source']}'")
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
                            author = ref.get("author", "") or ""
                            source = ref.get("source", "") or ""
                            title = ref.get("title", "") or ""
                            
                            # 디버깅: 실제 값 확인
                            print(f"[VectorDB] 레퍼런스 텍스트 생성 - title='{title}', author='{author}', source='{source}'")
                            
                            if not title:
                                print(f"[VectorDB] 경고: 제목이 없는 레퍼런스 건너뜀")
                                continue
                            
                            # 중복 체크용 키 생성
                            ref_key = f"{title}|{author}|{source}"
                            if ref_key in seen_refs:
                                continue
                            seen_refs.add(ref_key)
                            
                            # "참고: [제목] ([저자], [출처])" 형식
                            # author와 source가 실제로 값이 있는지 확인 (빈 문자열이 아닌지)
                            author_has_value = author and author.strip()
                            source_has_value = source and source.strip()
                            
                            if author_has_value and source_has_value:
                                ref_text = f"참고: '{title}' ({author}, {source})"
                            elif author_has_value:
                                ref_text = f"참고: '{title}' ({author})"
                            elif source_has_value:
                                ref_text = f"참고: '{title}' ({source})"
                            else:
                                ref_text = f"참고: '{title}'"
                            
                            print(f"[VectorDB] 생성된 레퍼런스 텍스트: '{ref_text}'")
                            reference_texts.append(ref_text)
                        
                        # rationale에 레퍼런스 추가 (레퍼런스가 있을 때만)
                        if reference_texts:
                            # 레퍼런스를 줄바꿈으로 구분하여 가독성 향상
                            references_section = " " + " ".join(reference_texts)
                            existing_rationale = coaching_data["challenge"].get("rationale", "")
                            
                            # 기존 rationale에서 이미 "참고:"가 포함되어 있는지 확인
                            if existing_rationale and "참고:" in existing_rationale:
                                # LLM이 생성한 레퍼런스가 불완전한지 확인 (저자/출처가 없는지)
                                # "참고: [제목]" 뒤에 "("가 없으면 불완전한 것으로 판단
                                import re as regex_module
                                
                                # "참고: [제목]" 패턴 찾기 (괄호가 없는 경우)
                                for ref in expert_references_list:
                                    title = ref.get("title", "").strip()
                                    author = ref.get("author", "").strip()
                                    source = ref.get("source", "").strip()
                                    
                                    if not title:
                                        continue
                                    
                                    # "참고: [제목]" 패턴이 있고, 저자/출처가 있는데 rationale에 포함되어 있지 않으면 추가
                                    incomplete_pattern = f"참고: {title}"
                                    if incomplete_pattern in existing_rationale and (author or source):
                                        # 저자/출처가 이미 포함되어 있는지 확인
                                        if not (author in existing_rationale or source in existing_rationale):
                                            # "참고: [제목]" 뒤에 저자/출처 추가
                                            if author and source:
                                                complete_ref = f" ({author}, {source})"
                                            elif author:
                                                complete_ref = f" ({author})"
                                            elif source:
                                                complete_ref = f" ({source})"
                                            else:
                                                complete_ref = ""
                                            
                                            if complete_ref:
                                                existing_rationale = existing_rationale.replace(incomplete_pattern, incomplete_pattern + complete_ref)
                                                print(f"[VectorDB] 레퍼런스에 저자/출처 추가: '{incomplete_pattern}' -> '{incomplete_pattern}{complete_ref}'")
                                                break
                                
                                coaching_data["challenge"]["rationale"] = existing_rationale
                                print(f"[VectorDB] rationale의 레퍼런스에 저자/출처 추가 완료")
                            elif existing_rationale:
                                # 기존 rationale 끝에 레퍼런스 추가
                                coaching_data["challenge"]["rationale"] = existing_rationale + references_section
                            else:
                                # rationale이 없으면 패턴 정보와 함께 생성 (레퍼런스와 함께)
                                most_frequent_pattern = _find_most_frequent_pattern(patterns, key_moments)
                                # 패턴명 정규화 (공백 제거)
                                if most_frequent_pattern:
                                    normalized_pattern = normalize_pattern_name(most_frequent_pattern)
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

