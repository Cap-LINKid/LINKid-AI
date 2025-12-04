from __future__ import annotations

from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a pediatric and parenting coach."
        " Use the provided DPICS-style annotations and extracted highlights to understand interaction functions."
        " Return concise, empathetic, and actionable advice suitable for caregivers in Korean."
        " Include: 1) 핵심 이슈, 2) 공감 멘트, 3) 즉시 실행 팁(3개), 4) 주의사항, 5) 하이라이트별 1문장 피드백."
        " Reference specific highlights in your advice and for each highlight provide one-sentence feedback that directly quotes or paraphrases the utterance."
    )),
    ("human", "원문 대화:\n{dialogue}\n\n라인별 DPICS 라벨링:\n{annotated}\n\n발화 하이라이트(원문):\n{highlights_str}\n\n추가 맥락(선택): {context}")
])


def _format_highlights(highlights: List[str]) -> str:
    if not highlights:
        return "(없음)"
    return "\n".join(f"- {h}" for h in highlights[:10])


def parenting_advice_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("[ExpertAgent] 전문가 조언 생성 시작")
    print("="*60)
    
    dialogue = state.get("message") or state.get("dialogue") or ""
    context = state.get("context") or ""
    if not dialogue.strip():
        print("[ExpertAgent] 경고: 대화 내용이 비어있음")
        print("="*60 + "\n")
        return {"advice": "대화 내용이 비어있어요. 부모-아이 발화를 함께 제공해주세요."}

    # DPICS 라벨은 sentiment_label_node가 생성한 annotated 사용
    annotated = state.get("annotated") or ""
    highlights = state.get("highlights") or []
    highlights_str = _format_highlights(highlights)
    
    print(f"[ExpertAgent] 입력 데이터 확인:")
    print(f"  - 대화 길이: {len(dialogue)} 문자")
    print(f"  - 하이라이트: {len(highlights)}개")
    print(f"  - DPICS 라벨: {'있음' if annotated else '없음'}")

    print("[ExpertAgent] LLM으로 조언 생성 중...")
    llm = get_llm(mini=False)
    chain = PROMPT | llm

    msgs = PROMPT.format_prompt(
        dialogue=dialogue,
        annotated=annotated,
        highlights_str=highlights_str,
        context=context,
    ).to_messages()
    printable = []
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = getattr(m, "content", "")
        printable.append(f"[{role}]\n{content}")
    print("\n=== Parenting Advice Prompt ===\n" + "\n\n".join(printable) + "\n=== END ===\n")

    res = chain.invoke({
        "dialogue": dialogue,
        "annotated": annotated,
        "highlights_str": highlights_str,
        "context": context,
    })
    advice = getattr(res, "content", str(res))
    print(f"[ExpertAgent] 조언 생성 완료 (길이: {len(advice)} 문자)")
    print("="*60 + "\n")
    return {"advice": advice}


if __name__ == "__main__":
    sample_text = "\n".join([
        "부모: 숙제 했니?",
        "아이: 하기 싫어.",
        "부모: 왜?",
        "아이: 너무 어려워.",
    ])
    s = {"message": sample_text}
    print(parenting_advice_node(s))
