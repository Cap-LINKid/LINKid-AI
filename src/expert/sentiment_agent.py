from __future__ import annotations

from typing import Dict, Any

from src.utils.dpics import annotate_dialogue_dpics


def sentiment_label_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("[SentimentAgent] 감정 라벨링 시작")
    print("="*60)
    
    dialogue = state.get("message") or state.get("dialogue") or ""
    if not dialogue or not str(dialogue).strip():
        print("[SentimentAgent] 경고: 분석할 대화가 없음")
        print("="*60 + "\n")
        return {"annotated": ""}
    
    lines = len(str(dialogue).strip().splitlines())
    print(f"[SentimentAgent] 대화 라인 수: {lines}개")
    print("[SentimentAgent] DPICS 라벨링 중...")
    annotated = annotate_dialogue_dpics(str(dialogue))
    print("[SentimentAgent] 라벨링 완료")
    print("="*60 + "\n")
    return {"annotated": annotated}


if __name__ == "__main__":
    sample = {
        "message": "부모: 숙제 했니?\n아이: 하기 싫어.",
    }
    print(sentiment_label_node(sample))
