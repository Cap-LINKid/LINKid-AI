from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.router.states import RouterState

# 새로운 플로우 에이전트들
from src.expert.preprocess_agent import preprocess_node
from src.expert.translate_agent import translate_ko_to_en_node
from src.expert.label_agent import label_utterances_node
from src.expert.pattern_agent import detect_patterns_node
from src.expert.summarize_agent import summarize_node
from src.expert.key_moments_agent import key_moments_node
from src.expert.style_agent import analyze_style_node
from src.expert.coaching_agent import coaching_plan_node
from src.expert.challenge_agent import challenge_eval_node
from src.expert.summary_diagnosis_agent import summary_diagnosis_node
from src.expert.aggregate_agent import aggregate_result_node

# 기존 에이전트들 (하위 호환성)
from src.expert.sentiment_agent import sentiment_label_node
from src.expert.highlight_agent import highlight_extract_node
from src.expert.expert_agent import parenting_advice_node


def build_question_router():
    """
    새로운 대화 분석 파이프라인:
    ① preprocess → ② translate → ③ label → ④ detect_patterns
    → ⑤-⑨ 병렬 분석 (5개) → ⑩ aggregate_result
    """
    graph = StateGraph(RouterState)

    # 순차 처리 단계
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("translate_ko_to_en", translate_ko_to_en_node)
    graph.add_node("label_utterances", label_utterances_node)
    graph.add_node("detect_patterns", detect_patterns_node)

    # 병렬 분석 단계
    graph.add_node("summarize", summarize_node)
    graph.add_node("key_moments", key_moments_node)
    graph.add_node("analyze_style", analyze_style_node)
    graph.add_node("coaching_plan", coaching_plan_node)
    graph.add_node("challenge_eval", challenge_eval_node)
    graph.add_node("summary_diagnosis", summary_diagnosis_node)

    # 최종 집계
    graph.add_node("aggregate_result", aggregate_result_node)

    # 엣지 구성: 순차 → 병렬 → 집계
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "translate_ko_to_en")
    graph.add_edge("translate_ko_to_en", "label_utterances")
    graph.add_edge("label_utterances", "detect_patterns")

    # detect_patterns 이후 병렬 실행
    graph.add_edge("detect_patterns", "summarize")
    graph.add_edge("detect_patterns", "key_moments")
    graph.add_edge("detect_patterns", "analyze_style")
    graph.add_edge("detect_patterns", "coaching_plan")
    graph.add_edge("detect_patterns", "challenge_eval")
    graph.add_edge("detect_patterns", "summary_diagnosis")

    # 모든 병렬 분석이 완료되면 집계
    graph.add_edge("summarize", "aggregate_result")
    graph.add_edge("key_moments", "aggregate_result")
    graph.add_edge("analyze_style", "aggregate_result")
    graph.add_edge("coaching_plan", "aggregate_result")
    graph.add_edge("challenge_eval", "aggregate_result")
    graph.add_edge("summary_diagnosis", "aggregate_result")

    graph.add_edge("aggregate_result", END)

    return graph.compile()


def build_legacy_router():
    """
    기존 라우터 (하위 호환성)
    """
    graph = StateGraph(RouterState)

    graph.add_node("sentiment_labeler", sentiment_label_node)
    graph.add_node("highlight_extractor", highlight_extract_node)
    graph.add_node("parenting_advice", parenting_advice_node)

    # parallel: both start from START
    graph.add_edge(START, "sentiment_labeler")
    graph.add_edge(START, "highlight_extractor")

    # converge into parenting_advice; it runs after both predecessors complete
    graph.add_edge("sentiment_labeler", "parenting_advice")
    graph.add_edge("highlight_extractor", "parenting_advice")
    graph.add_edge("parenting_advice", END)

    return graph.compile()
