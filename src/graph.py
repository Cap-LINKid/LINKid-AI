from __future__ import annotations

from dotenv import load_dotenv
from typing import Dict, Any

from src.router.router import build_question_router
from src.vs.ddl import get_tdl

# Exported graph object for LangGraph Dev UI
_compiled_graph = build_question_router()

# LangGraph UI에서 사용할 graph 객체
# LangGraph UI는 전체 state를 반환하므로, result 필드에 최종 결과가 들어있음
graph = _compiled_graph


def run(message: str) -> Dict[str, Any]:
    load_dotenv()
    state = {"message": message, "tdl": get_tdl()}
    result = graph.invoke(state)
    return result


if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) or "샘플 메시지"
    print(run(msg))
