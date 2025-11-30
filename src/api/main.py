from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.status import (
    create_execution, 
    update_node_status, 
    update_execution_status, 
    get_execution_status,
    get_all_executions,
    ExecutionStatus,
    NodeStatus
)
from src.router.router import build_question_router

app = FastAPI(
    title="LinkID AI API",
    description="부모-아이 대화 분석 API",
    version="1.0.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 에러 핸들러 - 더 명확한 에러 메시지 제공"""
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "요청 데이터 검증 실패",
            "errors": errors
        }
    )

# 그래프 인스턴스
graph = build_question_router()


# Request/Response 모델
class UtteranceItem(BaseModel):
    speaker: str = Field(..., description="발화자 (예: 'A', 'B', 'parent', 'child')")
    text: str = Field(..., description="발화 내용")
    timestamp: Optional[int] = Field(None, description="타임스탬프 (밀리초, 선택적)")


class ActionItem(BaseModel):
    action_id: str = Field(..., description="액션 ID")
    content: str = Field(..., description="액션 내용")


class ChallengeSpec(BaseModel):
    challenge_id: Optional[str] = Field(None, description="챌린지 ID")
    title: str = Field(..., description="챌린지 제목")
    goal: str = Field(..., description="챌린지 목표")
    actions: List[ActionItem] = Field(..., description="액션 리스트")


class DialogueRequest(BaseModel):
    utterances_ko: List[UtteranceItem] = Field(..., description="한국어 대화 발화 리스트")
    challenge_spec: Optional[Dict[str, Any]] = None  # 하위 호환성: 단일 챌린지
    challenge_specs: Optional[List[ChallengeSpec]] = None  # 여러 챌린지 (새로운 형식)
    meta: Optional[Dict[str, Any]] = None


class ExecutionResponse(BaseModel):
    execution_id: str
    status: str  # analysis_status
    message: str  # status_message
    progress_percentage: Optional[int] = None


def _run_analysis_with_status(execution_id: str, state: Dict[str, Any]):
    """상태 추적하며 분석 실행"""
    try:
        update_execution_status(execution_id, ExecutionStatus.RUNNING)
        
        final_result = None
        
        # LangGraph의 astream을 사용하여 각 노드 실행 추적
        try:
            import asyncio
            
            async def run_with_tracking():
                nonlocal final_result
                current_state = state
                
                # 노드 실행 순서 정의
                sequential_nodes = ["preprocess", "translate_ko_to_en", "label_utterances", "detect_patterns"]
                parallel_nodes = ["summarize", "key_moments", "analyze_style", "coaching_plan", "challenge_eval", "summary_diagnosis"]
                
                # 첫 번째 노드를 running으로 표시
                if sequential_nodes:
                    update_node_status(execution_id, sequential_nodes[0], NodeStatus.RUNNING)
                
                # astream으로 각 노드 실행 추적
                # astream은 각 노드 실행 후 {node_name: output} 형식의 딕셔너리를 반환
                async for node_output in graph.astream(current_state):
                    if isinstance(node_output, dict):
                        for node_name in node_output.keys():
                            if node_name in sequential_nodes + parallel_nodes + ["aggregate_result"]:
                                # 노드 완료 표시
                                update_node_status(execution_id, node_name, NodeStatus.COMPLETED)
                                
                                # 다음 노드를 running으로 표시
                                if node_name in sequential_nodes:
                                    idx = sequential_nodes.index(node_name)
                                    if idx + 1 < len(sequential_nodes):
                                        # 다음 순차 노드
                                        update_node_status(execution_id, sequential_nodes[idx + 1], NodeStatus.RUNNING)
                                    elif idx + 1 == len(sequential_nodes):
                                        # 병렬 노드 시작
                                        for p_node in parallel_nodes:
                                            update_node_status(execution_id, p_node, NodeStatus.RUNNING)
                                elif node_name in parallel_nodes:
                                    # 병렬 노드가 모두 완료되었는지 확인
                                    completed_parallel = sum(1 for p_node in parallel_nodes 
                                                           if get_execution_status(execution_id)["nodes"][p_node]["status"] == NodeStatus.COMPLETED)
                                    if completed_parallel == len(parallel_nodes):
                                        # 모든 병렬 노드 완료, 집계 노드 시작
                                        update_node_status(execution_id, "aggregate_result", NodeStatus.RUNNING)
                                
                                if node_name == "aggregate_result":
                                    final_result = node_output.get(node_name, {}).get("result")
                        current_state = {**current_state, **node_output}
                
                return final_result
            
            # 비동기 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_with_tracking())
                if result:
                    final_result = result
            finally:
                loop.close()
                
        except Exception as stream_error:
            # 스트리밍 실패 시 일반 invoke 사용
            print(f"Streaming failed, using invoke: {stream_error}")
            try:
                result = graph.invoke(state)
                final_result = result.get("result")
                # 모든 노드를 완료로 표시
                for node_name in ["preprocess", "translate_ko_to_en", "label_utterances", 
                                 "detect_patterns", "summarize", "key_moments", "analyze_style",
                                 "coaching_plan", "challenge_eval", "summary_diagnosis", "aggregate_result"]:
                    update_node_status(execution_id, node_name, NodeStatus.COMPLETED)
            except Exception as invoke_error:
                update_execution_status(execution_id, ExecutionStatus.FAILED, error=str(invoke_error))
                return
        
        # 완료
        if final_result:
            update_execution_status(execution_id, ExecutionStatus.COMPLETED, result=final_result)
        else:
            update_execution_status(execution_id, ExecutionStatus.FAILED, error="No result returned")
            
    except Exception as e:
        update_execution_status(execution_id, ExecutionStatus.FAILED, error=str(e))


@app.post("/analyze", response_model=ExecutionResponse)
async def analyze_dialogue(request: DialogueRequest, background_tasks: BackgroundTasks):
    """대화 분석 실행 (비동기)"""
    # challenge_specs 우선, 없으면 challenge_spec을 리스트로 변환
    challenge_specs = request.challenge_specs
    if challenge_specs is None:
        if request.challenge_spec:
            challenge_specs = [request.challenge_spec]
        else:
            challenge_specs = []
    
    # Pydantic 모델을 딕셔너리로 변환
    utterances_ko_dict = [
        {
            "speaker": utt.speaker,
            "text": utt.text,
            **({"timestamp": utt.timestamp} if utt.timestamp is not None else {})
        }
        for utt in request.utterances_ko
    ]
    
    # ChallengeSpec 모델을 딕셔너리로 변환
    challenge_specs_dict = []
    for spec in challenge_specs:
        if isinstance(spec, ChallengeSpec):
            # Pydantic 모델인 경우 딕셔너리로 변환
            spec_dict = spec.model_dump()
            # actions를 딕셔너리 리스트로 변환
            spec_dict["actions"] = [
                {"action_id": action.action_id, "content": action.content}
                for action in spec.actions
            ]
            challenge_specs_dict.append(spec_dict)
        else:
            # 이미 딕셔너리인 경우 (하위 호환성)
            challenge_specs_dict.append(spec)
    
    state = {
        "utterances_ko": utterances_ko_dict,
        "challenge_specs": challenge_specs_dict,
        "challenge_spec": request.challenge_spec or {},  # 하위 호환성 유지
        "meta": request.meta or {}
    }
    
    execution_id = create_execution(state)
    update_execution_status(execution_id, ExecutionStatus.RUNNING)
    
    # 백그라운드에서 실행
    background_tasks.add_task(_run_analysis_with_status, execution_id, state)
    
    # 초기 상태 정보 가져오기
    status_info = get_execution_status(execution_id)
    
    return ExecutionResponse(
        execution_id=execution_id,
        status=status_info.get("analysis_status", "pending"),
        message=status_info.get("status_message", "분석이 시작되었습니다."),
        progress_percentage=status_info.get("progress_percentage", 0)
    )


@app.get("/status/{execution_id}")
async def get_status(execution_id: str):
    """실행 상태 조회"""
    status = get_execution_status(execution_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # nodes 필드 제거 (상세 노드 정보는 불필요)
    response = {k: v for k, v in status.items() if k != "nodes"}
    return response


@app.get("/executions")
async def list_executions():
    """모든 실행 목록 조회"""
    all_executions = get_all_executions()
    # nodes 필드 제거
    executions_list = [
        {k: v for k, v in exec_data.items() if k != "nodes"}
        for exec_data in all_executions.values()
    ]
    return {
        "executions": executions_list
    }


@app.get("/")
async def root():
    """API 루트"""
    return {
        "service": "LinkID AI API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze",
            "status": "GET /status/{execution_id}",
            "executions": "GET /executions"
        }
    }

