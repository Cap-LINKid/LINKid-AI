from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from src.api.storage import get_storage_instance


class AnalysisStatus:
    """분석 진행 상태 (이름, 진행률, 메시지)"""
    TRANSLATING = ("translating", 10, "대화를 번역하는 중입니다")
    LABELING = ("labeling", 30, "발화를 분류하는 중입니다")
    PATTERN_DETECTION = ("pattern_detection", 50, "패턴을 감지하는 중입니다")
    STYLE_ANALYSIS = ("style_analysis", 70, "스타일을 분석하는 중입니다")
    COACHING_GENERATION = ("coaching_generation", 85, "코칭을 생성하는 중입니다")
    REPORT_FINALIZING = ("report_finalizing", 95, "리포트를 작성하는 중입니다")
    COMPLETED = ("completed", 100, "분석이 완료되었습니다")
    FAILED = ("failed", 0, "분석 중 오류가 발생했습니다")
    
    @classmethod
    def get_status_info(cls, status_name: str) -> Tuple[str, int, str]:
        """상태 이름으로 정보 조회"""
        status_map = {
            "translating": cls.TRANSLATING,
            "labeling": cls.LABELING,
            "pattern_detection": cls.PATTERN_DETECTION,
            "style_analysis": cls.STYLE_ANALYSIS,
            "coaching_generation": cls.COACHING_GENERATION,
            "report_finalizing": cls.REPORT_FINALIZING,
            "completed": cls.COMPLETED,
            "failed": cls.FAILED,
        }
        return status_map.get(status_name, cls.FAILED)


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStatus(str, Enum):
    """노드 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def create_execution(initial_state: Dict[str, Any]) -> str:
    """새 실행 생성 및 ID 반환"""
    execution_id = str(uuid.uuid4())
    storage = get_storage_instance()
    
    execution_data = {
        "execution_id": execution_id,
        "status": ExecutionStatus.PENDING,
        "analysis_status": "pending",  # AnalysisStatus의 상태 이름
        "progress_percentage": 0,
        "status_message": "분석을 준비하는 중입니다",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "nodes": {
            "preprocess": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "translate_ko_to_en": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "label_utterances": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "detect_patterns": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "summarize": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "key_moments": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "analyze_style": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "coaching_plan": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "challenge_eval": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
            "aggregate_result": {"status": NodeStatus.PENDING, "started_at": None, "completed_at": None},
        },
        "current_node": None,
        "error": None,
        "result": None
    }
    
    storage.create_execution(execution_id, execution_data)
    return execution_id


def update_analysis_status(execution_id: str, status_name: str):
    """분석 상태 업데이트"""
    storage = get_storage_instance()
    exec_state = storage.get_execution(execution_id)
    
    if not exec_state:
        return
    
    name, progress, message = AnalysisStatus.get_status_info(status_name)
    
    updates = {
        "analysis_status": name,
        "progress_percentage": progress,
        "status_message": message,
        "updated_at": datetime.now().isoformat()
    }
    
    storage.update_execution(execution_id, updates)


def update_node_status(execution_id: str, node_name: str, status: NodeStatus, error: Optional[str] = None):
    """노드 상태 업데이트 및 분석 상태 자동 업데이트"""
    storage = get_storage_instance()
    exec_state = storage.get_execution(execution_id)
    
    if not exec_state:
        return
    
    node_state = exec_state["nodes"].get(node_name)
    
    if node_state:
        node_state["status"] = status
        if status == NodeStatus.RUNNING and not node_state["started_at"]:
            node_state["started_at"] = datetime.now().isoformat()
            # 노드 시작 시 분석 상태 업데이트
            _update_status_by_node(execution_id, node_name, is_start=True)
        elif status in [NodeStatus.COMPLETED, NodeStatus.FAILED]:
            node_state["completed_at"] = datetime.now().isoformat()
            if error:
                node_state["error"] = error
            # 노드 완료 시 분석 상태 업데이트
            if status == NodeStatus.COMPLETED:
                _update_status_by_node(execution_id, node_name, is_start=False)
        
        updates = {
            "nodes": exec_state["nodes"],
            "updated_at": datetime.now().isoformat(),
            "current_node": node_name if status == NodeStatus.RUNNING else None
        }
        
        storage.update_execution(execution_id, updates)


def _update_status_by_node(execution_id: str, node_name: str, is_start: bool):
    """노드 이름에 따라 분석 상태 업데이트"""
    # 노드 -> 분석 상태 매핑
    node_to_status = {
        "preprocess": None,  # preprocess는 상태 변경 없음
        "translate_ko_to_en": "translating",
        "label_utterances": "labeling",
        "detect_patterns": "pattern_detection",
        "summarize": "style_analysis",
        "key_moments": "style_analysis",
        "analyze_style": "style_analysis",
        "coaching_plan": "coaching_generation",
        "challenge_eval": "coaching_generation",
        "aggregate_result": "report_finalizing",
    }
    
    status_name = node_to_status.get(node_name)
    if status_name:
        update_analysis_status(execution_id, status_name)


def update_execution_status(execution_id: str, status: ExecutionStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """실행 상태 업데이트"""
    storage = get_storage_instance()
    exec_state = storage.get_execution(execution_id)
    
    if not exec_state:
        return
    
    updates = {
        "status": status,
        "updated_at": datetime.now().isoformat()
    }
    
    # 완료 또는 실패 시 분석 상태 업데이트
    if status == ExecutionStatus.COMPLETED:
        update_analysis_status(execution_id, "completed")
    elif status == ExecutionStatus.FAILED:
        update_analysis_status(execution_id, "failed")
        if error:
            updates["status_message"] = f"오류: {error}"
    
    if result:
        updates["result"] = result
    if error:
        updates["error"] = error
    
    storage.update_execution(execution_id, updates)


def get_execution_status(execution_id: str) -> Optional[Dict[str, Any]]:
    """실행 상태 조회"""
    storage = get_storage_instance()
    return storage.get_execution(execution_id)


def get_all_executions() -> Dict[str, Dict[str, Any]]:
    """모든 실행 목록 조회"""
    storage = get_storage_instance()
    return storage.get_all_executions()

