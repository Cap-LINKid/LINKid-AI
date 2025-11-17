from __future__ import annotations

import os
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 환경 변수로 저장소 타입 선택 (memory, sqlite, mysql, redis)
STORAGE_TYPE = os.getenv("EXECUTION_STORAGE_TYPE", "sqlite").lower()


class StorageBackend(ABC):
    """저장소 백엔드 추상 클래스"""
    
    @abstractmethod
    def create_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        """실행 생성"""
        pass
    
    @abstractmethod
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """실행 조회"""
        pass
    
    @abstractmethod
    def update_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        """실행 업데이트"""
        pass
    
    @abstractmethod
    def get_all_executions(self) -> Dict[str, Dict[str, Any]]:
        """모든 실행 조회"""
        pass


class MemoryStorage(StorageBackend):
    """메모리 저장소 (기본)"""
    
    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
    
    def create_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        self._storage[execution_id] = data
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(execution_id)
    
    def update_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        if execution_id in self._storage:
            self._storage[execution_id].update(data)
    
    def get_all_executions(self) -> Dict[str, Dict[str, Any]]:
        return self._storage.copy()


class SQLiteStorage(StorageBackend):
    """SQLite 저장소"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # 프로젝트 루트에 저장
            project_root = Path(__file__).parent.parent.parent
            db_path = str(project_root / "executions.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated_at 
            ON executions(updated_at DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_conn(self):
        """연결 가져오기"""
        return sqlite3.connect(self.db_path)
    
    def create_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO executions (execution_id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (execution_id, json.dumps(data, ensure_ascii=False), now, now))
        
        conn.commit()
        conn.close()
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data FROM executions WHERE execution_id = ?
        """, (execution_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def update_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 기존 데이터 가져오기
        existing = self.get_execution(execution_id)
        if existing:
            existing.update(data)
            updated_data = existing
        else:
            updated_data = data
        
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO executions (execution_id, data, created_at, updated_at)
            VALUES (?, ?, 
                COALESCE((SELECT created_at FROM executions WHERE execution_id = ?), ?),
                ?)
        """, (execution_id, json.dumps(updated_data, ensure_ascii=False), 
              execution_id, now, now))
        
        conn.commit()
        conn.close()
    
    def get_all_executions(self) -> Dict[str, Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT execution_id, data FROM executions
            ORDER BY updated_at DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            execution_id, data_str = row
            results[execution_id] = json.loads(data_str)
        
        conn.close()
        return results


class MySQLStorage(StorageBackend):
    """MySQL 저장소"""
    
    def __init__(self):
        try:
            from src.utils.sql import get_mysql_conn
            self.get_conn = get_mysql_conn
            self._init_db()
        except ImportError:
            raise ImportError("pymysql이 설치되어 있지 않습니다.")
    
    def _init_db(self):
        """데이터베이스 초기화"""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                execution_id VARCHAR(255) PRIMARY KEY,
                data JSON NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_updated_at (updated_at DESC)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        conn.commit()
        conn.close()
    
    def create_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        conn = self.get_conn()
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute("""
            INSERT INTO executions (execution_id, data, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
        """, (execution_id, json.dumps(data, ensure_ascii=False), now, now))
        
        conn.commit()
        conn.close()
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data FROM executions WHERE execution_id = %s
        """, (execution_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def update_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        conn = self.get_conn()
        cursor = conn.cursor()
        
        # 기존 데이터 가져오기
        existing = self.get_execution(execution_id)
        if existing:
            existing.update(data)
            updated_data = existing
        else:
            updated_data = data
        
        now = datetime.now()
        cursor.execute("""
            INSERT INTO executions (execution_id, data, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                data = VALUES(data),
                updated_at = VALUES(updated_at)
        """, (execution_id, json.dumps(updated_data, ensure_ascii=False), 
              existing.get("created_at") if existing else now, now))
        
        conn.commit()
        conn.close()
    
    def get_all_executions(self) -> Dict[str, Dict[str, Any]]:
        conn = self.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT execution_id, data FROM executions
            ORDER BY updated_at DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            execution_id, data_str = row
            results[execution_id] = json.loads(data_str)
        
        conn.close()
        return results


class RedisStorage(StorageBackend):
    """Redis 저장소"""
    
    def __init__(self):
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.client = redis.from_url(redis_url, decode_responses=True)
        except ImportError:
            raise ImportError("redis 패키지가 설치되어 있지 않습니다. pip install redis")
    
    def create_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        key = f"execution:{execution_id}"
        self.client.set(key, json.dumps(data, ensure_ascii=False))
        # TTL 설정 (24시간)
        self.client.expire(key, 86400)
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        key = f"execution:{execution_id}"
        data_str = self.client.get(key)
        if data_str:
            return json.loads(data_str)
        return None
    
    def update_execution(self, execution_id: str, data: Dict[str, Any]) -> None:
        key = f"execution:{execution_id}"
        existing = self.get_execution(execution_id)
        if existing:
            existing.update(data)
            updated_data = existing
        else:
            updated_data = data
        
        self.client.set(key, json.dumps(updated_data, ensure_ascii=False))
        self.client.expire(key, 86400)
    
    def get_all_executions(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for key in self.client.scan_iter(match="execution:*"):
            execution_id = key.replace("execution:", "")
            data_str = self.client.get(key)
            if data_str:
                results[execution_id] = json.loads(data_str)
        return results


def get_storage() -> StorageBackend:
    """저장소 인스턴스 가져오기"""
    if STORAGE_TYPE == "sqlite":
        db_path = os.getenv("SQLITE_DB_PATH")
        return SQLiteStorage(db_path)
    elif STORAGE_TYPE == "mysql":
        return MySQLStorage()
    elif STORAGE_TYPE == "redis":
        return RedisStorage()
    else:
        # 기본값: 메모리
        return MemoryStorage()


# 전역 저장소 인스턴스
_storage: Optional[StorageBackend] = None


def get_storage_instance() -> StorageBackend:
    """싱글톤 저장소 인스턴스"""
    global _storage
    if _storage is None:
        _storage = get_storage()
    return _storage

