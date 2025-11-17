#!/bin/bash
# FastAPI 서버 실행
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

