FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성 설치 (PostgreSQL 클라이언트 라이브러리 등)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY data ./data
COPY src ./src
# COPY models ./models
COPY langgraph.json ./

# 포트 노출
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
