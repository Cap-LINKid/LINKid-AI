-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 전문가 조언 벡터 테이블 (CSV/TSV 컬럼 구조에 맞춤)
CREATE TABLE IF NOT EXISTS expert_advice (
    id SERIAL PRIMARY KEY,

    -- 사용자가 정의한 주요 컬럼
    category TEXT,                       -- Category
    age TEXT,                            -- Age
    related_dpics TEXT,                  -- Related_DPICS (관련 패턴) - 원문 문자열 그대로 저장
    keyword TEXT,                        -- Keyword
    situation TEXT,                      -- Situation
    type TEXT,                           -- Type (예: Positive / Negative 등)
    advice TEXT,                         -- Advice (긴 설명)
    reference TEXT,                      -- Reference (출처 정보)

    -- 벡터 임베딩
    embedding vector(1536),              -- OpenAI text-embedding-3-small 기준 (필요시 차원 수정)

    -- 메타데이터
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 벡터 검색용 HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_expert_advice_embedding_hnsw
ON expert_advice
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 자주 필터링/정렬할 수 있는 컬럼 인덱스
CREATE INDEX IF NOT EXISTS idx_expert_advice_category ON expert_advice (category);
CREATE INDEX IF NOT EXISTS idx_expert_advice_age ON expert_advice (age);
CREATE INDEX IF NOT EXISTS idx_expert_advice_type ON expert_advice (type);

-- updated_at 자동 업데이트 트리거 함수
CREATE OR REPLACE FUNCTION update_expert_advice_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- updated_at 자동 업데이트 트리거
CREATE TRIGGER trg_update_expert_advice_updated_at
    BEFORE UPDATE ON expert_advice
    FOR EACH ROW
    EXECUTE FUNCTION update_expert_advice_updated_at();

-- 테이블/컬럼 설명
COMMENT ON TABLE expert_advice IS '전문가 조언 벡터 데이터베이스 (Category~Reference 컬럼을 하나의 문서로 합쳐 임베딩)';
COMMENT ON COLUMN expert_advice.embedding IS 'Category~Reference 전체를 하나의 문서로 합쳐 생성한 텍스트 임베딩 벡터';