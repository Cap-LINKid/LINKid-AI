-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 전문가 조언 테이블 생성
CREATE TABLE IF NOT EXISTS expert_advice (
    id SERIAL PRIMARY KEY,
    
    -- 기본 정보
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,  -- 벡터화할 메인 콘텐츠
    summary TEXT,  -- 간단 요약
    
    -- 카테고리 및 분류
    advice_type VARCHAR(50) NOT NULL,  -- 'coaching', 'challenge_guide', 'qa_tip', 'pattern_advice'
    category VARCHAR(100),  -- '긍정강화', '공감', '지시' 등
    
    -- 메타데이터 (검색 필터링용)
    pattern_names TEXT[],  -- ['긍정기회놓치기', '명령과제시'] 등
    dpics_labels TEXT[],  -- ['PR', 'RD', 'CMD'] 등
    interaction_stages TEXT[],  -- ['공감적 협력', '개선이 필요한 상호작용'] 등
    severity_levels TEXT[],  -- ['low', 'medium', 'high']
    
    -- 관련 정보
    related_challenges TEXT[],  -- 관련 챌린지 ID 또는 이름
    tags TEXT[],  -- ['칭찬', '긍정강화', '공감'] 등
    
    -- 벡터 임베딩
    embedding vector(1536),  -- OpenAI text-embedding-3-small 기준 (다른 모델은 차원 변경 필요)
    
    -- 메타데이터
    source VARCHAR(200),  -- 출처 (예: 'DPICS 가이드', '전문가 조언')
    author VARCHAR(200),  -- 저자/연구자 (예: '오은영 박사 연구', 'DPICS 연구팀')
    priority INTEGER DEFAULT 0,  -- 우선순위 (높을수록 우선)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 제약 조건
    CONSTRAINT valid_advice_type CHECK (advice_type IN ('coaching', 'challenge_guide', 'qa_tip', 'pattern_advice', 'general'))
);

-- 벡터 검색 성능을 위한 인덱스 (ivfflat)
-- 주의: 데이터가 충분히 많을 때만 생성 (최소 1000개 이상 권장)
-- CREATE INDEX ON expert_advice USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);  -- 데이터 크기에 따라 조정 (리스트 수 = sqrt(행 수) 정도)

-- 초기에는 HNSW 인덱스 사용 권장 (더 빠르고 정확)
CREATE INDEX ON expert_advice USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 메타데이터 필터링을 위한 인덱스
CREATE INDEX idx_pattern_names ON expert_advice USING GIN (pattern_names);
CREATE INDEX idx_dpics_labels ON expert_advice USING GIN (dpics_labels);
CREATE INDEX idx_advice_type ON expert_advice (advice_type);
CREATE INDEX idx_category ON expert_advice (category);
CREATE INDEX idx_author ON expert_advice (author);
CREATE INDEX idx_source ON expert_advice (source);
CREATE INDEX idx_priority ON expert_advice (priority DESC);

-- updated_at 자동 업데이트 트리거 함수
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- updated_at 자동 업데이트 트리거
CREATE TRIGGER update_expert_advice_updated_at
    BEFORE UPDATE ON expert_advice
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 테이블 코멘트
COMMENT ON TABLE expert_advice IS '전문가 조언 벡터 데이터베이스';
COMMENT ON COLUMN expert_advice.embedding IS '텍스트 임베딩 벡터 (차원은 모델에 따라 변경 필요)';
COMMENT ON COLUMN expert_advice.pattern_names IS '관련 패턴명 배열 (예: ["긍정기회놓치기", "명령과제시"])';
COMMENT ON COLUMN expert_advice.dpics_labels IS '관련 DPICS 라벨 배열 (예: ["PR", "RD", "CMD"])';

