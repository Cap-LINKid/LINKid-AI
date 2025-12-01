-- PostgreSQL에서 직접 벡터 검색 테스트 쿼리

-- ============================================
-- 1. 데이터 확인
-- ============================================

-- 전체 데이터 개수 확인
SELECT COUNT(*) as total_count FROM expert_advice;

-- 샘플 데이터 확인 (메타데이터)
SELECT 
    id,
    title,
    source,
    author,
    advice_type,
    category,
    pattern_names,
    dpics_labels
FROM expert_advice 
LIMIT 5;

-- 벡터 차원 확인
SELECT 
    id,
    title,
    array_length(embedding::float[], 1) as embedding_dimension
FROM expert_advice 
LIMIT 1;

-- ============================================
-- 2. 기본 벡터 검색 (유사도 기반)
-- ============================================

-- 예시: 특정 텍스트의 임베딩 벡터를 생성한 후 검색
-- 주의: 실제로는 Python에서 임베딩을 생성하고 그 벡터를 사용해야 합니다

-- 예시 벡터 (1536차원, 실제로는 OpenAI API로 생성)
-- 여기서는 첫 번째 문서의 임베딩을 쿼리로 사용하는 예시
WITH query_vector AS (
    SELECT embedding as query_emb
    FROM expert_advice
    WHERE id = 1
    LIMIT 1
)
SELECT 
    ea.id,
    ea.title,
    ea.source,
    ea.author,
    ea.advice_type,
    -- 코사인 유사도 계산 (1 - 거리)
    1 - (ea.embedding <=> qv.query_emb) as similarity_score,
    -- 또는 L2 거리
    ea.embedding <-> qv.query_emb as l2_distance
FROM expert_advice ea, query_vector qv
WHERE ea.id != 1  -- 자기 자신 제외
ORDER BY ea.embedding <=> qv.query_emb  -- 코사인 거리 기준 정렬
LIMIT 5;

-- ============================================
-- 3. 메타데이터 필터링과 함께 검색
-- ============================================

-- 특정 패턴명으로 필터링
SELECT 
    id,
    title,
    pattern_names,
    advice_type,
    source
FROM expert_advice
WHERE '긍정기회놓치기' = ANY(pattern_names)
LIMIT 10;

-- 특정 DPICS 라벨로 필터링
SELECT 
    id,
    title,
    dpics_labels,
    advice_type
FROM expert_advice
WHERE 'PR' = ANY(dpics_labels)
LIMIT 10;

-- advice_type으로 필터링
SELECT 
    id,
    title,
    advice_type,
    category
FROM expert_advice
WHERE advice_type = 'pattern_advice'
LIMIT 10;

-- ============================================
-- 4. 벡터 검색 + 메타데이터 필터링 조합
-- ============================================

-- 예시: 특정 패턴과 유사한 조언 찾기
WITH query_vector AS (
    SELECT embedding as query_emb
    FROM expert_advice
    WHERE '긍정기회놓치기' = ANY(pattern_names)
    LIMIT 1
)
SELECT 
    ea.id,
    ea.title,
    ea.pattern_names,
    ea.source,
    ea.author,
    1 - (ea.embedding <=> qv.query_emb) as similarity_score
FROM expert_advice ea, query_vector qv
WHERE ea.advice_type = 'pattern_advice'
  AND 1 - (ea.embedding <=> qv.query_emb) > 0.7  -- 유사도 임계값
ORDER BY ea.embedding <=> qv.query_emb
LIMIT 5;

-- ============================================
-- 5. 인덱스 사용 확인
-- ============================================

-- 인덱스 목록 확인
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'expert_advice';

-- 쿼리 실행 계획 확인 (인덱스 사용 여부)
EXPLAIN ANALYZE
SELECT 
    id,
    title,
    1 - (embedding <=> (SELECT embedding FROM expert_advice WHERE id = 1 LIMIT 1)) as similarity
FROM expert_advice
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM expert_advice WHERE id = 1 LIMIT 1)
LIMIT 5;

-- ============================================
-- 6. 통계 및 분석 쿼리
-- ============================================

-- advice_type별 개수
SELECT 
    advice_type,
    COUNT(*) as count
FROM expert_advice
GROUP BY advice_type
ORDER BY count DESC;

-- 패턴명별 개수
SELECT 
    unnest(pattern_names) as pattern_name,
    COUNT(*) as count
FROM expert_advice
GROUP BY pattern_name
ORDER BY count DESC;

-- DPICS 라벨별 개수
SELECT 
    unnest(dpics_labels) as dpics_label,
    COUNT(*) as count
FROM expert_advice
GROUP BY dpics_label
ORDER BY count DESC;

-- ============================================
-- 7. Python에서 생성한 임베딩 벡터로 검색하는 방법
-- ============================================

-- Python에서 다음과 같이 임베딩을 생성한 후:
-- embedding = [0.1, 0.2, 0.3, ...]  # 1536차원 벡터
-- 
-- PostgreSQL에서 검색:
-- SELECT 
--     id,
--     title,
--     1 - (embedding <=> '[0.1,0.2,0.3,...]'::vector) as similarity
-- FROM expert_advice
-- ORDER BY embedding <=> '[0.1,0.2,0.3,...]'::vector
-- LIMIT 5;

-- ============================================
-- 8. 유용한 헬퍼 함수
-- ============================================

-- 가장 유사한 문서 찾기 (함수)
CREATE OR REPLACE FUNCTION find_similar_advice(
    query_embedding vector(1536),
    top_k int DEFAULT 5,
    min_similarity float DEFAULT 0.7,
    filter_advice_type text DEFAULT NULL
)
RETURNS TABLE (
    id int,
    title text,
    source text,
    author text,
    advice_type text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ea.id,
        ea.title,
        ea.source,
        ea.author,
        ea.advice_type,
        1 - (ea.embedding <=> query_embedding)::float as similarity
    FROM expert_advice ea
    WHERE (filter_advice_type IS NULL OR ea.advice_type = filter_advice_type)
      AND 1 - (ea.embedding <=> query_embedding) >= min_similarity
    ORDER BY ea.embedding <=> query_embedding
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

-- 함수 사용 예시:
-- SELECT * FROM find_similar_advice(
--     (SELECT embedding FROM expert_advice WHERE id = 1),
--     top_k := 5,
--     min_similarity := 0.7,
--     filter_advice_type := 'pattern_advice'
-- );

