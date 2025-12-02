-- expert_advice 테이블의 advice_type 체크 제약 조건 업데이트
-- limit_setting, self_reflection 추가

-- 기존 제약 조건 삭제
ALTER TABLE expert_advice DROP CONSTRAINT IF EXISTS valid_advice_type;

-- 새로운 제약 조건 추가 (limit_setting, self_reflection 포함)
ALTER TABLE expert_advice 
ADD CONSTRAINT valid_advice_type 
CHECK (advice_type IN (
    'coaching', 
    'challenge_guide', 
    'qa_tip', 
    'pattern_advice', 
    'general', 
    'limit_setting', 
    'self_reflection'
));


