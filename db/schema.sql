-- 1. VIEW를 테이블로 복사
CREATE TABLE pothole_features_copy AS
SELECT * FROM feature_pothole_all;

-- 2. 컬럼 추가
ALTER TABLE pothole_features_copy
ADD COLUMN subsidence_occurrence INTEGER;

-- 3. 침하 발생 여부 채우기
UPDATE pothole_features_copy
SET subsidence_occurrence = 1
WHERE grid_id IN (SELECT grid_id FROM grid_sinkhole_table);

UPDATE pothole_features_copy
SET subsidence_occurrence = 0
WHERE subsidence_occurrence IS NULL;