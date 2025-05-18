-- -- ================================================
-- -- 1. 사고 리스트 테이블
-- -- ================================================
-- CREATE TABLE IF NOT EXISTS subsidence_accident_list (
--   sagoNo       VARCHAR(20)     PRIMARY KEY,      -- 사고번호
--   siDo         VARCHAR(20),                       -- 시도명
--   siGunGu      VARCHAR(20),                       -- 시군구명
--   sagoReason   TEXT,                              -- 상세 발생원인
--   sagoDate     DATE,                              -- 사고 발생일자
-- --   no           INT,                               -- 순번
--
--   sinkWidth    NUMERIC(9,2),                      -- 발생규모 폭 (m)
--   sinkExtend   NUMERIC(9,2),                      -- 발생규모 연장 (m)
--   sinkDepth    NUMERIC(9,2),                      -- 발생규모 깊이 (m)
--
--   grdKind      VARCHAR(20),                       -- 지질종류
--   sagoReason   TEXT,                              -- 상세 발생원인
--
--   deathCnt     INT,                               -- 사망자 수
--   injuryCnt    INT,                               -- 부상자 수
--   vehicleCnt   INT,                               -- 차량 대수
--
--   trAmount     NUMERIC(9),                        -- 복구 비용
--   raw_payload  JSONB                              -- 원본 JSON
--
--   raw_payload  JSONB                             -- 원본 JSON
-- );
--
-- -- ================================================
-- -- 2. 사고 상세 테이블
-- -- ================================================



-- CREATE TABLE IF NOT EXISTS subsidence_accident_info (
--   sagoNo       VARCHAR(20)     PRIMARY KEY       REFERENCES subsidence_accident_list(sagoNo),
--
--   siDo         VARCHAR(20),                       -- 시도명
--   siGunGu      VARCHAR(20),                       -- 시군구명
--   dong         VARCHAR(50),                       -- 읍면동명
--   addr         VARCHAR(200),                      -- 상세주소
--   sagoDate     DATE,                              -- 사고 발생일자
--
--   sinkWidth    NUMERIC(9,2),                      -- 발생규모 폭 (m)
--   sinkExtend   NUMERIC(9,2),                      -- 발생규모 연장 (m)
--   sinkDepth    NUMERIC(9,2),                      -- 발생규모 깊이 (m)
--
--   grdKind      VARCHAR(20),                       -- 지질종류
--   sagoReason   TEXT,                              -- 상세 발생원인
--
--   deathCnt     INT,                               -- 사망자 수
--   injuryCnt    INT,                               -- 부상자 수
--   vehicleCnt   INT,                               -- 차량 대수
--
--   trAmount     NUMERIC(9),                        -- 복구 비용
--   raw_payload  JSONB                              -- 원본 JSON
-- );
--
-- -- ================================================
-- -- 3. (선택) TimescaleDB 하이퍼테이블 생성
-- --    *psql* 세션에서 한 번만 실행하세요.
-- -- ================================================
-- -- SELECT create_hypertable('subsidence_accident_list', 'sagoDate', if_not_exists => TRUE);

-- ───────────────────────────────────────────────────────────────────────────
-- 1) subsidence_accident_info 테이블에서 불필요 컬럼 제거
-- ───────────────────────────────────────────────────────────────────────────
ALTER TABLE subsidence_accident_info
  DROP COLUMN IF EXISTS tramount,
  DROP COLUMN IF EXISTS trmethod,
  DROP COLUMN IF EXISTS trfndate,
  DROP COLUMN IF EXISTS dastdate,
  DROP COLUMN IF EXISTS trstatus;

-- ───────────────────────────────────────────────────────────────────────────
-- 2) subsidence_accident_info 에 list 테이블의 sido/sigungu 값 채우기
-- ───────────────────────────────────────────────────────────────────────────
UPDATE subsidence_accident_info AS info
SET
  sido    = list.sido,
  sigungu = list.sigungu
FROM subsidence_accident_list AS list
WHERE
  info.sagono = list.sagoNo
  AND (info.sido    IS NULL OR info.sido    = '')
  AND (info.sigungu IS NULL OR info.sigungu = '');

-- ───────────────────────────────────────────────────────────────────────────
-- 3) 통계 정보 최적화 (선택)
-- ───────────────────────────────────────────────────────────────────────────
  VACUUM ANALYZE subsidence_accident_info;