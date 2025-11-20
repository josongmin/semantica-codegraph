-- SymbolIndex 최적화 인덱스
-- code_nodes 테이블을 직접 검색하기 위한 인덱스 추가

-- 1. 심볼 이름 검색 (대소문자 무시)
CREATE INDEX IF NOT EXISTS idx_nodes_name_lower
ON code_nodes(repo_id, LOWER(name));

-- 2. 심볼 종류 + 이름 검색
CREATE INDEX IF NOT EXISTS idx_nodes_kind_name
ON code_nodes(repo_id, kind, LOWER(name));

-- 3. 파일 내 심볼 검색
CREATE INDEX IF NOT EXISTS idx_nodes_file_kind
ON code_nodes(repo_id, file_path, kind);

-- 4. Decorator 검색 (JSONB GIN 인덱스)
CREATE INDEX IF NOT EXISTS idx_nodes_decorators
ON code_nodes USING GIN ((attrs->'decorators'));

-- 5. 심볼 이름 패턴 검색 (trigram 유사도)
-- pg_trgm 확장 필요
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS idx_nodes_name_trgm
ON code_nodes USING gin (name gin_trgm_ops);

COMMENT ON INDEX idx_nodes_name_lower IS '심볼 이름 검색 (대소문자 무시)';
COMMENT ON INDEX idx_nodes_kind_name IS '심볼 종류별 검색 최적화';
COMMENT ON INDEX idx_nodes_decorators IS 'Decorator 검색 (@router.get 등)';
COMMENT ON INDEX idx_nodes_name_trgm IS 'Trigram 기반 퍼지 검색';
