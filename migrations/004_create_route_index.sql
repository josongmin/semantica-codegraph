-- Route Index 테이블
-- API 엔드포인트 전용 인덱스

CREATE TABLE IF NOT EXISTS route_index (
    repo_id TEXT NOT NULL,
    route_id TEXT NOT NULL,
    http_method TEXT NOT NULL,
    http_path TEXT NOT NULL,
    handler_symbol_id TEXT,
    handler_name TEXT,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    router_prefix TEXT,
    framework TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (repo_id, route_id),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_route_method_path
ON route_index(repo_id, http_method, http_path);

CREATE INDEX IF NOT EXISTS idx_route_path_pattern
ON route_index(repo_id, http_path);

CREATE INDEX IF NOT EXISTS idx_route_file
ON route_index(repo_id, file_path);

CREATE INDEX IF NOT EXISTS idx_route_framework
ON route_index(repo_id, framework);

COMMENT ON TABLE route_index IS 'API 엔드포인트 인덱스 (FastAPI, Express, Spring 등)';
COMMENT ON COLUMN route_index.route_id IS '라우트 고유 ID (hash)';
COMMENT ON COLUMN route_index.handler_symbol_id IS 'code_nodes.id 참조';
COMMENT ON INDEX idx_route_method_path IS 'HTTP 메서드+경로 검색';
