-- Query Logs 테이블
-- 검색 쿼리 로그 수집 및 분석
--
-- 목적:
-- 1. Weight 튜닝 (query_type별 최적 가중치)
-- 2. 인기도 추적 (자주 검색되는 노드)
-- 3. A/B 테스트 (weight 변경 효과)
-- 4. 검색 성능 분석

CREATE TABLE IF NOT EXISTS query_logs (
    id BIGSERIAL PRIMARY KEY,
    
    -- 쿼리 정보
    repo_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_type TEXT,                    -- 'api_location' | 'log_location' | 'structure' | 'function_impl' | 'general'
    query_embedding vector(1536),       -- 쿼리 임베딩 (유사 쿼리 분석용)
    
    -- 검색 설정
    weights JSONB,                      -- 사용한 weight 설정
    filters JSONB,                      -- 적용한 필터 (node_type, language 등)
    k INTEGER,                          -- 요청한 결과 수
    
    -- 검색 결과
    result_count INTEGER,               -- 반환된 결과 수
    top_results JSONB,                  -- 상위 결과 [{node_id, score, rank, signals}, ...]
    
    -- 사용자 피드백 (optional)
    clicked_node_ids TEXT[],            -- 클릭한 노드 ID 리스트
    feedback_score FLOAT,               -- 0-1 (사용자 만족도)
    
    -- 성능
    latency_ms INTEGER,                 -- 응답 시간 (밀리초)
    backend_latencies JSONB,            -- 백엔드별 레이턴시 {lexical: 10ms, semantic: 50ms}
    
    -- 메타데이터
    client_info JSONB,                  -- 클라이언트 정보 (CLI, API, MCP 등)
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- ============================================================================
-- 인덱스
-- ============================================================================

-- 기본 조회
CREATE INDEX idx_query_logs_repo_time ON query_logs(repo_id, created_at DESC);
CREATE INDEX idx_query_logs_query_type ON query_logs(query_type, created_at DESC);

-- 쿼리 텍스트 검색
CREATE INDEX idx_query_logs_query_text ON query_logs USING GIN (to_tsvector('english', query_text));

-- 쿼리 임베딩 유사도 (유사 쿼리 분석)
CREATE INDEX IF NOT EXISTS idx_query_logs_embedding
ON query_logs USING ivfflat (query_embedding vector_cosine_ops)
WITH (lists = 100);

-- 메타데이터 검색
CREATE INDEX idx_query_logs_weights ON query_logs USING GIN (weights);
CREATE INDEX idx_query_logs_filters ON query_logs USING GIN (filters);
CREATE INDEX idx_query_logs_top_results ON query_logs USING GIN (top_results);

-- ============================================================================
-- 노드 인기도 집계 테이블
-- ============================================================================

CREATE TABLE IF NOT EXISTS node_popularity (
    repo_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,            -- 'symbol' | 'route' | 'doc'
    
    -- 쿼리 빈도
    query_count_7d INTEGER DEFAULT 0,   -- 7일간 쿼리 결과에 포함된 횟수
    query_count_30d INTEGER DEFAULT 0,  -- 30일간
    
    -- 클릭 빈도
    click_count_7d INTEGER DEFAULT 0,   -- 7일간 클릭 횟수
    click_count_30d INTEGER DEFAULT 0,  -- 30일간
    
    -- 평균 순위
    avg_rank FLOAT,                     -- 평균 검색 순위 (낮을수록 좋음)
    
    -- 마지막 접근
    last_queried_at TIMESTAMP,
    last_clicked_at TIMESTAMP,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (repo_id, node_id),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_node_popularity_hot_7d ON node_popularity(repo_id, query_count_7d DESC);
CREATE INDEX idx_node_popularity_hot_30d ON node_popularity(repo_id, query_count_30d DESC);
CREATE INDEX idx_node_popularity_clicked ON node_popularity(repo_id, click_count_7d DESC);
CREATE INDEX idx_node_popularity_type ON node_popularity(repo_id, node_type);

-- ============================================================================
-- 코멘트
-- ============================================================================

COMMENT ON TABLE query_logs IS '검색 쿼리 로그 (weight 튜닝, 인기도 추적, A/B 테스트)';
COMMENT ON COLUMN query_logs.query_type IS 'QueryClassifier 출력';
COMMENT ON COLUMN query_logs.weights IS '사용한 가중치 설정 (QUERY_TYPE_WEIGHTS)';
COMMENT ON COLUMN query_logs.top_results IS '상위 결과 [{node_id, score, rank, signals}, ...]';
COMMENT ON COLUMN query_logs.clicked_node_ids IS '사용자가 클릭한 노드 ID 리스트';
COMMENT ON COLUMN query_logs.backend_latencies IS '백엔드별 레이턴시 {lexical: 10, semantic: 50}';

COMMENT ON TABLE node_popularity IS '노드 인기도 집계 (중요 노드 선정용)';
COMMENT ON COLUMN node_popularity.query_count_7d IS '7일간 검색 결과에 포함된 횟수';
COMMENT ON COLUMN node_popularity.avg_rank IS '평균 검색 순위 (1-based, 낮을수록 좋음)';

