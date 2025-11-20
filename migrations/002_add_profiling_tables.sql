-- 저장소 프로파일 테이블 추가
-- Repo Profiling 및 File Profiling 지원

-- 1. 저장소 프로파일 테이블
CREATE TABLE IF NOT EXISTS repo_profile (
    repo_id VARCHAR(255) PRIMARY KEY,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_repo_profile_framework ON repo_profile ((profile_data->>'framework'));
CREATE INDEX IF NOT EXISTS idx_repo_profile_project_type ON repo_profile ((profile_data->>'project_type'));

-- 2. 파일 프로파일 테이블
CREATE TABLE IF NOT EXISTS file_profile (
    id SERIAL PRIMARY KEY,
    repo_id VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(repo_id, file_path),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_file_profile_repo ON file_profile(repo_id);
CREATE INDEX IF NOT EXISTS idx_file_profile_is_api ON file_profile ((profile_data->>'is_api_file'));
CREATE INDEX IF NOT EXISTS idx_file_profile_is_test ON file_profile ((profile_data->>'is_test_file'));

-- 3. 청크 메타데이터 확장 (기존 code_chunks 테이블에 컬럼 추가)
ALTER TABLE code_chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- 청크 메타데이터 인덱스
CREATE INDEX IF NOT EXISTS idx_code_chunks_metadata ON code_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_code_chunks_is_api_endpoint ON code_chunks ((metadata->>'is_api_endpoint_chunk'));

-- 4. 노드 중요도 점수 (기존 code_nodes 테이블에 컬럼 추가)
ALTER TABLE code_nodes ADD COLUMN IF NOT EXISTS importance_score FLOAT DEFAULT 0.0;

-- 노드 중요도 인덱스
CREATE INDEX IF NOT EXISTS idx_code_nodes_importance ON code_nodes(repo_id, importance_score DESC);

COMMENT ON TABLE repo_profile IS '저장소 구조 프로파일 (검색 최적화용)';
COMMENT ON TABLE file_profile IS '파일 역할 프로파일 (검색 재순위화용)';
COMMENT ON COLUMN code_chunks.metadata IS '청크 메타데이터 (API endpoint, http method 등)';
COMMENT ON COLUMN code_nodes.importance_score IS '노드 중요도 점수 (PageRank 스타일)';
