-- Semantica Codegraph Database Schema
-- PostgreSQL + pgvector

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. 저장소 메타데이터
-- ============================================================================
CREATE TABLE IF NOT EXISTS repo_metadata (
    repo_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL,
    git_url TEXT,
    default_branch TEXT DEFAULT 'main',
    languages TEXT[],
    
    -- 통계
    total_files INTEGER DEFAULT 0,
    total_nodes INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    
    -- 인덱싱 상태
    indexing_status TEXT DEFAULT 'pending',  -- "pending" | "indexing" | "completed" | "failed"
    indexing_progress FLOAT DEFAULT 0.0,
    indexing_started_at TIMESTAMP,
    indexing_completed_at TIMESTAMP,
    indexing_error TEXT,
    
    -- 타임스탬프
    created_at TIMESTAMP DEFAULT NOW(),
    last_indexed_at TIMESTAMP,
    
    -- 저장소별 설정
    config JSONB
);

CREATE INDEX IF NOT EXISTS idx_repos_status 
ON repo_metadata(indexing_status);

-- ============================================================================
-- 2. 코드 그래프 (Nodes)
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_nodes (
    repo_id TEXT NOT NULL,
    id TEXT NOT NULL,
    kind TEXT NOT NULL,
    language TEXT NOT NULL,
    file_path TEXT NOT NULL,
    span_start_line INTEGER NOT NULL,
    span_start_col INTEGER NOT NULL,
    span_end_line INTEGER NOT NULL,
    span_end_col INTEGER NOT NULL,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    attrs JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (repo_id, id),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_file_path 
ON code_nodes(repo_id, file_path);

CREATE INDEX IF NOT EXISTS idx_nodes_location 
ON code_nodes(repo_id, file_path, span_start_line, span_end_line);

CREATE INDEX IF NOT EXISTS idx_nodes_name 
ON code_nodes(repo_id, name);

CREATE INDEX IF NOT EXISTS idx_nodes_kind 
ON code_nodes(repo_id, kind);

-- ============================================================================
-- 3. 코드 그래프 (Edges)
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_edges (
    repo_id TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (repo_id, src_id, dst_id, type),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_src 
ON code_edges(repo_id, src_id);

CREATE INDEX IF NOT EXISTS idx_edges_dst 
ON code_edges(repo_id, dst_id);

CREATE INDEX IF NOT EXISTS idx_edges_type 
ON code_edges(repo_id, type);

-- ============================================================================
-- 4. 코드 청크 (검색/RAG 단위)
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_chunks (
    repo_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    span_start_line INTEGER NOT NULL,
    span_start_col INTEGER NOT NULL,
    span_end_line INTEGER NOT NULL,
    span_end_col INTEGER NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER,
    attrs JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (repo_id, chunk_id),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- Zoekt 매핑에 필수!
CREATE INDEX IF NOT EXISTS idx_chunks_location 
ON code_chunks(repo_id, file_path, span_start_line, span_end_line);

CREATE INDEX IF NOT EXISTS idx_chunks_node 
ON code_chunks(repo_id, node_id);

CREATE INDEX IF NOT EXISTS idx_chunks_language 
ON code_chunks(repo_id, language);

-- ============================================================================
-- 5. 임베딩 (벡터 검색)
-- ============================================================================
CREATE TABLE IF NOT EXISTS embeddings (
    repo_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    model TEXT NOT NULL,                -- "text-embedding-3-small" 등
    embedding vector(384),              -- pgvector (차원은 모델에 따라)
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (repo_id, chunk_id, model),
    FOREIGN KEY (repo_id, chunk_id) REFERENCES code_chunks(repo_id, chunk_id) ON DELETE CASCADE
);

-- pgvector 인덱스 (코사인 유사도)
-- HNSW: 빠른 근사 검색 (PostgreSQL 16+)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
ON embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 또는 IVFFlat: 메모리 효율적 (PostgreSQL 11+)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat 
-- ON embeddings USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embeddings_model 
ON embeddings(repo_id, model);

-- ============================================================================
-- 6. 파일 메타데이터 (증분 인덱싱용, 선택)
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_files (
    repo_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    size_bytes INTEGER,
    line_count INTEGER,
    content_hash TEXT NOT NULL,         -- SHA256 해시
    is_test BOOLEAN DEFAULT FALSE,
    last_modified TIMESTAMP,
    indexed_at TIMESTAMP,
    parsing_status TEXT DEFAULT 'pending',  -- "pending" | "success" | "failed"
    parsing_error TEXT,
    PRIMARY KEY (repo_id, file_path),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_files_language 
ON code_files(repo_id, language);

CREATE INDEX IF NOT EXISTS idx_files_status 
ON code_files(repo_id, parsing_status);

CREATE INDEX IF NOT EXISTS idx_files_hash 
ON code_files(repo_id, content_hash);

-- ============================================================================
-- Views (조회 편의용)
-- ============================================================================

-- 저장소 요약 뷰
CREATE OR REPLACE VIEW repo_summary AS
SELECT 
    r.repo_id,
    r.name,
    r.indexing_status,
    r.total_files,
    r.total_nodes,
    r.total_chunks,
    r.languages,
    r.last_indexed_at,
    COUNT(DISTINCT e.chunk_id) as embedded_chunks
FROM repo_metadata r
LEFT JOIN embeddings e ON r.repo_id = e.repo_id
GROUP BY r.repo_id, r.name, r.indexing_status, r.total_files, 
         r.total_nodes, r.total_chunks, r.languages, r.last_indexed_at;

-- 파일별 통계 뷰
CREATE OR REPLACE VIEW file_stats AS
SELECT 
    f.repo_id,
    f.file_path,
    f.language,
    f.parsing_status,
    COUNT(DISTINCT n.id) as node_count,
    COUNT(DISTINCT c.chunk_id) as chunk_count
FROM code_files f
LEFT JOIN code_nodes n ON f.repo_id = n.repo_id AND f.file_path = n.file_path
LEFT JOIN code_chunks c ON f.repo_id = c.repo_id AND f.file_path = c.file_path
GROUP BY f.repo_id, f.file_path, f.language, f.parsing_status;

-- ============================================================================
-- 함수 (유틸리티)
-- ============================================================================

-- 청크 토큰 수 업데이트 함수
CREATE OR REPLACE FUNCTION update_chunk_token_count()
RETURNS TRIGGER AS $$
BEGIN
    -- 간단한 토큰 추정: 단어 수 * 1.3
    NEW.token_count := CEIL(array_length(string_to_array(NEW.text, ' '), 1) * 1.3);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_chunk_token_count
BEFORE INSERT OR UPDATE ON code_chunks
FOR EACH ROW
EXECUTE FUNCTION update_chunk_token_count();

-- 저장소 통계 업데이트 함수
CREATE OR REPLACE FUNCTION update_repo_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE repo_metadata SET
        total_nodes = (SELECT COUNT(*) FROM code_nodes WHERE repo_id = NEW.repo_id),
        total_chunks = (SELECT COUNT(*) FROM code_chunks WHERE repo_id = NEW.repo_id),
        last_indexed_at = NOW()
    WHERE repo_id = NEW.repo_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 노드 삽입/업데이트 시 통계 갱신
CREATE TRIGGER trigger_update_repo_stats_nodes
AFTER INSERT OR UPDATE OR DELETE ON code_nodes
FOR EACH ROW
EXECUTE FUNCTION update_repo_stats();

-- 청크 삽입/업데이트 시 통계 갱신
CREATE TRIGGER trigger_update_repo_stats_chunks
AFTER INSERT OR UPDATE OR DELETE ON code_chunks
FOR EACH ROW
EXECUTE FUNCTION update_repo_stats();

