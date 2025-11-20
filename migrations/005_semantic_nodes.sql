-- Semantic Nodes 테이블
-- 노드 수준 요약 + 임베딩 (symbol/route/doc/issue 통합)
--
-- 설계 원칙:
-- 1. 모든 고급 노드(symbol, route, doc)의 요약을 통합 관리
-- 2. 템플릿 요약 / LLM 요약 구분
-- 3. 다중 모델 지원 (3-small, 3-large)
-- 4. 차원 1536 고정 (3072는 별도 테이블)
-- 5. node_id = 원본 테이블 PK (prefix 없이)

CREATE TABLE IF NOT EXISTS semantic_nodes (
    repo_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,          -- 'symbol' | 'route' | 'doc' | 'issue'
    doc_type TEXT,                     -- node_type='doc'일 때: 'readme' | 'adr' | 'design' | 'api_doc'
    
    -- 요약 텍스트
    summary TEXT NOT NULL,             -- 검색용 텍스트 (템플릿 or LLM 생성)
    summary_method TEXT NOT NULL,     -- 'template' | 'llm'
    
    -- 임베딩
    model TEXT NOT NULL,               -- 'text-embedding-3-small' | 'text-embedding-3-large' (풀 네임)
    embedding vector(1536),            -- pgvector (1536 차원 고정, 3072는 별도 테이블)
    
    -- 원본 참조
    source_table TEXT,                 -- 'code_nodes' | 'route_index' | 'documents'
    source_id TEXT,                    -- 원본 테이블의 PK
    
    -- 메타데이터 (JSONB)
    metadata JSONB DEFAULT '{}',       -- importance_score, is_api_handler, query_count, line_count 등
    
    -- 타임스탬프
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (repo_id, node_id, node_type, model),
    FOREIGN KEY (repo_id) REFERENCES repo_metadata(repo_id) ON DELETE CASCADE
);

-- ============================================================================
-- 인덱스
-- ============================================================================

-- 기본 검색 필터
CREATE INDEX idx_semantic_nodes_type_model 
ON semantic_nodes(repo_id, node_type, model);

-- 문서 타입 필터
CREATE INDEX idx_semantic_nodes_doc_type 
ON semantic_nodes(repo_id, doc_type) 
WHERE doc_type IS NOT NULL;

-- 원본 참조 (디버깅/조인용)
CREATE INDEX idx_semantic_nodes_source 
ON semantic_nodes(repo_id, source_table, source_id);

-- 메타데이터 GIN 인덱스
CREATE INDEX idx_semantic_nodes_metadata 
ON semantic_nodes USING GIN (metadata);

-- Importance 정렬
CREATE INDEX idx_semantic_nodes_importance 
ON semantic_nodes(repo_id, ((metadata->>'importance_score')::float) DESC NULLS LAST);

-- Summary method 필터
CREATE INDEX idx_semantic_nodes_summary_method
ON semantic_nodes(repo_id, summary_method);

-- ============================================================================
-- pgvector 인덱스 (모델별 partial index)
-- ============================================================================

-- 3-small 전용 IVFFlat 인덱스 (PostgreSQL 16 호환)
CREATE INDEX IF NOT EXISTS idx_semantic_nodes_ivfflat_small
ON semantic_nodes USING ivfflat (embedding vector_cosine_ops)
WHERE model = 'text-embedding-3-small'
WITH (lists = 100);

-- 3-large 전용 IVFFlat 인덱스 (PostgreSQL 16 호환)
CREATE INDEX IF NOT EXISTS idx_semantic_nodes_ivfflat_large
ON semantic_nodes USING ivfflat (embedding vector_cosine_ops)
WHERE model = 'text-embedding-3-large'
WITH (lists = 100);

-- ============================================================================
-- 코멘트
-- ============================================================================

COMMENT ON TABLE semantic_nodes IS '노드 수준 요약 + 임베딩 (symbol/route/doc/issue 통합)';
COMMENT ON COLUMN semantic_nodes.node_type IS 'symbol: 코드 심볼, route: API 엔드포인트, doc: 문서, issue: 이슈';
COMMENT ON COLUMN semantic_nodes.summary_method IS 'template: 템플릿 기반, llm: LLM 요약';
COMMENT ON COLUMN semantic_nodes.model IS '임베딩 모델 풀 네임 (예: text-embedding-3-small)';
COMMENT ON COLUMN semantic_nodes.metadata IS 'importance_score, query_count, is_api_handler, line_count 등';
COMMENT ON COLUMN semantic_nodes.source_table IS '원본 테이블 (code_nodes, route_index 등)';
COMMENT ON COLUMN semantic_nodes.source_id IS '원본 테이블의 PK';

COMMENT ON INDEX idx_semantic_nodes_ivfflat_small IS 'IVFFlat 벡터 인덱스 (3-small 전용)';
COMMENT ON INDEX idx_semantic_nodes_ivfflat_large IS 'IVFFlat 벡터 인덱스 (3-large 전용)';

-- ============================================================================
-- 통계 업데이트 트리거
-- ============================================================================

CREATE OR REPLACE FUNCTION update_semantic_nodes_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_semantic_nodes_timestamp
BEFORE UPDATE ON semantic_nodes
FOR EACH ROW
EXECUTE FUNCTION update_semantic_nodes_timestamp();

