"""IndexingPipeline 테스트"""

import pytest
from pathlib import Path

from src.core.bootstrap import create_bootstrap
from src.core.models import RepoConfig


@pytest.fixture
def test_repo(tmp_path):
    """임시 테스트 저장소 생성"""
    # Python 파일 생성
    (tmp_path / "main.py").write_text("""
def hello(name):
    \"\"\"Say hello\"\"\"
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name):
        return hello(name)
""")
    
    (tmp_path / "utils.py").write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
    
    return tmp_path


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_pipeline_basic_flow(test_repo):
    """기본 파이프라인 테스트"""
    bootstrap = create_bootstrap()
    
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        name="test-repo"
    )
    
    assert result.status == "completed"
    assert result.total_files == 2
    assert result.processed_files == 2
    assert result.total_nodes > 0
    assert result.total_chunks > 0
    assert result.duration_seconds > 0


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_pipeline_with_config(test_repo):
    """설정이 있는 파이프라인 테스트"""
    bootstrap = create_bootstrap()
    
    config = RepoConfig(
        languages=["python"],
        include_tests=True
    )
    
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        name="test-repo-config",
        config=config
    )
    
    assert result.status == "completed"
    assert result.total_files == 2


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_pipeline_custom_repo_id(test_repo):
    """커스텀 repo_id 테스트"""
    bootstrap = create_bootstrap()
    
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        repo_id="my-custom-id",
        name="test-repo-custom"
    )
    
    assert result.status == "completed"
    assert result.repo_id == "my-custom-id"


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_pipeline_empty_directory(tmp_path):
    """빈 디렉토리 테스트"""
    bootstrap = create_bootstrap()
    
    result = bootstrap.pipeline.index_repository(
        root_path=str(tmp_path),
        name="empty-repo"
    )
    
    assert result.status == "completed"
    assert result.total_files == 0
    assert result.total_nodes == 0


@pytest.mark.skip(reason="Requires PostgreSQL connection")
def test_pipeline_metadata_persistence(test_repo):
    """메타데이터 저장 테스트"""
    bootstrap = create_bootstrap()
    
    # 인덱싱
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        repo_id="test-persistence",
        name="test-repo-metadata"
    )
    
    # 메타데이터 조회
    metadata = bootstrap.repo_store.get("test-persistence")
    
    assert metadata is not None
    assert metadata.repo_id == "test-persistence"
    assert metadata.name == "test-repo-metadata"
    assert metadata.total_files == result.total_files
    assert metadata.total_nodes == result.total_nodes
    assert metadata.total_chunks == result.total_chunks
    assert "python" in metadata.languages


@pytest.mark.skip(reason="Requires full database and API keys")
def test_pipeline_full_integration(test_repo):
    """전체 통합 테스트 (DB + API 키 필요)"""
    import os
    
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("EMBEDDING_API_KEY not set")
    
    bootstrap = create_bootstrap()
    
    result = bootstrap.pipeline.index_repository(
        root_path=str(test_repo),
        repo_id="test-full-integration",
        name="test-repo-full"
    )
    
    assert result.status == "completed"
    
    # 그래프 조회 테스트
    nodes = bootstrap.graph_store.get_nodes_by_repo("test-full-integration")
    assert len(nodes) > 0
    
    # 청크 조회 테스트
    chunks = bootstrap.chunk_store.get_chunks_by_repo("test-full-integration")
    assert len(chunks) > 0
    
    # Lexical 검색 테스트
    results = bootstrap.lexical_search.search(
        repo_id="test-full-integration",
        query="hello",
        k=5
    )
    assert len(results) > 0

