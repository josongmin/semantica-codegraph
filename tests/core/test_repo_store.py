"""RepoMetadataStore 테스트"""

import pytest
from datetime import datetime

from src.core.models import RepoMetadata


@pytest.fixture
def sample_metadata():
    """샘플 RepoMetadata"""
    return RepoMetadata(
        repo_id="test-repo",
        name="Test Repository",
        root_path="/path/to/repo",
        languages=["python", "typescript"],
        indexed_at=datetime.now(),
        total_files=10,
        total_nodes=100,
        total_chunks=50,
        attrs={"git_url": "https://github.com/user/repo"}
    )


def test_repo_metadata_creation(sample_metadata):
    """RepoMetadata 생성 테스트"""
    assert sample_metadata.repo_id == "test-repo"
    assert sample_metadata.name == "Test Repository"
    assert len(sample_metadata.languages) == 2


def test_save_metadata(sample_metadata):
    """메타데이터 저장 테스트"""
    from src.core.repo_store import RepoMetadataStore
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    store = RepoMetadataStore(conn_str)
    
    store.save(sample_metadata)
    
    # 조회로 검증
    retrieved = store.get("test-repo")
    assert retrieved is not None
    assert retrieved.name == "Test Repository"
    assert retrieved.total_files == 10


def test_list_all_repos():
    """모든 저장소 목록 조회 테스트"""
    from src.core.repo_store import RepoMetadataStore
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    store = RepoMetadataStore(conn_str)
    
    repos = store.list_all()
    assert isinstance(repos, list)


def test_update_indexing_status():
    """인덱싱 상태 업데이트 테스트"""
    from src.core.repo_store import RepoMetadataStore
    from src.core.models import RepoMetadata
    
    conn_str = "host=localhost port=5433 dbname=semantica_test user=semantica password=semantica"
    store = RepoMetadataStore(conn_str)
    
    # 저장소 등록
    metadata = RepoMetadata(
        repo_id="test-repo",
        name="Test",
        root_path="/path"
    )
    store.save(metadata)
    
    # 상태 업데이트
    store.update_indexing_status("test-repo", "indexing", progress=0.5)
    store.update_indexing_status("test-repo", "completed")
    
    # 확인
    updated = store.get("test-repo")
    # 실제 DB에서는 indexing_status 조회 가능


def test_repo_metadata_attrs():
    """RepoMetadata attrs 테스트"""
    metadata = RepoMetadata(
        repo_id="test",
        name="Test",
        root_path="/path",
        attrs={
            "git_url": "https://github.com/user/repo",
            "branch": "main",
            "custom_config": {"chunk_size": 512}
        }
    )
    
    assert metadata.attrs["git_url"] == "https://github.com/user/repo"
    assert metadata.attrs["custom_config"]["chunk_size"] == 512

