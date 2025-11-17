"""공통 테스트 fixture"""
import pytest
from datetime import datetime

from src.core.models import RepoMetadata
from src.core.repo_store import RepoMetadataStore


@pytest.fixture
def test_repo_metadata():
    """테스트용 저장소 메타데이터"""
    return RepoMetadata(
        repo_id="test-repo",
        name="Test Repository",
        root_path="/path/to/repo",
        languages=["python"],
        total_files=0,
        total_nodes=0,
        total_chunks=0,
        attrs={}
    )


@pytest.fixture
def ensure_test_repo():
    """테스트 저장소가 DB에 존재하도록 보장"""
    def _ensure(conn_str: str, repo_id: str = "test-repo"):
        store = RepoMetadataStore(conn_str)
        metadata = RepoMetadata(
            repo_id=repo_id,
            name="Test Repository",
            root_path="/path/to/repo",
            languages=["python"],
            total_files=0,
            total_nodes=0,
            total_chunks=0,
            attrs={}
        )
        store.save(metadata)
        return metadata
    return _ensure

