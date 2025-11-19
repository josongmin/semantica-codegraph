"""IndexingPipeline 테스트"""

import logging

import pytest

from src.core.bootstrap import create_bootstrap
from src.core.models import RepoConfig

logger = logging.getLogger(__name__)


def test_pipeline_basic_flow_python(python_test_repo):
    """Python 프로젝트 기본 파이프라인 테스트"""
    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(python_test_repo), name="test-repo-python"
    )

    assert result.status == "completed"
    assert result.total_files > 0
    assert result.processed_files > 0
    assert result.total_nodes > 0
    assert result.total_chunks > 0
    assert result.duration_seconds > 0


def test_pipeline_basic_flow_typescript(typescript_test_repo):
    """TypeScript 프로젝트 기본 파이프라인 테스트"""
    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(typescript_test_repo), name="test-repo-typescript"
    )

    assert result.status == "completed"
    assert result.total_files > 0
    assert result.processed_files > 0
    assert result.total_nodes > 0
    assert result.total_chunks > 0
    assert result.duration_seconds > 0


def test_pipeline_with_config_python(python_test_repo):
    """Python 프로젝트 설정이 있는 파이프라인 테스트"""
    bootstrap = create_bootstrap()

    config = RepoConfig(languages=["python"], include_tests=True)

    result = bootstrap.pipeline.index_repository(
        root_path=str(python_test_repo), name="test-repo-config-python", config=config
    )

    assert result.status == "completed"
    assert result.total_files > 0


def test_pipeline_with_config_typescript(typescript_test_repo):
    """TypeScript 프로젝트 설정이 있는 파이프라인 테스트"""
    bootstrap = create_bootstrap()

    config = RepoConfig(languages=["typescript"], include_tests=True)

    result = bootstrap.pipeline.index_repository(
        root_path=str(typescript_test_repo), name="test-repo-config-typescript", config=config
    )

    assert result.status == "completed"
    assert result.total_files > 0


def test_pipeline_custom_repo_id_python(python_test_repo):
    """Python 프로젝트 커스텀 repo_id 테스트"""
    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(python_test_repo),
        repo_id="my-custom-id-python",
        name="test-repo-custom-python",
    )

    assert result.status == "completed"
    assert result.repo_id == "my-custom-id-python"


def test_pipeline_custom_repo_id_typescript(typescript_test_repo):
    """TypeScript 프로젝트 커스텀 repo_id 테스트"""
    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(typescript_test_repo),
        repo_id="my-custom-id-typescript",
        name="test-repo-custom-typescript",
    )

    assert result.status == "completed"
    assert result.repo_id == "my-custom-id-typescript"


def test_pipeline_empty_directory(tmp_path):
    """빈 디렉토리 테스트"""
    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(root_path=str(tmp_path), name="empty-repo")

    assert result.status == "completed"
    assert result.total_files == 0
    assert result.total_nodes == 0


def test_pipeline_metadata_persistence_python(python_test_repo):
    """Python 프로젝트 메타데이터 저장 테스트"""
    bootstrap = create_bootstrap()

    # 인덱싱
    result = bootstrap.pipeline.index_repository(
        root_path=str(python_test_repo),
        repo_id="test-persistence-python",
        name="test-repo-metadata-python",
    )

    # 메타데이터 조회
    metadata = bootstrap.repo_store.get("test-persistence-python")

    assert metadata is not None
    assert metadata.repo_id == "test-persistence-python"
    assert metadata.name == "test-repo-metadata-python"
    assert metadata.total_files == result.total_files
    assert metadata.total_nodes == result.total_nodes
    assert metadata.total_chunks == result.total_chunks
    assert "python" in metadata.languages


def test_pipeline_metadata_persistence_typescript(typescript_test_repo):
    """TypeScript 프로젝트 메타데이터 저장 테스트"""
    bootstrap = create_bootstrap()

    # 인덱싱
    result = bootstrap.pipeline.index_repository(
        root_path=str(typescript_test_repo),
        repo_id="test-persistence-typescript",
        name="test-repo-metadata-typescript",
    )

    # 메타데이터 조회
    metadata = bootstrap.repo_store.get("test-persistence-typescript")

    assert metadata is not None
    assert metadata.repo_id == "test-persistence-typescript"
    assert metadata.name == "test-repo-metadata-typescript"
    assert metadata.total_files == result.total_files
    assert metadata.total_nodes == result.total_nodes
    assert metadata.total_chunks == result.total_chunks
    assert "typescript" in metadata.languages


def test_pipeline_full_integration_python(python_test_repo):
    """Python 프로젝트 전체 통합 테스트 (DB + API 키 필요)"""
    from src.core.config import Config

    config = Config.from_env()
    if not config.embedding_api_key:
        pytest.skip("EMBEDDING_API_KEY not set")

    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(python_test_repo),
        repo_id="test-full-integration-python",
        name="test-repo-full-python",
    )

    assert result.status == "completed"
    assert result.total_nodes > 0, "최소 1개 이상의 노드가 인덱싱되어야 함"
    assert result.total_chunks > 0, "최소 1개 이상의 청크가 생성되어야 함"

    # 그래프 조회 테스트
    nodes = bootstrap.graph_store.list_nodes("test-full-integration-python")
    assert len(nodes) > 0, f"노드가 조회되어야 함 (인덱싱: {result.total_nodes}개)"

    # 메타데이터 확인
    metadata = bootstrap.repo_store.get("test-full-integration-python")
    assert metadata is not None
    assert metadata.total_nodes == result.total_nodes
    assert metadata.total_chunks == result.total_chunks

    # Lexical 검색 테스트 (선택적)
    try:
        results = bootstrap.lexical_search.search(
            repo_id="test-full-integration-python", query="def", k=5
        )
        # 결과가 있으면 좋지만, lexical 검색이 안 되어도 실패로 처리하지 않음
        logger.info(f"Lexical search results: {len(results)}")
    except Exception as e:
        logger.warning(f"Lexical search not available: {e}")


def test_pipeline_full_integration_typescript(typescript_test_repo):
    """TypeScript 프로젝트 전체 통합 테스트 (DB + API 키 필요)"""
    from src.core.config import Config

    config = Config.from_env()
    if not config.embedding_api_key:
        pytest.skip("EMBEDDING_API_KEY not set")

    bootstrap = create_bootstrap()

    result = bootstrap.pipeline.index_repository(
        root_path=str(typescript_test_repo),
        repo_id="test-full-integration-typescript",
        name="test-repo-full-typescript",
    )

    assert result.status == "completed"
    assert result.total_nodes > 0, "최소 1개 이상의 노드가 인덱싱되어야 함"
    assert result.total_chunks > 0, "최소 1개 이상의 청크가 생성되어야 함"

    # 그래프 조회 테스트
    nodes = bootstrap.graph_store.list_nodes("test-full-integration-typescript")
    assert len(nodes) > 0, f"노드가 조회되어야 함 (인덱싱: {result.total_nodes}개)"

    # 메타데이터 확인
    metadata = bootstrap.repo_store.get("test-full-integration-typescript")
    assert metadata is not None
    assert metadata.total_nodes == result.total_nodes
    assert metadata.total_chunks == result.total_chunks

    # Lexical 검색 테스트 (선택적)
    try:
        results = bootstrap.lexical_search.search(
            repo_id="test-full-integration-typescript", query="function", k=5
        )
        # 결과가 있으면 좋지만, lexical 검색이 안 되어도 실패로 처리하지 않음
        logger.info(f"Lexical search results: {len(results)}")
    except Exception as e:
        logger.warning(f"Lexical search not available: {e}")
