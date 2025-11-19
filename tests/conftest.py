"""공통 테스트 fixture"""

import logging
import os

import pytest

from src.core.models import RepoMetadata
from src.core.repo_store import RepoMetadataStore

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """pytest 커맨드라인 옵션 추가"""
    parser.addoption(
        "--cleanup-db",
        action="store_true",
        default=False,
        help="테스트 실행 전에 DB의 테스트 데이터를 정리합니다",
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data(request):
    """
    각 테스트 함수 실행 전에 테스트 데이터 정리

    --cleanup-db 옵션을 지정했을 때만 실행됩니다.

    테스트에서 사용하는 repo_id 패턴:
    - test-repo* (test-repo-python, test-repo-typescript 등)
    - test* (test-persistence-python, test-full-integration-typescript 등)
    - my-custom-id* (my-custom-id-python, my-custom-id-typescript 등)
    """
    # --cleanup-db 옵션이 없으면 스킵
    if not request.config.getoption("--cleanup-db"):
        yield
        return

    import psycopg2

    # 환경변수 또는 기본값으로 DB 연결
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "7711"))
    postgres_user = os.getenv("POSTGRES_USER", "semantica")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "semantica")
    postgres_db = os.getenv("POSTGRES_DB", "semantica_codegraph")

    conn_str = (
        f"host={postgres_host} "
        f"port={postgres_port} "
        f"user={postgres_user} "
        f"password={postgres_password} "
        f"dbname={postgres_db}"
    )

    try:
        # DB 연결
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        # 테스트용 repo_id 패턴 삭제
        test_patterns = [
            "test%",
            "my-custom-id%",
        ]

        for pattern in test_patterns:
            # repo_metadata 테이블에서 삭제 (CASCADE로 다른 테이블도 함께 삭제됨)
            cursor.execute("DELETE FROM repo_metadata WHERE repo_id LIKE %s", (pattern,))

            # 명시적으로 다른 테이블도 정리 (CASCADE가 없는 경우를 대비)
            for table in ["code_chunks", "embeddings", "code_edges", "code_nodes"]:
                cursor.execute(f"DELETE FROM {table} WHERE repo_id LIKE %s", (pattern,))

        conn.commit()
        deleted_count = cursor.rowcount

        if deleted_count > 0:
            logger.debug(f"Cleaned up test data (affected rows: {deleted_count})")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.warning(f"Failed to cleanup test data: {e}")

    # 테스트 실행
    yield

    # 테스트 후 정리 (필요시)
    # 현재는 다음 테스트 전에 정리하므로 후처리는 불필요


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
        attrs={},
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
            attrs={},
        )
        store.save(metadata)
        return metadata

    return _ensure


@pytest.fixture
def python_test_repo():
    """Python 프로젝트 fixture 디렉토리 경로 반환"""
    from pathlib import Path

    # 현재 파일의 위치에서 fixture 디렉토리 찾기
    current_dir = Path(__file__).parent
    fixture_path = current_dir / "fixtures" / "python_project"

    if not fixture_path.exists():
        pytest.skip(f"Python fixture not found at {fixture_path}")

    return fixture_path


@pytest.fixture
def typescript_test_repo():
    """TypeScript 프로젝트 fixture 디렉토리 경로 반환"""
    from pathlib import Path

    # 현재 파일의 위치에서 fixture 디렉토리 찾기
    current_dir = Path(__file__).parent
    fixture_path = current_dir / "fixtures" / "typescript_project"

    if not fixture_path.exists():
        pytest.skip(f"TypeScript fixture not found at {fixture_path}")

    return fixture_path
