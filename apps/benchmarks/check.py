#!/usr/bin/env python3
"""
벤치마크 사전 체크 - 실행 전 환경 확인

사용법:
    python apps/benchmarks/check.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_docker():
    """Docker 실행 확인"""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def check_postgres():
    """PostgreSQL 연결 확인"""
    try:
        from src.core.config import Config
        config = Config.from_env()

        import psycopg2
        conn = psycopg2.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            user=config.postgres_user,
            password=config.postgres_password,
            dbname=config.postgres_db,
            connect_timeout=3
        )
        conn.close()
        return True
    except:
        return False


def check_meilisearch():
    """MeiliSearch 연결 확인"""
    try:
        from src.core.config import Config
        config = Config.from_env()

        import requests
        response = requests.get(
            f"{config.meilisearch_url}/health",
            timeout=3
        )
        return response.status_code == 200
    except:
        return False


def check_indexed_repos():
    """인덱싱된 저장소 확인"""
    try:
        from src.core.bootstrap import Bootstrap
        from src.core.config import Config

        config = Config.from_env()
        bootstrap = Bootstrap(config)
        repo_store = bootstrap.repo_store()

        repos = repo_store.list_repos()
        return len(repos) > 0, repos
    except Exception:
        return False, []


def check_env_file():
    """환경 설정 파일 확인"""
    env_file = project_root / ".env"
    return env_file.exists()


def main():
    print("=" * 80)
    print("🔍 벤치마크 환경 체크")
    print("=" * 80)
    print()

    checks = []

    # 1. Docker
    print("1️⃣  Docker 확인...", end=" ")
    if check_docker():
        print("✅")
        checks.append(True)
    else:
        print("❌")
        print("   Docker가 실행 중이 아닙니다.")
        print("   → docker 명령어를 사용할 수 없습니다.")
        checks.append(False)

    # 2. .env 파일
    print("2️⃣  환경 설정 확인...", end=" ")
    if check_env_file():
        print("✅")
        checks.append(True)
    else:
        print("⚠️")
        print("   .env 파일이 없습니다.")
        print("   → 기본값이 사용됩니다.")
        checks.append(None)

    # 3. PostgreSQL
    print("3️⃣  PostgreSQL 연결...", end=" ")
    if check_postgres():
        print("✅")
        checks.append(True)
    else:
        print("❌")
        print("   PostgreSQL에 연결할 수 없습니다.")
        print("   → docker-compose up -d 를 실행하세요.")
        checks.append(False)

    # 4. MeiliSearch
    print("4️⃣  MeiliSearch 연결...", end=" ")
    if check_meilisearch():
        print("✅")
        checks.append(True)
    else:
        print("❌")
        print("   MeiliSearch에 연결할 수 없습니다.")
        print("   → docker-compose up -d 를 실행하세요.")
        checks.append(False)

    # 5. 인덱싱된 저장소
    print("5️⃣  인덱싱된 저장소...", end=" ")
    has_repos, repos = check_indexed_repos()
    if has_repos:
        print(f"✅ ({len(repos)}개)")
        for repo in repos[:3]:
            print(f"   - {repo.repo_id}")
        if len(repos) > 3:
            print(f"   ... 외 {len(repos) - 3}개")
        checks.append(True)
    else:
        print("❌")
        print("   인덱싱된 저장소가 없습니다.")
        print("   → semantica index /path/to/repo 를 실행하세요.")
        checks.append(False)

    print()
    print("=" * 80)

    # 결과 요약
    sum(1 for c in checks if c is True)
    failed = sum(1 for c in checks if c is False)

    if failed == 0:
        print("✅ 모든 체크 통과! 벤치마크를 실행할 수 있습니다.")
        print()
        print("실행:")
        print("  ./benchmark")
        print("  또는")
        print("  python apps/benchmarks/run.py")
        return 0
    else:
        print(f"❌ {failed}개 항목 실패")
        print()
        print("문제 해결:")
        if not checks[2] or not checks[3]:  # PostgreSQL or MeiliSearch
            print("  1. Docker 서비스 시작:")
            print("     cd /Users/josongmin/Documents/jo-codes/semantica-codegraph")
            print("     docker-compose up -d")
            print()
        if not checks[4]:  # 저장소
            print("  2. 저장소 인덱싱:")
            print("     semantica index /path/to/your/repo")
            print("     또는")
            print("     python -m apps.cli.main index /path/to/your/repo")
            print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

