#!/usr/bin/env python3
"""엣지 생성 확인 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config


def check_edges():
    """데이터베이스의 엣지 확인"""
    print("=" * 60)
    print("엣지 생성 확인")
    print("=" * 60)

    config = Config()

    try:
        import psycopg2

        # DB 연결
        conn_str = (
            f"host={config.postgres_host} "
            f"port={config.postgres_port} "
            f"dbname={config.postgres_db} "
            f"user={config.postgres_user} "
            f"password={config.postgres_password}"
        )

        print(f"\nDB 연결 중: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}")
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()

        # 전체 엣지 개수
        cur.execute("SELECT COUNT(*) FROM code_edges")
        total_edges = cur.fetchone()[0]
        print(f"\n전체 엣지: {total_edges}개")

        if total_edges == 0:
            print("\n⚠️  엣지가 없습니다.")
            print("   - 저장소를 인덱싱했나요?")
            print("   - 인덱싱이 성공적으로 완료되었나요?")
            print("   - 파서가 관계를 추출했나요?")
            return False

        # 저장소별 엣지 개수
        cur.execute("""
            SELECT repo_id, COUNT(*)
            FROM code_edges
            GROUP BY repo_id
            ORDER BY COUNT(*) DESC
        """)

        repo_edges = cur.fetchall()
        if repo_edges:
            print("\n저장소별 엣지:")
            for repo_id, count in repo_edges:
                print(f"  - {repo_id}: {count}개")

        # 엣지 타입별 분포
        cur.execute("""
            SELECT type, COUNT(*)
            FROM code_edges
            GROUP BY type
            ORDER BY COUNT(*) DESC
        """)

        edge_types = cur.fetchall()
        if edge_types:
            print("\n엣지 타입별 분포:")
            for edge_type, count in edge_types:
                print(f"  - {edge_type}: {count}개")

        # 엣지 예시 (상세)
        cur.execute("""
            SELECT
                e.repo_id,
                e.src_id,
                e.dst_id,
                e.type,
                n1.name as src_name,
                n1.kind as src_kind,
                n2.name as dst_name,
                n2.kind as dst_kind
            FROM code_edges e
            JOIN code_nodes n1 ON e.src_id = n1.id AND e.repo_id = n1.repo_id
            JOIN code_nodes n2 ON e.dst_id = n2.id AND e.repo_id = n2.repo_id
            LIMIT 10
        """)

        edges = cur.fetchall()
        if edges:
            print("\n엣지 예시 (최대 10개):")
            for repo_id, _src_id, _dst_id, edge_type, src_name, src_kind, dst_name, dst_kind in edges:
                print(f"  - {src_name}({src_kind}) --[{edge_type}]--> {dst_name}({dst_kind})")

        conn.close()

        print("\n" + "=" * 60)
        print("✅ 엣지가 정상적으로 구성되어 있습니다!")
        return True

    except psycopg2.OperationalError as e:
        print(f"\n❌ DB 연결 실패: {e}")
        print("\n해결 방법:")
        print("  1. PostgreSQL이 실행 중인지 확인")
        print("  2. docker-compose up -d로 DB 시작")
        print("  3. .env 파일의 DB 설정 확인")
        return False
    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_edges()
    sys.exit(0 if success else 1)

