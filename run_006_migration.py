#!/usr/bin/env python
"""006 Query Logs Migration 실행"""

import psycopg2
from pathlib import Path

print("="*70)
print("🔧 Migration 006: Query Logs 테이블 생성")
print("="*70)

try:
    conn = psycopg2.connect(
        host="localhost",
        port=7711,
        user="semantica",
        password="semantica",
        database="semantica_codegraph"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Migration SQL 읽기
    migration_file = Path(__file__).parent / "migrations" / "006_query_logs.sql"
    print(f"\n[1] Migration 파일 읽기: {migration_file.name}")
    
    with open(migration_file, "r", encoding="utf-8") as f:
        sql = f.read()
    
    # SQL 구문 분리 실행
    print("\n[2] Migration 실행 중...")
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
    
    for i, stmt in enumerate(statements, 1):
        try:
            cur.execute(stmt)
            if 'CREATE TABLE' in stmt.upper():
                table_name = 'query_logs' if 'query_logs' in stmt else 'node_popularity'
                print(f"  ✓ [{i}/{len(statements)}] {table_name} 테이블 생성")
            elif 'CREATE INDEX' in stmt.upper():
                pass  # 조용히 실행
            elif 'COMMENT' in stmt.upper():
                pass
        except Exception as e:
            if 'already exists' in str(e).lower():
                pass  # 이미 존재하면 무시
            else:
                print(f"  ⚠️  구문 {i} 실패: {e}")
    
    # 확인
    print("\n[3] 테이블 확인")
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_name IN ('query_logs', 'node_popularity')
    """)
    tables = [row[0] for row in cur.fetchall()]
    
    for table in tables:
        print(f"  ✓ {table} 테이블 생성 확인")
    
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("🎉 Migration 006 완료!")
    print("="*70)
    print("\n이제 검색 시 자동으로 query_logs에 기록됩니다!")
    print()
    
except Exception as e:
    print(f"\n❌ Migration 실패: {e}")
    import traceback
    traceback.print_exc()

