#!/usr/bin/env python
"""PostgreSQL 연결 정리"""

import psycopg2

print("PostgreSQL 연결 정리 중...")

try:
    # postgres DB에 연결 (시스템 DB)
    conn = psycopg2.connect(
        host="localhost",
        port=7711,
        user="semantica",
        password="semantica",
        database="postgres"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # 1. 현재 연결 수 확인
    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname IN ('semantica', 'semantica_codegraph');")
    before_count = cur.fetchone()[0]
    print(f"정리 전 연결 수: {before_count}")
    
    # 2. idle 연결 종료
    cur.execute("""
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE datname IN ('semantica', 'semantica_codegraph')
          AND state = 'idle'
          AND pid <> pg_backend_pid()
    """)
    
    terminated = cur.fetchall()
    print(f"idle 연결 {len(terminated)}개 종료")
    
    # 3. 정리 후 연결 수 확인
    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname IN ('semantica', 'semantica_codegraph');")
    after_count = cur.fetchone()[0]
    print(f"정리 후 연결 수: {after_count}")
    
    # 4. max_connections 확인
    cur.execute("SHOW max_connections;")
    max_conn = cur.fetchone()[0]
    print(f"최대 연결 수: {max_conn}")
    
    cur.close()
    conn.close()
    
    print("\n✅ 연결 정리 완료!")
    print(f"   {before_count} → {after_count} ({before_count - after_count}개 정리)")
    
except Exception as e:
    print(f"❌ 에러: {e}")
    import traceback
    traceback.print_exc()

