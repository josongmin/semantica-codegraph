#!/usr/bin/env python
"""마이그레이션 실행 스크립트"""

import psycopg2

print("="*70)
print("🔧 DB 마이그레이션 실행")
print("="*70)

try:
    # DB 연결
    conn = psycopg2.connect(
        host="localhost",
        port=7711,
        user="semantica",
        password="semantica",
        database="semantica_codegraph"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    print("\n[1] repo_profile 테이블 생성")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS repo_profile (
            repo_id VARCHAR(255) PRIMARY KEY,
            profile_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    print("  ✓ repo_profile 테이블 생성됨")
    
    print("\n[2] file_profile 테이블 생성")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_profile (
            id SERIAL PRIMARY KEY,
            repo_id VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            profile_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(repo_id, file_path)
        )
    """)
    print("  ✓ file_profile 테이블 생성됨")
    
    print("\n[3] code_chunks.metadata 컬럼 추가")
    cur.execute("ALTER TABLE code_chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'")
    print("  ✓ code_chunks.metadata 컬럼 추가됨")
    
    print("\n[4] code_nodes.importance_score 컬럼 추가")
    cur.execute("ALTER TABLE code_nodes ADD COLUMN IF NOT EXISTS importance_score FLOAT DEFAULT 0.0")
    print("  ✓ code_nodes.importance_score 컬럼 추가됨")
    
    print("\n[5] 인덱스 생성")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_metadata ON code_chunks USING GIN (metadata)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_nodes_importance ON code_nodes(repo_id, importance_score DESC)")
    print("  ✓ 인덱스 생성됨")
    
    # 확인
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'code_chunks' AND column_name = 'metadata'
    """)
    if cur.fetchone():
        print("\n✅ code_chunks.metadata 존재 확인")
    
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'code_nodes' AND column_name = 'importance_score'
    """)
    if cur.fetchone():
        print("✅ code_nodes.importance_score 존재 확인")
    
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("🎉 마이그레이션 완료!")
    print("="*70)
    print("\n이제 재인덱싱을 실행하세요:")
    print("  python reindex_with_profiling.py")
    print()
    
except Exception as e:
    print(f"\n❌ 마이그레이션 실패: {e}")
    import traceback
    traceback.print_exc()

