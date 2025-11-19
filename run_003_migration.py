"""003 마이그레이션 실행"""

import psycopg2

# DB 연결 설정
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="semantica",
    user="semantica",
    password="semantica"
)

try:
    with conn.cursor() as cur:
        # 마이그레이션 파일 읽기
        with open("migrations/003_add_symbol_indices.sql", "r") as f:
            sql = f.read()
        
        # 실행
        cur.execute(sql)
        conn.commit()
        
        print("✅ 003 마이그레이션 완료!")
        print("   - idx_nodes_name_lower")
        print("   - idx_nodes_kind_name")
        print("   - idx_nodes_file_kind")
        print("   - idx_nodes_decorators")
        print("   - idx_nodes_name_trgm")
        
finally:
    conn.close()

