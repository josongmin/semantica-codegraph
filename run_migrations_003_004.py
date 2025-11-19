"""003, 004 마이그레이션 실행 스크립트"""

import sys
import psycopg2

def run_migrations():
    """마이그레이션 실행"""
    
    # DB 연결 (docker-compose.yml 설정에 맞춤)
    print("🔌 DB 연결 중...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=7711,  # docker-compose.yml의 POSTGRES_PORT
            dbname='semantica_codegraph',
            user='semantica',
            password='semantica'
        )
        print("✅ DB 연결 성공!\n")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        print("\n💡 Docker 컨테이너가 실행 중인지 확인하세요:")
        print("   docker ps | grep semantica-postgres")
        sys.exit(1)
    
    try:
        with conn.cursor() as cur:
            # 003 마이그레이션 (SymbolIndex)
            print("=" * 70)
            print("📦 003 마이그레이션: SymbolIndex 인덱스 생성")
            print("=" * 70)
            
            try:
                with open('migrations/003_add_symbol_indices.sql', 'r') as f:
                    sql_003 = f.read()
                
                cur.execute(sql_003)
                conn.commit()
                
                print("✅ idx_nodes_name_lower 생성")
                print("✅ idx_nodes_kind_name 생성")
                print("✅ idx_nodes_file_kind 생성")
                print("✅ idx_nodes_decorators 생성 (GIN)")
                print("✅ idx_nodes_name_trgm 생성 (Trigram)")
                print("✅ pg_trgm 확장 설치")
                print()
                
            except Exception as e:
                print(f"⚠️  003 마이그레이션 실행 중 오류: {e}")
                if "already exists" in str(e):
                    print("   (인덱스가 이미 존재합니다 - 무시해도 됩니다)")
                else:
                    raise
            
            # 004 마이그레이션 (RouteIndex)
            print("=" * 70)
            print("📦 004 마이그레이션: RouteIndex 테이블 생성")
            print("=" * 70)
            
            try:
                with open('migrations/004_create_route_index.sql', 'r') as f:
                    sql_004 = f.read()
                
                cur.execute(sql_004)
                conn.commit()
                
                print("✅ route_index 테이블 생성")
                print("✅ idx_route_method_path 생성")
                print("✅ idx_route_path_pattern 생성")
                print("✅ idx_route_file 생성")
                print("✅ idx_route_framework 생성")
                print()
                
            except Exception as e:
                print(f"⚠️  004 마이그레이션 실행 중 오류: {e}")
                if "already exists" in str(e):
                    print("   (테이블이 이미 존재합니다 - 무시해도 됩니다)")
                else:
                    raise
            
            # 확인
            print("=" * 70)
            print("🔍 마이그레이션 확인")
            print("=" * 70)
            
            # 인덱스 확인
            cur.execute("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE indexname LIKE 'idx_nodes_%' 
                   OR indexname LIKE 'idx_route_%'
                ORDER BY tablename, indexname
            """)
            indexes = cur.fetchall()
            
            print(f"\n생성된 인덱스 ({len(indexes)}개):")
            for idx_name, table_name in indexes:
                print(f"  - {table_name}.{idx_name}")
            
            # 테이블 확인
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'route_index'
            """)
            route_table = cur.fetchone()
            
            if route_table:
                print(f"\n✅ route_index 테이블 존재 확인")
                
                # route_index 컬럼 확인
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'route_index'
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                print(f"   컬럼 ({len(columns)}개):")
                for col_name, data_type in columns:
                    print(f"     - {col_name} ({data_type})")
            else:
                print("\n❌ route_index 테이블이 생성되지 않았습니다")
            
            print("\n" + "=" * 70)
            print("🎉 마이그레이션 완료!")
            print("=" * 70)
            print()
            print("다음 단계:")
            print("  1. API 재시작: ./run_api.sh")
            print("  2. 재인덱싱:")
            print("     curl -X POST http://localhost:8000/api/repos \\")
            print("       -H 'Content-Type: application/json' \\")
            print("       -d '{")
            print('         "repo_id": "codegraph",')
            print('         "repo_path": ".",')
            print('         "name": "codegraph"')
            print("       }'")
            print()
            print("  3. 테스트:")
            print("     curl 'http://localhost:8000/hybrid/symbols?repo_id=codegraph&query=HybridRetriever'")
            print("     curl 'http://localhost:8000/hybrid/endpoints?repo_id=codegraph'")
            print()
            
    except Exception as e:
        print(f"\n❌ 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        sys.exit(1)
        
    finally:
        conn.close()
        print("🔌 DB 연결 종료")

if __name__ == "__main__":
    run_migrations()

