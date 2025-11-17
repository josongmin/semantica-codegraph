"""실제 DB에서 검색 테스트"""
import os
from src.core.config import Config
from src.core.models import CodeChunk
from src.embedding.service import EmbeddingService
from src.embedding.store_pgvector import PgVectorStore
from src.chunking.store import PostgresChunkStore
from src.core.repo_store import RepoMetadataStore

# 설정 로드
config = Config.from_env()

if not config.embedding_api_key or config.embedding_api_key == "your_mistral_api_key_here":
    print("❌ EMBEDDING_API_KEY가 설정되지 않았습니다.")
    exit(1)

# 연결 문자열
conn_str = (
    f"host={config.postgres_host} "
    f"port={config.postgres_port} "
    f"dbname=semantica_test "
    f"user={config.postgres_user} "
    f"password={config.postgres_password}"
)

print("=" * 60)
print("실제 DB 검색 테스트")
print("=" * 60)

# 1. 저장소 메타데이터 생성
print("\n1. 저장소 메타데이터 생성 중...")
repo_store = RepoMetadataStore(conn_str)
from src.core.models import RepoMetadata
from datetime import datetime

repo_metadata = RepoMetadata(
    repo_id="search-test-repo",
    name="Search Test Repository",
    root_path="/test/path",
    languages=["python"],
    total_files=0,
    total_nodes=0,
    total_chunks=0,
    attrs={}
)
repo_store.save(repo_metadata)
print("✓ 저장소 메타데이터 생성 완료")

# 2. 테스트 청크 생성
print("\n2. 테스트 청크 생성 중...")
test_chunks = [
    CodeChunk(
        repo_id="search-test-repo",
        id="chunk-add",
        node_id="node-add",
        file_path="calculator.py",
        span=(0, 0, 5, 0),
        language="python",
        text="def add(a, b):\n    \"\"\"두 수를 더합니다\"\"\"\n    return a + b",
        attrs={"docstring": "두 수를 더합니다"}
    ),
    CodeChunk(
        repo_id="search-test-repo",
        id="chunk-multiply",
        node_id="node-multiply",
        file_path="calculator.py",
        span=(6, 0, 10, 0),
        language="python",
        text="def multiply(a, b):\n    \"\"\"두 수를 곱합니다\"\"\"\n    return a * b",
        attrs={"docstring": "두 수를 곱합니다"}
    ),
    CodeChunk(
        repo_id="search-test-repo",
        id="chunk-format",
        node_id="node-format",
        file_path="utils.py",
        span=(0, 0, 3, 0),
        language="python",
        text="def format_number(n):\n    \"\"\"숫자를 문자열로 변환\"\"\"\n    return str(n)",
        attrs={"docstring": "숫자를 문자열로 변환"}
    ),
]
print(f"✓ {len(test_chunks)}개 청크 생성 완료")

# 3. 청크 저장
print("\n3. 청크를 DB에 저장 중...")
chunk_store = PostgresChunkStore(conn_str)
chunk_store.save_chunks(test_chunks)
print("✓ 청크 저장 완료")

# 4. 임베딩 생성
print("\n4. 임베딩 생성 중... (Mistral API 호출)")
embedding_service = EmbeddingService(
    model=config.embedding_model,
    api_key=config.embedding_api_key,
    api_base=config.mistral_api_base,
    dimension=config.embedding_dimension
)
vectors = embedding_service.embed_chunks(test_chunks)
print(f"✓ {len(vectors)}개 임베딩 생성 완료 (차원: {len(vectors[0])})")

# 5. 벡터 저장
print("\n5. 벡터를 DB에 저장 중...")
vector_store = PgVectorStore(
    connection_string=conn_str,
    embedding_dimension=embedding_service.get_dimension(),
    model_name=config.embedding_model.value
)
chunk_ids = [chunk.id for chunk in test_chunks]
vector_store.save_embeddings("search-test-repo", chunk_ids, vectors)
print("✓ 벡터 저장 완료")

# 6. 검색 테스트
print("\n" + "=" * 60)
print("검색 테스트")
print("=" * 60)

queries = [
    "두 숫자를 더하는 함수",
    "곱셈을 수행하는 함수",
    "숫자를 문자열로 바꾸는 함수"
]

for query_text in queries:
    print(f"\n🔍 쿼리: '{query_text}'")
    print("-" * 60)
    
    # 쿼리 임베딩 생성
    query_vector = embedding_service.embed_text(query_text)
    
    # 검색
    results = vector_store.search_by_vector(
        repo_id="search-test-repo",
        vector=query_vector,
        k=3
    )
    
    print(f"검색 결과: {len(results)}개")
    for i, result in enumerate(results, 1):
        chunk = chunk_store.get_chunk("search-test-repo", result.chunk_id)
        print(f"\n  [{i}] 유사도: {result.score:.4f}")
        print(f"      청크 ID: {result.chunk_id}")
        print(f"      파일: {result.file_path}")
        if chunk:
            print(f"      코드: {chunk.text[:50]}...")

# 7. 필터 검색 테스트
print("\n" + "=" * 60)
print("필터 검색 테스트 (calculator.py만)")
print("=" * 60)

query_text = "계산 함수"
query_vector = embedding_service.embed_text(query_text)

filtered_results = vector_store.search_by_vector(
    repo_id="search-test-repo",
    vector=query_vector,
    k=3,
    filters={"file_path": "calculator.py"}
)

print(f"검색 결과: {len(filtered_results)}개")
for i, result in enumerate(filtered_results, 1):
    chunk = chunk_store.get_chunk("search-test-repo", result.chunk_id)
    print(f"\n  [{i}] 유사도: {result.score:.4f}")
    print(f"      청크 ID: {result.chunk_id}")
    print(f"      파일: {result.file_path}")
    if chunk:
        print(f"      코드: {chunk.text[:50]}...")

print("\n" + "=" * 60)
print("✅ 모든 테스트 완료!")
print("=" * 60)

