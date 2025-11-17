"""HTTP API 서버 진입점"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.bootstrap import create_bootstrap

# Bootstrap 인스턴스 생성 (전역)
bootstrap = create_bootstrap()

app = FastAPI(
    title="Semantica Codegraph API",
    description="Code graph indexing and search system API",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """헬스 체크"""
    return {"status": "ok", "service": "semantica-codegraph"}


@app.get("/health")
def health():
    """상세 헬스 체크"""
    return {
        "status": "ok",
        "database": "connected",  # TODO: 실제 DB 연결 확인
    }


# 라우터 등록
from .routes import repos, search, nodes

app.include_router(repos.router, prefix="/api/repos", tags=["repositories"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(nodes.router, prefix="/api/nodes", tags=["nodes"])


def main():
    """서버 실행"""
    import uvicorn
    
    uvicorn.run(
        "apps.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
