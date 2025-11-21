"""HTTP API 서버 진입점"""

import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.bootstrap import create_bootstrap

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Bootstrap 인스턴스 생성 (전역)
bootstrap = create_bootstrap()

app = FastAPI(
    title="Semantica Codegraph API",
    description="Code graph indexing and search system API",
    version="0.1.0",
)

# OpenTelemetry FastAPI 계측
if bootstrap.config.otel_enabled:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed")

# OpenLIT 초기화 (LLM 호출 추적)
if bootstrap.config.otel_enabled:
    try:
        import openlit

        openlit.init(
            otlp_endpoint=bootstrap.config.otel_endpoint,
            application_name=bootstrap.config.otel_service_name,
            environment=bootstrap.config.environment,
        )
        logger.info(f"OpenLIT initialized: endpoint={bootstrap.config.otel_endpoint}")
    except ImportError:
        logger.warning("openlit not installed - LLM tracing unavailable")


class LoggingMiddleware(BaseHTTPMiddleware):
    """API 요청/응답 로깅 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 요청 정보 로깅
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)

        logger.info(
            f"요청 시작: {method} {path} | "
            f"IP: {client_ip} | "
            f"Query: {query_params if query_params else 'None'}"
        )

        # 요청 본문 로깅 (작은 요청만)
        try:
            body = await request.body()
            if body and len(body) < 1000:  # 1KB 미만만 로깅
                try:
                    body_str = body.decode("utf-8")
                    logger.debug(f"요청 본문: {body_str}")
                except Exception:
                    logger.debug(f"요청 본문 (바이너리): {len(body)} bytes")
            elif body:
                logger.debug(f"요청 본문: {len(body)} bytes (너무 커서 생략)")

            # body를 다시 설정 (한 번만 읽을 수 있음)
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive
        except Exception as e:
            logger.debug(f"요청 본문 읽기 실패: {e}")

        # 응답 처리
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # 응답 정보 로깅
            status_code = response.status_code
            response_size = response.headers.get("content-length", "unknown")

            logger.info(
                f"응답 완료: {method} {path} | "
                f"상태: {status_code} | "
                f"소요시간: {process_time:.3f}s | "
                f"크기: {response_size}"
            )

            # 응답 본문 로깅 (에러인 경우만)
            if status_code >= 400:
                try:
                    # 응답 본문을 읽기 위해 복사
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk

                    if response_body and len(response_body) < 2000:  # 2KB 미만만 로깅
                        try:
                            body_str = response_body.decode("utf-8")
                            logger.warning(f"에러 응답 본문: {body_str}")
                        except Exception:
                            logger.warning(f"에러 응답 본문 (바이너리): {len(response_body)} bytes")
                    elif response_body:
                        logger.warning(f"에러 응답 본문: {len(response_body)} bytes (너무 커서 생략)")

                    # 응답 재생성
                    from starlette.responses import Response as StarletteResponse

                    return StarletteResponse(
                        content=response_body,
                        status_code=status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                    )
                except Exception as e:
                    logger.debug(f"응답 본문 읽기 실패: {e}")
                    # 실패해도 원본 응답 반환
                    return response

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"요청 처리 실패: {method} {path} | 소요시간: {process_time:.3f}s | 에러: {str(e)}",
                exc_info=True,
            )
            raise


# 미들웨어 등록 (순서 중요: LoggingMiddleware가 먼저)
app.add_middleware(LoggingMiddleware)

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
from .routes import hybrid, repos

app.include_router(repos.router, prefix="/api/repos", tags=["repositories"])
app.include_router(hybrid.router, prefix="/hybrid", tags=["hybrid"])


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
