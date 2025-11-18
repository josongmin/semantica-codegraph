"""설정 관련 코드"""

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "mydb"
    username: str = "user"
    password: str = "password"

    @property
    def connection_string(self) -> str:
        """연결 문자열 생성"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class APIConfig:
    """API 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class CacheConfig:
    """캐시 설정"""
    enabled: bool = True
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    ttl: int = 3600


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    app_name: str = "MyApp"
    version: str = "1.0.0"
    environment: str = "development"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """환경 변수로부터 설정 생성"""
        config = cls()

        # 데이터베이스 설정
        config.database.host = os.getenv("DB_HOST", config.database.host)
        config.database.port = int(os.getenv("DB_PORT", str(config.database.port)))
        config.database.database = os.getenv("DB_NAME", config.database.database)
        config.database.username = os.getenv("DB_USER", config.database.username)
        config.database.password = os.getenv("DB_PASSWORD", config.database.password)

        # API 설정
        config.api.host = os.getenv("API_HOST", config.api.host)
        config.api.port = int(os.getenv("API_PORT", str(config.api.port)))
        config.api.debug = os.getenv("DEBUG", "False").lower() == "true"

        # 환경 설정
        config.environment = os.getenv("ENVIRONMENT", config.environment)

        return config

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug
            }
        }


class ConfigManager:
    """설정 관리자"""

    def __init__(self, config: AppConfig | None = None):
        self._config = config or AppConfig()

    @property
    def config(self) -> AppConfig:
        """설정 반환"""
        return self._config

    def update(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        return value

    def reload(self) -> None:
        """환경 변수로부터 설정 재로드"""
        self._config = AppConfig.from_env()


# 전역 설정 인스턴스
config_manager = ConfigManager()

