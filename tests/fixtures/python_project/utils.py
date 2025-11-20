"""유틸리티 함수들"""

import json
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def calculate_total(items: list[dict[str, Any]]) -> float:
    """아이템 목록의 총액 계산"""
    total = 0.0
    for item in items:
        total += item.get("price", 0.0)
    return total


def filter_by_price(
    items: list[dict[str, Any]], min_price: float, max_price: float
) -> list[dict[str, Any]]:
    """가격 범위로 필터링"""
    return [item for item in items if min_price <= item.get("price", 0.0) <= max_price]


def group_by_category(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """카테고리별로 그룹화"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        category = item.get("category", "uncategorized")
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(item)
    return grouped


def timing_decorator(func: Callable) -> Callable:
    """실행 시간 측정 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 실행 시간: {end - start:.4f}초")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """재시도 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise last_exception from e
            return None

        return wrapper

    return decorator


@timing_decorator
def process_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """데이터 처리"""
    total = calculate_total(data)
    filtered = filter_by_price(data, 0, 1000)
    grouped = group_by_category(data)

    return {"total": total, "filtered_count": len(filtered), "categories": list(grouped.keys())}


async def fetch_data(url: str) -> dict[str, Any] | None:
    """URL에서 데이터 가져오기"""
    # 실제 구현에서는 HTTP 요청을 수행함
    return {"url": url, "data": []}


async def fetch_multiple(urls: list[str]) -> list[dict[str, Any] | None]:
    """여러 URL에서 데이터 가져오기"""
    results = []
    for url in urls:
        result = await fetch_data(url)
        results.append(result)
    return results


def serialize_to_json(obj: Any) -> str:
    """객체를 JSON 문자열로 직렬화"""
    return json.dumps(obj, default=str, ensure_ascii=False)


def deserialize_from_json(json_str: str) -> Any:
    """JSON 문자열을 객체로 역직렬화"""
    return json.loads(json_str)


class Cache:
    """간단한 캐시 클래스"""

    def __init__(self, max_size: int = 100):
        self._cache: dict[str, Any] = {}
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        """캐시에서 값 가져오기"""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        if len(self._cache) >= self.max_size:
            # 가장 오래된 항목 제거 (간단한 구현)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

    def clear(self) -> None:
        """캐시 비우기"""
        self._cache.clear()

    def size(self) -> int:
        """캐시 크기 반환"""
        return len(self._cache)


def find_max(items: list[T], key: Callable[[T], Any] | None = None) -> T | None:
    """최대값 찾기"""
    if not items:
        return None
    if key:
        return max(items, key=key)
    return max(items)


def find_min(items: list[T], key: Callable[[T], Any] | None = None) -> T | None:
    """최소값 찾기"""
    if not items:
        return None
    if key:
        return min(items, key=key)
    return min(items)
