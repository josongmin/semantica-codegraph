"""엣지 케이스 및 이상한 패턴들"""

import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

# 복잡한 타입 별칭
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

NestedDict = dict[str, str | int | dict[str, Any] | list[Any]]
ComplexType = Union[list[dict[str, int | str | None]], dict[str, list[int]]]


# TypedDict 사용
class UserData(TypedDict, total=False):
    name: str
    age: int
    email: str | None
    metadata: dict[str, Any]


# 복잡한 Enum
class Status(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

    @classmethod
    def from_string(cls, value: str) -> "Status":
        """문자열로부터 Status 생성"""
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.PENDING


# 메타클래스
class SingletonMeta(type):
    """싱글톤 메타클래스"""
    _instances: dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """싱글톤 데이터베이스 연결"""

    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False


# 복잡한 제네릭 클래스
class Container(Generic[T]):
    """제네릭 컨테이너"""

    def __init__(self, value: T):
        self._value = value

    def get(self) -> T:
        return self._value

    def set(self, value: T) -> None:
        self._value = value

    def map(self, func: Callable[[T], K]) -> "Container[K]":
        return Container(func(self._value))


class MultiContainer(Generic[T, K]):
    """다중 타입 파라미터 제네릭"""

    def __init__(self, first: T, second: K):
        self.first = first
        self.second = second

    def swap(self) -> "MultiContainer[K, T]":
        return MultiContainer(self.second, self.first)


# 오버로드 사용
class Calculator:
    """오버로드를 사용하는 계산기"""

    @overload
    def add(self, x: int, y: int) -> int: ...

    @overload
    def add(self, x: str, y: str) -> str: ...

    @overload
    def add(self, x: list[int], y: list[int]) -> list[int]: ...

    def add(self, x: int | str | list[int], y: int | str | list[int]) -> int | str | list[int]:
        """타입에 따라 다른 덧셈 수행"""
        if isinstance(x, int) and isinstance(y, int) or isinstance(x, str) and isinstance(y, str):
            return x + y
        elif isinstance(x, list) and isinstance(y, list):
            return [a + b for a, b in zip(x, y, strict=False)]
        raise TypeError("지원하지 않는 타입")


# 복잡한 데코레이터 체이닝
def validate_input(validator: Callable[[Any], bool]):
    """입력 검증 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if not validator(arg):
                    raise ValueError(f"Invalid input: {arg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    """실행 로깅 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper


def retry_on_failure(max_retries: int = 3):
    """실패 시 재시도 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
            return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
            return None

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


@validate_input(lambda x: isinstance(x, int) and x > 0)
@log_execution
@retry_on_failure(max_retries=2)
def complex_function(value: int) -> int:
    """복잡한 데코레이터가 적용된 함수"""
    return value * 2


# 동적 속성을 가진 클래스
class DynamicClass:
    """동적으로 속성을 추가할 수 있는 클래스"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """존재하지 않는 속성 접근 시 기본값 반환"""
        return f"<default_{name}>"

    def __setattr__(self, name: str, value: Any) -> None:
        """속성 설정 시 검증"""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """속성 삭제 시 특별 처리"""
        if name.startswith("protected_"):
            raise AttributeError(f"Cannot delete protected attribute: {name}")
        object.__delattr__(self, name)


# 복잡한 예외 처리
class CustomError(Exception):
    """커스텀 예외"""

    def __init__(self, message: str, code: int = 0):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ValidationError(CustomError):
    """검증 에러"""
    pass


class ProcessingError(CustomError):
    """처리 에러"""
    pass


def risky_operation(data: Any) -> Any:
    """예외를 발생시킬 수 있는 작업"""
    if not data:
        raise ValidationError("Data is required", code=1001)

    if isinstance(data, str) and len(data) > 1000:
        raise ProcessingError("Data too large", code=2001)

    try:
        result = int(data) if isinstance(data, str) else data
        return result * 2
    except (ValueError, TypeError) as e:
        raise ProcessingError(f"Failed to process: {e}", code=2002) from e


# 복잡한 람다 및 함수형 패턴
def create_filter_function(threshold: int) -> Callable[[list[int]], list[int]]:
    """필터 함수를 생성하는 팩토리"""
    return lambda items: [x for x in items if x > threshold]


def compose(*functions: Callable) -> Callable:
    """함수 합성"""
    def composed(x: Any) -> Any:
        result = x
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


# 복잡한 컨텍스트 매니저
@contextmanager
def managed_resource(name: str):
    """리소스 관리 컨텍스트 매니저"""
    print(f"Acquiring resource: {name}")
    resource = {"name": name, "acquired": True}
    try:
        yield resource
    finally:
        print(f"Releasing resource: {name}")
        resource["acquired"] = False


class ResourceManager:
    """리소스 관리자 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.acquired = False

    def __enter__(self):
        self.acquired = True
        print(f"Entering context: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.acquired = False
        print(f"Exiting context: {self.name}")
        return False  # 예외를 전파


# 복잡한 프로토콜
@runtime_checkable
class Drawable(Protocol):
    """그리기 가능한 프로토콜"""

    def draw(self) -> None: ...

    def get_area(self) -> float: ...


@runtime_checkable
class Movable(Protocol):
    """이동 가능한 프로토콜"""

    def move(self, x: float, y: float) -> None: ...


class Shape:
    """도형 클래스"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def draw(self) -> None:
        print(f"Drawing shape at ({self.x}, {self.y})")

    def get_area(self) -> float:
        return 0.0


class Circle(Shape):
    """원 클래스"""

    def __init__(self, x: float, y: float, radius: float):
        super().__init__(x, y)
        self.radius = radius

    def get_area(self) -> float:
        return 3.14159 * self.radius ** 2

    def move(self, x: float, y: float) -> None:
        self.x += x
        self.y += y


# 복잡한 네스팅
class OuterClass:
    """외부 클래스"""

    class InnerClass:
        """내부 클래스"""

        class NestedClass:
            """중첩 클래스"""

            @staticmethod
            def deeply_nested_method() -> str:
                """깊게 중첩된 메서드"""
                return "deeply nested"

        def inner_method(self) -> str:
            """내부 메서드"""
            return "inner"

    def outer_method(self) -> "OuterClass.InnerClass":
        """외부 메서드"""
        return OuterClass.InnerClass()


# 이상한 네이밍 패턴
class _PrivateClass:
    """프라이빗 클래스 (언더스코어로 시작)"""
    pass


class __DunderClass__:
    """던더 클래스"""
    def __init__(self):
        self.__private_attr = "secret"


class CamelCaseClass:
    """카멜케이스 클래스"""
    def mixedCaseMethod(self) -> None:
        """혼합 케이스 메서드"""
        pass


class snake_case_class:
    """스네이크 케이스 클래스"""
    def method_name(self) -> None:
        pass


# 특수 메서드 오버로딩
class Vector:
    """벡터 클래스 (연산자 오버로딩)"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")

    def __iter__(self):
        yield self.x
        yield self.y


# 복잡한 비동기 패턴
class AsyncProcessor:
    """비동기 프로세서"""

    async def process_item(self, item: Any) -> Any:
        """아이템 처리"""
        await asyncio.sleep(0.1)
        return item * 2

    async def process_batch(self, items: list[Any]) -> list[Any]:
        """배치 처리"""
        tasks = [self.process_item(item) for item in items]
        return await asyncio.gather(*tasks)

    async def process_with_retry(self, item: Any, max_retries: int = 3) -> Any:
        """재시도와 함께 처리"""
        for attempt in range(max_retries):
            try:
                return await self.process_item(item)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))


# 데코레이터 팩토리
def register_handler(event_type: str):
    """핸들러 등록 데코레이터 팩토리"""
    handlers: dict[str, list[Callable]] = {}

    def decorator(func: Callable) -> Callable:
        if event_type not in handlers:
            handlers[event_type] = []
        handlers[event_type].append(func)
        return func

    decorator.handlers = handlers
    return decorator


@register_handler("user_created")
def handle_user_created(user: dict[str, Any]) -> None:
    """사용자 생성 핸들러"""
    print(f"User created: {user}")


@register_handler("user_deleted")
def handle_user_deleted(user_id: str) -> None:
    """사용자 삭제 핸들러"""
    print(f"User deleted: {user_id}")


# 복잡한 타입 가드
def is_valid_user(obj: Any) -> bool:
    """타입 가드 함수"""
    return (
        isinstance(obj, dict) and
        "name" in obj and
        "age" in obj and
        isinstance(obj["name"], str) and
        isinstance(obj["age"], int)
    )


def process_user_data(data: Any) -> dict[str, Any] | None:
    """타입 가드를 사용하는 함수"""
    if is_valid_user(data):
        return {"processed": True, "user": data}
    return None


# 이상한 주석 패턴
class CommentedClass:
    """
    매우 긴 독스트링을 가진 클래스.

    이 클래스는 여러 줄에 걸친 설명을 가지고 있으며,
    복잡한 기능을 수행합니다.

    Args:
        value: 초기값

    Returns:
        CommentedClass 인스턴스

    Raises:
        ValueError: 잘못된 값이 전달될 경우

    Example:
        >>> obj = CommentedClass(42)
        >>> print(obj.value)
        42
    """

    def __init__(self, value: int):
        # 인라인 주석
        self.value = value  # 값 설정

    def method(self):
        """메서드 독스트링"""
        # TODO: 구현 필요
        # FIXME: 버그 수정 필요
        # NOTE: 중요 참고사항
        # XXX: 위험한 코드
        pass


# 복잡한 조건부 타입 (타입 힌트)
def process_data(data: str | int | list[int]) -> str | int | list[int]:
    """조건부 처리를 하는 함수"""
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return data * 2
    elif isinstance(data, list):
        return [x * 2 for x in data]
    else:
        raise TypeError("Unsupported type")


# Literal 타입 사용
def get_status_color(status: Literal["success", "error", "warning"]) -> str:
    """상태에 따른 색상 반환"""
    colors = {
        "success": "green",
        "error": "red",
        "warning": "yellow"
    }
    return colors[status]


# 복잡한 함수 시그니처
def complex_function_signature(
    required: str,
    optional: int | None = None,
    *args: str,
    keyword_only: bool = False,
    **kwargs: Any
) -> dict[str, Any]:
    """복잡한 함수 시그니처"""
    return {
        "required": required,
        "optional": optional,
        "args": args,
        "keyword_only": keyword_only,
        "kwargs": kwargs
    }

