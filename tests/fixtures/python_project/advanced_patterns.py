"""고급 패턴 및 실전 케이스"""

import asyncio
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial, reduce, wraps
from threading import Lock
from typing import Annotated, Any, Generic, Protocol, TypeVar

# 고급 제네릭 패턴
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")


class Monad(Generic[T]):
    """모나드 패턴 구현"""

    def __init__(self, value: T):
        self._value = value

    def bind(self, func: Callable[[T], "Monad[R]"]) -> "Monad[R]":
        """바인드 연산"""
        return func(self._value)

    def map(self, func: Callable[[T], R]) -> "Monad[R]":
        """맵 연산"""
        return Monad(func(self._value))

    @property
    def value(self) -> T:
        return self._value


class Maybe(Generic[T]):
    """Maybe 모나드"""

    def __init__(self, value: T | None = None):
        self._value = value

    @classmethod
    def just(cls, value: T) -> "Maybe[T]":
        return cls(value)

    @classmethod
    def nothing(cls) -> "Maybe[T]":
        return cls(None)

    def is_just(self) -> bool:
        return self._value is not None

    def is_nothing(self) -> bool:
        return self._value is None

    def map(self, func: Callable[[T], R]) -> "Maybe[R]":
        if self.is_just():
            return Maybe.just(func(self._value))
        return Maybe.nothing()

    def bind(self, func: Callable[[T], "Maybe[R]"]) -> "Maybe[R]":
        if self.is_just():
            return func(self._value)
        return Maybe.nothing()

    def get_or_else(self, default: T) -> T:
        return self._value if self.is_just() else default


# 고급 데코레이터 패턴
def memoize(maxsize: int | None = None):
    """메모이제이션 데코레이터"""

    def decorator(func: Callable) -> Callable:
        cache: dict[tuple, Any] = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            if maxsize is None or len(cache) < maxsize:
                cache[key] = result
            return result

        return wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """레이트 리미팅 데코레이터"""
    calls_history: deque = deque()
    lock = Lock()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # 오래된 호출 제거
                while calls_history and calls_history[0] < now - period:
                    calls_history.popleft()

                if len(calls_history) >= calls:
                    raise RuntimeError("Rate limit exceeded")

                calls_history.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def timeout(seconds: float):
    """타임아웃 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 동기 함수는 스레드에서 실행
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=seconds)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """지수 백오프 재시도 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
            return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
            return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# 고급 컨텍스트 매니저
class Transaction:
    """트랜잭션 컨텍스트 매니저"""

    def __init__(self):
        self.operations: list[Callable] = []
        self.rollback_operations: list[Callable] = []
        self.committed = False

    def add_operation(self, operation: Callable, rollback: Callable):
        """작업 추가"""
        self.operations.append(operation)
        self.rollback_operations.append(rollback)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 롤백
            for rollback_op in reversed(self.rollback_operations):
                with suppress(Exception):
                    rollback_op()
            return False

        # 커밋
        for op in self.operations:
            op()
        self.committed = True
        return False


class ResourcePool:
    """리소스 풀 패턴"""

    def __init__(self, factory: Callable[[], T], max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque()
        self.in_use: set[T] = set()
        self.lock = Lock()

    def acquire(self) -> T:
        """리소스 획득"""
        with self.lock:
            if self.pool:
                resource = self.pool.popleft()
            elif len(self.in_use) < self.max_size:
                resource = self.factory()
            else:
                raise RuntimeError("Resource pool exhausted")

            self.in_use.add(resource)
            return resource

    def release(self, resource: T):
        """리소스 반환"""
        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                self.pool.append(resource)

    @contextmanager
    def resource(self):
        """컨텍스트 매니저로 리소스 사용"""
        res = self.acquire()
        try:
            yield res
        finally:
            self.release(res)


# 옵저버 패턴
class Observer(Protocol):
    """옵저버 프로토콜"""

    def update(self, event: str, data: Any) -> None: ...


class Observable:
    """관찰 가능한 객체"""

    def __init__(self):
        self._observers: list[Observer] = []
        self._lock = Lock()

    def attach(self, observer: Observer) -> None:
        """옵저버 추가"""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """옵저버 제거"""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def notify(self, event: str, data: Any) -> None:
        """모든 옵저버에 알림"""
        with self._lock:
            observers = self._observers.copy()

        for observer in observers:
            with suppress(Exception):
                observer.update(event, data)


# 전략 패턴
class Strategy(Protocol):
    """전략 프로토콜"""

    def execute(self, data: Any) -> Any: ...


class Context:
    """컨텍스트 클래스"""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def set_strategy(self, strategy: Strategy) -> None:
        """전략 변경"""
        self.strategy = strategy

    def execute(self, data: Any) -> Any:
        """전략 실행"""
        return self.strategy.execute(data)


class ConcreteStrategyA:
    """구체적인 전략 A"""

    def execute(self, data: Any) -> Any:
        return f"Strategy A: {data}"


class ConcreteStrategyB:
    """구체적인 전략 B"""

    def execute(self, data: Any) -> Any:
        return f"Strategy B: {data}"


# 팩토리 패턴
class Product(ABC):
    """제품 추상 클래스"""

    @abstractmethod
    def operation(self) -> str:
        pass


class ConcreteProductA(Product):
    """구체적인 제품 A"""

    def operation(self) -> str:
        return "Product A"


class ConcreteProductB(Product):
    """구체적인 제품 B"""

    def operation(self) -> str:
        return "Product B"


class ProductFactory:
    """제품 팩토리"""

    _products: dict[str, type] = {"A": ConcreteProductA, "B": ConcreteProductB}

    @classmethod
    def create(cls, product_type: str) -> Product:
        """제품 생성"""
        product_class = cls._products.get(product_type)
        if product_class:
            return product_class()
        raise ValueError(f"Unknown product type: {product_type}")

    @classmethod
    def register(cls, product_type: str, product_class: type):
        """제품 타입 등록"""
        cls._products[product_type] = product_class


# 빌더 패턴
@dataclass
class Query:
    """쿼리 클래스"""

    select: list[str] = field(default_factory=list)
    from_table: str | None = None
    where: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None


class QueryBuilder:
    """쿼리 빌더"""

    def __init__(self):
        self.query = Query()

    def select(self, *columns: str) -> "QueryBuilder":
        """SELECT 절 추가"""
        self.query.select.extend(columns)
        return self

    def from_table(self, table: str) -> "QueryBuilder":
        """FROM 절 추가"""
        self.query.from_table = table
        return self

    def where(self, condition: str) -> "QueryBuilder":
        """WHERE 절 추가"""
        self.query.where.append(condition)
        return self

    def order_by(self, *columns: str) -> "QueryBuilder":
        """ORDER BY 절 추가"""
        self.query.order_by.extend(columns)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """LIMIT 절 추가"""
        self.query.limit = n
        return self

    def build(self) -> Query:
        """쿼리 빌드"""
        return self.query


# 체인 오브 리스폰시빌리티 패턴
class Handler(ABC):
    """핸들러 추상 클래스"""

    def __init__(self):
        self._next: Handler | None = None

    def set_next(self, handler: "Handler") -> "Handler":
        """다음 핸들러 설정"""
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, request: Any) -> Any | None:
        """요청 처리"""
        pass

    def _handle_next(self, request: Any) -> Any | None:
        """다음 핸들러로 전달"""
        if self._next:
            return self._next.handle(request)
        return None


class ConcreteHandlerA(Handler):
    """구체적인 핸들러 A"""

    def handle(self, request: Any) -> Any | None:
        if isinstance(request, str) and request.startswith("A"):
            return f"Handler A processed: {request}"
        return self._handle_next(request)


class ConcreteHandlerB(Handler):
    """구체적인 핸들러 B"""

    def handle(self, request: Any) -> Any | None:
        if isinstance(request, str) and request.startswith("B"):
            return f"Handler B processed: {request}"
        return self._handle_next(request)


# 비동기 패턴
class AsyncQueue:
    """비동기 큐"""

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: Any) -> None:
        """아이템 추가"""
        await self._queue.put(item)

    async def get(self) -> Any:
        """아이템 가져오기"""
        return await self._queue.get()

    async def task_done(self) -> None:
        """작업 완료 표시"""
        self._queue.task_done()

    async def join(self) -> None:
        """모든 작업 완료 대기"""
        await self._queue.join()


class AsyncWorker:
    """비동기 워커"""

    def __init__(self, queue: AsyncQueue, worker_id: int):
        self.queue = queue
        self.worker_id = worker_id
        self.running = False

    async def start(self, handler: Callable[[Any], Any]):
        """워커 시작"""
        self.running = True
        while self.running:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                try:
                    await handler(item)
                finally:
                    await self.queue.task_done()
            except asyncio.TimeoutError:
                continue

    def stop(self):
        """워커 중지"""
        self.running = False


# 이터레이터 패턴
class Tree:
    """트리 구조"""

    def __init__(self, value: Any):
        self.value = value
        self.children: list[Tree] = []

    def add_child(self, child: "Tree") -> None:
        """자식 추가"""
        self.children.append(child)

    def __iter__(self) -> Iterator[Any]:
        """전위 순회"""
        yield self.value
        for child in self.children:
            yield from child

    def depth_first(self) -> Iterator[Any]:
        """깊이 우선 탐색"""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node.value
            stack.extend(reversed(node.children))

    def breadth_first(self) -> Iterator[Any]:
        """너비 우선 탐색"""
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node.value
            queue.extend(node.children)


# 어댑터 패턴
class OldInterface:
    """구 인터페이스"""

    def old_method(self, x: int, y: int) -> int:
        return x + y


class NewInterface(Protocol):
    """신 인터페이스"""

    def new_method(self, data: dict[str, int]) -> int: ...


class Adapter:
    """어댑터"""

    def __init__(self, old_interface: OldInterface):
        self.old_interface = old_interface

    def new_method(self, data: dict[str, int]) -> int:
        """신 인터페이스 구현"""
        return self.old_interface.old_method(data.get("x", 0), data.get("y", 0))


# 프록시 패턴
class Subject(ABC):
    """주제 추상 클래스"""

    @abstractmethod
    def request(self) -> str:
        pass


class RealSubject(Subject):
    """실제 주제"""

    def request(self) -> str:
        return "RealSubject: Handling request"


class Proxy(Subject):
    """프록시"""

    def __init__(self, real_subject: RealSubject):
        self._real_subject = real_subject
        self._cache: str | None = None

    def request(self) -> str:
        """요청 처리 (캐싱 포함)"""
        if self._cache is None:
            self._cache = self._real_subject.request()
        return f"Proxy: {self._cache}"


# 데코레이터 패턴 (구조적)
class Component(ABC):
    """컴포넌트 추상 클래스"""

    @abstractmethod
    def operation(self) -> str:
        pass


class ConcreteComponent(Component):
    """구체적인 컴포넌트"""

    def operation(self) -> str:
        return "ConcreteComponent"


class Decorator(Component):
    """데코레이터"""

    def __init__(self, component: Component):
        self._component = component

    def operation(self) -> str:
        return self._component.operation()


class ConcreteDecoratorA(Decorator):
    """구체적인 데코레이터 A"""

    def operation(self) -> str:
        return f"ConcreteDecoratorA({super().operation()})"


class ConcreteDecoratorB(Decorator):
    """구체적인 데코레이터 B"""

    def operation(self) -> str:
        return f"ConcreteDecoratorB({super().operation()})"


# 상태 패턴
class State(ABC):
    """상태 추상 클래스"""

    @abstractmethod
    def handle(self, context: "Context") -> None:
        pass


class ConcreteStateA(State):
    """구체적인 상태 A"""

    def handle(self, context: "Context") -> None:
        print("State A handling")
        context.state = ConcreteStateB()


class ConcreteStateB(State):
    """구체적인 상태 B"""

    def handle(self, context: "Context") -> None:
        print("State B handling")
        context.state = ConcreteStateA()


class StateContext:
    """상태 컨텍스트"""

    def __init__(self, state: State):
        self.state = state

    def request(self) -> None:
        """요청 처리"""
        self.state.handle(self)


# 비지터 패턴
class Visitor(ABC):
    """비지터 추상 클래스"""

    @abstractmethod
    def visit_element_a(self, element: "ElementA") -> None:
        pass

    @abstractmethod
    def visit_element_b(self, element: "ElementB") -> None:
        pass


class Element(ABC):
    """엘리먼트 추상 클래스"""

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass


class ElementA(Element):
    """엘리먼트 A"""

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_element_a(self)

    def operation_a(self) -> str:
        return "ElementA"


class ElementB(Element):
    """엘리먼트 B"""

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_element_b(self)

    def operation_b(self) -> str:
        return "ElementB"


class ConcreteVisitor(Visitor):
    """구체적인 비지터"""

    def visit_element_a(self, element: ElementA) -> None:
        print(f"Visiting {element.operation_a()}")

    def visit_element_b(self, element: ElementB) -> None:
        print(f"Visiting {element.operation_b()}")


# 템플릿 메서드 패턴
class AbstractClass(ABC):
    """추상 클래스"""

    def template_method(self) -> str:
        """템플릿 메서드"""
        return f"{self.primitive_operation1()}-{self.primitive_operation2()}"

    @abstractmethod
    def primitive_operation1(self) -> str:
        pass

    @abstractmethod
    def primitive_operation2(self) -> str:
        pass


class ConcreteClass(AbstractClass):
    """구체적인 클래스"""

    def primitive_operation1(self) -> str:
        return "Operation1"

    def primitive_operation2(self) -> str:
        return "Operation2"


# 고급 함수형 패턴
def compose(*functions: Callable) -> Callable:
    """함수 합성"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


def pipe(value: Any, *functions: Callable) -> Any:
    """파이프 연산"""
    return reduce(lambda acc, f: f(acc), functions, value)


def curry(func: Callable) -> Callable:
    """커링"""

    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *args2, **kwargs2: curried(*(args + args2), **{**kwargs, **kwargs2})

    return curried


def partial_application(func: Callable, *args, **kwargs) -> Callable:
    """부분 적용"""
    return partial(func, *args, **kwargs)


# 고급 타입 힌트
PositiveInt = Annotated[int, "Must be positive"]
NonEmptyString = Annotated[str, "Must not be empty"]


def process_positive(value: PositiveInt) -> PositiveInt:
    """양수 처리"""
    if value <= 0:
        raise ValueError("Value must be positive")
    return value


# WeakReference 패턴
class WeakRefContainer:
    """약한 참조 컨테이너"""

    def __init__(self):
        self._refs: list[weakref.ref] = []

    def add(self, obj: Any) -> None:
        """객체 추가"""
        self._refs.append(weakref.ref(obj))

    def get_alive(self) -> list[Any]:
        """살아있는 객체 반환"""
        alive = []
        for ref in self._refs:
            obj = ref()
            if obj is not None:
                alive.append(obj)
        return alive


# 레지스트리 패턴
class Registry:
    """레지스트리 패턴"""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """클래스 등록 데코레이터"""

        def decorator(klass: type):
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def get(cls, name: str) -> type | None:
        """클래스 가져오기"""
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        """모든 등록된 이름 반환"""
        return list(cls._registry.keys())


@Registry.register("type_a")
class RegisteredTypeA:
    pass


@Registry.register("type_b")
class RegisteredTypeB:
    pass
