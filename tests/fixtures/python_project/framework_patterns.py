"""프레임워크 스타일 패턴 및 실전 케이스"""

import asyncio
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, Optional


# 미들웨어 패턴
class Middleware:
    """미들웨어 기본 클래스"""

    def __init__(self, next_middleware: Optional['Middleware'] = None):
        self.next = next_middleware

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """요청 처리"""
        # 전처리
        request = self.process_request(request)

        # 다음 미들웨어로 전달
        response = self.next.handle(request) if self.next else {}

        # 후처리
        response = self.process_response(response)
        return response

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """요청 전처리"""
        return request

    def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """응답 후처리"""
        return response


class AuthMiddleware(Middleware):
    """인증 미들웨어"""

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        token = request.get("token")
        if not token:
            raise ValueError("Authentication required")
        request["user"] = self.authenticate(token)
        return request

    def authenticate(self, token: str) -> dict[str, Any]:
        """토큰 인증"""
        return {"id": "user123", "name": "Test User"}


class LoggingMiddleware(Middleware):
    """로깅 미들웨어"""

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        print(f"[REQUEST] {datetime.now()}: {request.get('path', 'unknown')}")
        return request

    def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        print(f"[RESPONSE] {datetime.now()}: {response.get('status', 'unknown')}")
        return response


class ValidationMiddleware(Middleware):
    """검증 미들웨어"""

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if "data" not in request:
            raise ValueError("Data is required")
        return request


# 라우터 패턴
class Route:
    """라우트 클래스"""

    def __init__(
        self,
        path: str,
        handler: Callable,
        methods: list[str] = None,
        middleware: list[Middleware] = None
    ):
        self.path = path
        self.handler = handler
        self.methods = methods or ["GET"]
        self.middleware = middleware or []
        self.pattern = self._compile_pattern(path)

    def _compile_pattern(self, path: str) -> re.Pattern:
        """경로 패턴 컴파일"""
        pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', path)
        return re.compile(f'^{pattern}$')

    def matches(self, path: str) -> dict[str, str] | None:
        """경로 매칭"""
        match = self.pattern.match(path)
        if match:
            return match.groupdict()
        return None


class Router:
    """라우터"""

    def __init__(self):
        self.routes: list[Route] = []

    def add_route(
        self,
        path: str,
        handler: Callable,
        methods: list[str] = None,
        middleware: list[Middleware] = None
    ):
        """라우트 추가"""
        route = Route(path, handler, methods, middleware)
        self.routes.append(route)

    def get(self, path: str, handler: Callable, middleware: list[Middleware] = None):
        """GET 라우트 추가"""
        self.add_route(path, handler, ["GET"], middleware)

    def post(self, path: str, handler: Callable, middleware: list[Middleware] = None):
        """POST 라우트 추가"""
        self.add_route(path, handler, ["POST"], middleware)

    def find_route(self, path: str, method: str) -> tuple[Route, dict[str, str]] | None:
        """라우트 찾기"""
        for route in self.routes:
            if method in route.methods:
                params = route.matches(path)
                if params is not None:
                    return route, params
        return None


# 의존성 주입 컨테이너
class ServiceContainer:
    """서비스 컨테이너"""

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Callable] = {}
        self._singletons: dict[str, Any] = {}

    def register(self, name: str, service: Any, singleton: bool = False):
        """서비스 등록"""
        if singleton:
            self._singletons[name] = service
        else:
            self._services[name] = service

    def register_factory(self, name: str, factory: Callable, singleton: bool = False):
        """팩토리 등록"""
        if singleton:
            self._factories[name] = lambda: self._get_singleton(name, factory)
        else:
            self._factories[name] = factory

    def _get_singleton(self, name: str, factory: Callable) -> Any:
        """싱글톤 가져오기"""
        if name not in self._singletons:
            self._singletons[name] = factory()
        return self._singletons[name]

    def get(self, name: str) -> Any:
        """서비스 가져오기"""
        if name in self._services:
            return self._services[name]
        if name in self._factories:
            return self._factories[name]()
        if name in self._singletons:
            return self._singletons[name]
        raise ValueError(f"Service {name} not found")

    def inject(self, func: Callable) -> Callable:
        """의존성 주입 데코레이터"""
        sig = inspect.signature(func)
        params = sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 매개변수 이름으로 서비스 주입
            for param_name, _param in params.items():
                if param_name not in kwargs and param_name in self._services:
                    kwargs[param_name] = self.get(param_name)
            return func(*args, **kwargs)

        return wrapper


# 이벤트 버스 패턴
class EventBus:
    """이벤트 버스"""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._async_handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable, async_handler: bool = False):
        """이벤트 구독"""
        if async_handler:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = []
            self._async_handlers[event_type].append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """이벤트 구독 해제"""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
        if event_type in self._async_handlers:
            if handler in self._async_handlers[event_type]:
                self._async_handlers[event_type].remove(handler)

    def publish(self, event_type: str, data: Any):
        """이벤트 발행"""
        # 동기 핸들러
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(data)

        # 비동기 핸들러
        if event_type in self._async_handlers:
            for handler in self._async_handlers[event_type]:
                asyncio.create_task(handler(data))


# 커맨드 패턴
class Command(ABC):
    """커맨드 추상 클래스"""

    @abstractmethod
    def execute(self) -> Any:
        pass

    @abstractmethod
    def undo(self) -> Any:
        pass


class CommandInvoker:
    """커맨드 실행자"""

    def __init__(self):
        self.history: list[Command] = []

    def execute(self, command: Command) -> Any:
        """커맨드 실행"""
        result = command.execute()
        self.history.append(command)
        return result

    def undo(self) -> Any:
        """마지막 커맨드 실행 취소"""
        if self.history:
            command = self.history.pop()
            return command.undo()
        return None


class AddCommand(Command):
    """더하기 커맨드"""

    def __init__(self, target: list[int], value: int):
        self.target = target
        self.value = value

    def execute(self) -> Any:
        self.target.append(self.value)
        return len(self.target)

    def undo(self) -> Any:
        if self.target and self.target[-1] == self.value:
            self.target.pop()
        return len(self.target)


# 메멘토 패턴
class Memento:
    """메멘토"""

    def __init__(self, state: dict[str, Any]):
        self._state = state.copy()

    def get_state(self) -> dict[str, Any]:
        """상태 가져오기"""
        return self._state.copy()


class Originator:
    """원본 객체"""

    def __init__(self):
        self._state: dict[str, Any] = {}

    def set_state(self, state: dict[str, Any]):
        """상태 설정"""
        self._state = state

    def save(self) -> Memento:
        """메멘토 생성"""
        return Memento(self._state)

    def restore(self, memento: Memento):
        """메멘토로부터 복원"""
        self._state = memento.get_state()

    def get_state(self) -> dict[str, Any]:
        """현재 상태 가져오기"""
        return self._state.copy()


class Caretaker:
    """관리자"""

    def __init__(self, originator: Originator):
        self.originator = originator
        self.history: list[Memento] = []

    def save(self):
        """상태 저장"""
        self.history.append(self.originator.save())

    def restore(self, index: int = -1):
        """상태 복원"""
        if 0 <= abs(index) <= len(self.history):
            self.originator.restore(self.history[index])


# 플라이웨이트 패턴
class Flyweight:
    """플라이웨이트"""

    def __init__(self, intrinsic_state: str):
        self.intrinsic_state = intrinsic_state

    def operation(self, extrinsic_state: str) -> str:
        """작업 수행"""
        return f"{self.intrinsic_state}-{extrinsic_state}"


class FlyweightFactory:
    """플라이웨이트 팩토리"""

    def __init__(self):
        self._flyweights: dict[str, Flyweight] = {}

    def get_flyweight(self, key: str) -> Flyweight:
        """플라이웨이트 가져오기"""
        if key not in self._flyweights:
            self._flyweights[key] = Flyweight(key)
        return self._flyweights[key]

    def count(self) -> int:
        """플라이웨이트 개수"""
        return len(self._flyweights)


# 인터프리터 패턴
class Expression(ABC):
    """표현식 추상 클래스"""

    @abstractmethod
    def interpret(self, context: dict[str, Any]) -> Any:
        pass


class TerminalExpression(Expression):
    """터미널 표현식"""

    def __init__(self, value: str):
        self.value = value

    def interpret(self, context: dict[str, Any]) -> Any:
        return context.get(self.value, self.value)


class NonTerminalExpression(Expression):
    """논터미널 표현식"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def interpret(self, context: dict[str, Any]) -> Any:
        left_value = self.left.interpret(context)
        right_value = self.right.interpret(context)
        return self.operate(left_value, right_value)

    @abstractmethod
    def operate(self, left: Any, right: Any) -> Any:
        pass


class AddExpression(NonTerminalExpression):
    """더하기 표현식"""

    def operate(self, left: Any, right: Any) -> Any:
        return left + right


class MultiplyExpression(NonTerminalExpression):
    """곱하기 표현식"""

    def operate(self, left: Any, right: Any) -> Any:
        return left * right


# 레포지토리 패턴
class Repository(ABC):
    """레포지토리 추상 클래스"""

    @abstractmethod
    def find_by_id(self, id: str) -> Any | None:
        pass

    @abstractmethod
    def find_all(self) -> list[Any]:
        pass

    @abstractmethod
    def save(self, entity: Any) -> Any:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass


class InMemoryRepository(Repository):
    """메모리 레포지토리"""

    def __init__(self):
        self._entities: dict[str, Any] = {}

    def find_by_id(self, id: str) -> Any | None:
        return self._entities.get(id)

    def find_all(self) -> list[Any]:
        return list(self._entities.values())

    def save(self, entity: Any) -> Any:
        entity_id = getattr(entity, "id", str(id(entity)))
        self._entities[entity_id] = entity
        return entity

    def delete(self, id: str) -> bool:
        if id in self._entities:
            del self._entities[id]
            return True
        return False


# 유닛 오브 워크 패턴
class UnitOfWork:
    """유닛 오브 워크"""

    def __init__(self):
        self.new_entities: list[Any] = []
        self.modified_entities: list[Any] = []
        self.deleted_entities: list[Any] = []

    def register_new(self, entity: Any):
        """새 엔티티 등록"""
        self.new_entities.append(entity)

    def register_modified(self, entity: Any):
        """수정된 엔티티 등록"""
        if entity not in self.modified_entities:
            self.modified_entities.append(entity)

    def register_deleted(self, entity: Any):
        """삭제된 엔티티 등록"""
        self.deleted_entities.append(entity)

    def commit(self):
        """커밋"""
        # 실제로는 데이터베이스에 저장
        for entity in self.new_entities:
            print(f"Inserting {entity}")
        for entity in self.modified_entities:
            print(f"Updating {entity}")
        for entity in self.deleted_entities:
            print(f"Deleting {entity}")

        self.new_entities.clear()
        self.modified_entities.clear()
        self.deleted_entities.clear()

    def rollback(self):
        """롤백"""
        self.new_entities.clear()
        self.modified_entities.clear()
        self.deleted_entities.clear()


# 스펙ification 패턴
class Specification(ABC):
    """스펙 추상 클래스"""

    @abstractmethod
    def is_satisfied_by(self, candidate: Any) -> bool:
        pass

    def and_spec(self, other: 'Specification') -> 'Specification':
        """AND 연산"""
        return AndSpecification(self, other)

    def or_spec(self, other: 'Specification') -> 'Specification':
        """OR 연산"""
        return OrSpecification(self, other)

    def not_spec(self) -> 'Specification':
        """NOT 연산"""
        return NotSpecification(self)


class AndSpecification(Specification):
    """AND 스펙"""

    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.spec1.is_satisfied_by(candidate) and self.spec2.is_satisfied_by(candidate)


class OrSpecification(Specification):
    """OR 스펙"""

    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.spec1.is_satisfied_by(candidate) or self.spec2.is_satisfied_by(candidate)


class NotSpecification(Specification):
    """NOT 스펙"""

    def __init__(self, spec: Specification):
        self.spec = spec

    def is_satisfied_by(self, candidate: Any) -> bool:
        return not self.spec.is_satisfied_by(candidate)


class AgeSpecification(Specification):
    """나이 스펙"""

    def __init__(self, min_age: int):
        self.min_age = min_age

    def is_satisfied_by(self, candidate: Any) -> bool:
        return getattr(candidate, "age", 0) >= self.min_age


# 파이프라인 패턴
class Pipeline:
    """파이프라인"""

    def __init__(self):
        self.stages: list[Callable] = []

    def add_stage(self, stage: Callable) -> 'Pipeline':
        """스테이지 추가"""
        self.stages.append(stage)
        return self

    def execute(self, data: Any) -> Any:
        """파이프라인 실행"""
        result = data
        for stage in self.stages:
            result = stage(result)
        return result


# 체이닝 빌더 패턴
class QueryBuilder:
    """쿼리 빌더 (체이닝)"""

    def __init__(self):
        self._select: list[str] = []
        self._from_table: str | None = None
        self._where: list[str] = []
        self._order_by: list[str] = []
        self._limit: int | None = None

    def select(self, *columns: str) -> 'QueryBuilder':
        self._select.extend(columns)
        return self

    def from_table(self, table: str) -> 'QueryBuilder':
        self._from_table = table
        return self

    def where(self, condition: str) -> 'QueryBuilder':
        self._where.append(condition)
        return self

    def order_by(self, *columns: str) -> 'QueryBuilder':
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        self._limit = n
        return self

    def build(self) -> str:
        """SQL 쿼리 빌드"""
        parts = []
        if self._select:
            parts.append(f"SELECT {', '.join(self._select)}")
        if self._from_table:
            parts.append(f"FROM {self._from_table}")
        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")
        if self._limit:
            parts.append(f"LIMIT {self._limit}")
        return " ".join(parts)

