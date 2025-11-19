"""이상한 패턴과 엣지 케이스"""

import inspect
from collections.abc import Callable
from functools import partial, wraps
from typing import Any

# 전역 변수 남용
_global_state: dict[str, Any] = {}
_module_cache: dict[str, Any] = {}


# 동적 모듈 생성
def create_dynamic_module(name: str) -> type:
    """동적으로 클래스를 생성하는 함수"""

    class DynamicModule:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    DynamicModule.__name__ = name
    return DynamicModule


# 메서드를 동적으로 추가
def add_method_to_class(cls: type, method_name: str, method: Callable):
    """클래스에 메서드를 동적으로 추가"""
    setattr(cls, method_name, method)


# 함수를 클래스로 변환
class FunctionAsClass:
    """함수를 클래스처럼 사용"""

    def __init__(self, func: Callable):
        self.func = func
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<FunctionAsClass: {self.__name__}>"


# 클래스를 함수처럼 사용
class CallableClass:
    """호출 가능한 클래스"""

    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier

    def __call__(self, value: int) -> int:
        return value * self.multiplier

    def __add__(self, other: "CallableClass") -> "CallableClass":
        return CallableClass(self.multiplier + other.multiplier)

    def __mul__(self, other: int) -> "CallableClass":
        return CallableClass(self.multiplier * other)


# 속성 접근을 함수 호출로
class FunctionAttribute:
    """속성 접근이 함수 호출인 클래스"""

    def __getattribute__(self, name: str):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return lambda *args, **kwargs: f"Called {name} with {args}, {kwargs}"


# 무한 재귀 가능한 클래스
class RecursiveClass:
    """자기 자신을 참조하는 클래스"""

    def __init__(self, value: Any = None):
        self.value = value
        self.self_ref: RecursiveClass | None = self

    def set_ref(self, ref: "RecursiveClass"):
        self.self_ref = ref

    def get_chain_length(self) -> int:
        """참조 체인 길이 계산"""
        if self.self_ref is None or self.self_ref is self:
            return 1
        return 1 + self.self_ref.get_chain_length()


# 순환 참조
class NodeA:
    """순환 참조를 가진 클래스 A"""

    def __init__(self):
        self.ref_b: NodeB | None = None


class NodeB:
    """순환 참조를 가진 클래스 B"""

    def __init__(self):
        self.ref_a: NodeA | None = None


# 메타프로그래밍
class MetaBuilder(type):
    """빌더 패턴을 자동으로 생성하는 메타클래스"""

    def __new__(mcs, name, bases, namespace):
        # 빌더 메서드 자동 생성
        for key, value in namespace.items():
            if not key.startswith("_") and not callable(value):

                def make_setter(k):
                    def setter(self, val):
                        setattr(self, f"_{k}", val)
                        return self

                    setter.__name__ = f"set_{k}"
                    return setter

                namespace[f"set_{key}"] = make_setter(key)

        return super().__new__(mcs, name, bases, namespace)


class BuiltClass(metaclass=MetaBuilder):
    """메타클래스로 빌더가 자동 생성되는 클래스"""

    name: str = ""
    age: int = 0


# 데코레이터 체이닝의 극단적 케이스
def decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator 1")
        return func(*args, **kwargs)

    return wrapper


def decorator2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator 2")
        return func(*args, **kwargs)

    return wrapper


def decorator3(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator 3")
        return func(*args, **kwargs)

    return wrapper


@decorator1
@decorator2
@decorator3
def heavily_decorated_function(x: int) -> int:
    """많은 데코레이터가 적용된 함수"""
    return x * 2


# 람다 체이닝
lambda_chain = (lambda x: x + 1, lambda x: x * 2, lambda x: x**2, lambda x: str(x))


def apply_lambda_chain(value: int) -> str:
    """람다 체인 적용"""
    result = value
    for func in lambda_chain:
        result = func(result)
    return result


# 클로저 남용
def create_counter(start: int = 0):
    """클로저를 사용한 카운터"""
    count = start

    def increment():
        nonlocal count
        count += 1
        return count

    def decrement():
        nonlocal count
        count -= 1
        return count

    def get():
        return count

    def reset():
        nonlocal count
        count = start

    return {"increment": increment, "decrement": decrement, "get": get, "reset": reset}


# 부분 적용 남용
def multiply(x: int, y: int, z: int) -> int:
    """부분 적용 예제"""
    return x * y * z


multiply_by_2 = partial(multiply, 2)
multiply_by_2_and_3 = partial(multiply_by_2, 3)


# 연산자 오버로딩 남용
class NumberLike:
    """숫자처럼 동작하는 클래스"""

    def __init__(self, value: int):
        self.value = value

    def __add__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value + other.value)
        return NumberLike(self.value + other)

    def __sub__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value - other.value)
        return NumberLike(self.value - other)

    def __mul__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value * other.value)
        return NumberLike(self.value * other)

    def __truediv__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value // other.value)
        return NumberLike(self.value // other)

    def __pow__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value**other.value)
        return NumberLike(self.value**other)

    def __mod__(self, other):
        if isinstance(other, NumberLike):
            return NumberLike(self.value % other.value)
        return NumberLike(self.value % other)

    def __eq__(self, other):
        if isinstance(other, NumberLike):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, NumberLike):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, NumberLike):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, NumberLike):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, NumberLike):
            return self.value >= other.value
        return self.value >= other

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"NumberLike({self.value})"


# 특수 메서드 남용
class EverythingClass:
    """모든 특수 메서드를 구현한 클래스"""

    def __init__(self, value: Any):
        self.value = value

    def __getattr__(self, name: str):
        return f"<attr:{name}>"

    def __setattr__(self, name: str, value: Any):
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str):
        if hasattr(self, name):
            object.__delattr__(self, name)

    def __getitem__(self, key: Any):
        return f"<item:{key}>"

    def __setitem__(self, key: Any, value: Any):
        object.__setattr__(self, f"_item_{key}", value)

    def __delitem__(self, key: Any):
        attr_name = f"_item_{key}"
        if hasattr(self, attr_name):
            object.__delattr__(self, attr_name)

    def __len__(self):
        return len(str(self.value))

    def __contains__(self, item: Any):
        return str(item) in str(self.value)

    def __iter__(self):
        return iter(str(self.value))

    def __reversed__(self):
        return reversed(str(self.value))

    def __call__(self, *args, **kwargs):
        return f"Called with {args}, {kwargs}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __bool__(self):
        return bool(self.value)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"EverythingClass({self.value})"

    def __repr__(self):
        return f"EverythingClass(value={repr(self.value)})"

    def __format__(self, format_spec: str):
        return format(self.value, format_spec)

    def __bytes__(self):
        return bytes(str(self.value), encoding="utf-8")

    def __dir__(self):
        return list(super().__dir__()) + ["custom_attr"]


# 동적 타입 생성
def create_type(name: str, bases: tuple = (), attrs: dict = None):
    """동적으로 타입 생성"""
    if attrs is None:
        attrs = {}
    return type(name, bases, attrs)


DynamicType = create_type("DynamicType", (), {"value": 42, "get_value": lambda self: self.value})


# 함수 시그니처 조작
def inspect_and_modify(func: Callable) -> Callable:
    """함수 시그니처를 조사하고 수정"""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    def wrapper(*args, **kwargs):
        print(f"Function: {func.__name__}")
        print(f"Parameters: {params}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        return func(*args, **kwargs)

    wrapper.__signature__ = sig
    wrapper.__annotations__ = func.__annotations__
    return wrapper


@inspect_and_modify
def annotated_function(x: int, y: str = "default") -> str:
    """타입 힌트가 있는 함수"""
    return f"{x}: {y}"


# 모듈 레벨 코드 실행
_module_init_code = """
# 모듈 레벨에서 실행되는 코드
print("Module is being imported!")
"""


# exec를 사용한 동적 코드 실행
def execute_dynamic_code(code: str, globals_dict: dict = None, locals_dict: dict = None):
    """동적으로 코드 실행"""
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}
    exec(code, globals_dict, locals_dict)
    return locals_dict


# 제너레이터 체이닝
def generator1(n: int):
    """첫 번째 제너레이터"""
    for i in range(n):
        yield i * 2


def generator2(gen):
    """두 번째 제너레이터"""
    for value in gen:
        yield value + 1


def generator3(gen):
    """세 번째 제너레이터"""
    for value in gen:
        yield value**2


def chained_generators(n: int):
    """체이닝된 제너레이터"""
    gen1 = generator1(n)
    gen2 = generator2(gen1)
    gen3 = generator3(gen2)
    return gen3


# 컨텍스트 매니저 체이닝
class Context1:
    def __enter__(self):
        print("Enter Context1")
        return self

    def __exit__(self, *args):
        print("Exit Context1")
        return False


class Context2:
    def __enter__(self):
        print("Enter Context2")
        return self

    def __exit__(self, *args):
        print("Exit Context2")
        return False


class Context3:
    def __enter__(self):
        print("Enter Context3")
        return self

    def __exit__(self, *args):
        print("Exit Context3")
        return False


def nested_contexts():
    """중첩된 컨텍스트 매니저"""
    with Context1(), Context2(), Context3():
        pass


# 예외 체이닝
def exception_chain():
    """예외 체이닝 예제"""
    try:
        try:
            raise ValueError("Inner error")
        except ValueError as e:
            raise RuntimeError("Outer error") from e
    except RuntimeError as e:
        raise TypeError("Final error") from e


# 얕은/깊은 복사 패턴
import copy


class CopyableClass:
    """복사 가능한 클래스"""

    def __init__(self, data: dict):
        self.data = data

    def __copy__(self):
        """얕은 복사"""
        return CopyableClass(copy.copy(self.data))

    def __deepcopy__(self, memo):
        """깊은 복사"""
        return CopyableClass(copy.deepcopy(self.data, memo))


# 프로퍼티 체이닝
class PropertyChain:
    """프로퍼티가 체이닝되는 클래스"""

    def __init__(self, value: int = 0):
        self._value = value

    @property
    def value(self) -> "PropertyChain":
        return self

    @value.setter
    def value(self, val: int):
        self._value = val

    def add(self, n: int) -> "PropertyChain":
        self._value += n
        return self

    def multiply(self, n: int) -> "PropertyChain":
        self._value *= n
        return self

    def get(self) -> int:
        return self._value


# 데코레이터 팩토리의 팩토리
def decorator_factory_factory(base_name: str):
    """데코레이터 팩토리를 생성하는 팩토리"""

    def decorator_factory(prefix: str = ""):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"{prefix}{base_name}: {func.__name__}")
                return func(*args, **kwargs)

            return wrapper

        return decorator

    return decorator_factory


my_decorator_factory = decorator_factory_factory("LOG")
my_decorator = my_decorator_factory("[INFO] ")


@my_decorator
def decorated_function():
    return "result"
