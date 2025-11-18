"""서비스 레이어"""

from abc import ABC, abstractmethod
from typing import Protocol

from .models import Admin, Order, Product, User


class UserRepository(Protocol):
    """사용자 저장소 프로토콜"""

    def save(self, user: User) -> None:
        """사용자 저장"""
        ...

    def find_by_id(self, user_id: str) -> User | None:
        """ID로 사용자 찾기"""
        ...

    def find_all(self) -> list[User]:
        """모든 사용자 찾기"""
        ...


class InMemoryUserRepository:
    """메모리 기반 사용자 저장소"""

    def __init__(self):
        self._users: dict[str, User] = {}

    def save(self, user: User) -> None:
        """사용자 저장"""
        self._users[user.name] = user

    def find_by_id(self, user_id: str) -> User | None:
        """ID로 사용자 찾기"""
        return self._users.get(user_id)

    def find_all(self) -> list[User]:
        """모든 사용자 찾기"""
        return list(self._users.values())


class ProductService(ABC):
    """상품 서비스 추상 클래스"""

    @abstractmethod
    def get_product(self, product_id: str) -> Product | None:
        """상품 조회"""
        pass

    @abstractmethod
    def list_products(self) -> list[Product]:
        """상품 목록 조회"""
        pass


class UserService:
    """사용자 서비스 클래스"""

    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, name: str, age: int, email: str | None = None) -> User:
        """사용자 생성"""
        user = User(name, age, email)
        self.repository.save(user)
        return user

    def get_user(self, user_id: str) -> User | None:
        """사용자 조회"""
        return self.repository.find_by_id(user_id)

    def list_users(self) -> list[User]:
        """사용자 목록 조회"""
        return self.repository.find_all()

    def update_user_email(self, user_id: str, email: str) -> bool:
        """사용자 이메일 업데이트"""
        user = self.repository.find_by_id(user_id)
        if user:
            user.email = email
            self.repository.save(user)
            return True
        return False


class OrderService:
    """주문 서비스 클래스"""

    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self._orders: dict[str, Order] = {}

    def create_order(self, order_id: str, user_id: str, products: list[Product]) -> Order | None:
        """주문 생성"""
        user = self.user_service.get_user(user_id)
        if not user:
            return None

        order = Order(order_id, user, products)
        self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> Order | None:
        """주문 조회"""
        return self._orders.get(order_id)

    def calculate_order_total(self, order_id: str) -> float | None:
        """주문 총액 계산"""
        order = self.get_order(order_id)
        return order.calculate_total() if order else None


class AuthService:
    """인증 서비스 클래스"""

    def __init__(self, user_service: UserService):
        self.user_service = user_service

    def authenticate(self, username: str, password: str) -> User | None:
        """인증 처리"""
        user = self.user_service.get_user(username)
        # 실제로는 비밀번호 검증 로직이 들어감
        return user

    def authorize(self, user: User, permission: str) -> bool:
        """권한 확인"""
        if isinstance(user, Admin):
            return permission in user.permissions
        return False


def validate_email(email: str) -> bool:
    """이메일 유효성 검사"""
    return "@" in email and "." in email.split("@")[1]

