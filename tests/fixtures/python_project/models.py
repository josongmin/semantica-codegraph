"""데이터 모델 정의"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class UserRole(str, Enum):
    """사용자 역할 열거형"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


@dataclass
class Address:
    """주소 데이터 클래스"""
    street: str
    city: str
    zip_code: str
    country: str = "KR"


class User:
    """사용자 모델 클래스"""

    def __init__(self, name: str, age: int, email: str | None = None):
        self.name = name
        self.age = age
        self.email = email
        self.role: UserRole = UserRole.USER
        self.created_at: datetime = datetime.now()
        self._address: Address | None = None

    @property
    def address(self) -> Address | None:
        """주소 프로퍼티"""
        return self._address

    @address.setter
    def address(self, value: Address):
        """주소 설정자"""
        self._address = value

    def greet(self) -> str:
        """인사 메시지 반환"""
        return f"Hello, {self.name}!"

    @staticmethod
    def create_default() -> "User":
        """기본 사용자 생성"""
        return User("Anonymous", 0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """딕셔너리로부터 사용자 생성"""
        user = cls(
            name=data.get("name", ""),
            age=data.get("age", 0),
            email=data.get("email")
        )
        if "role" in data:
            user.role = UserRole(data["role"])
        return user

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "age": self.age,
            "email": self.email,
            "role": self.role.value
        }


class Admin(User):
    """관리자 사용자 클래스"""

    def __init__(self, name: str, age: int, permissions: list[str], email: str | None = None):
        super().__init__(name, age, email)
        self.role = UserRole.ADMIN
        self.permissions: list[str] = permissions

    async def check_permission(self, permission: str) -> bool:
        """권한 확인"""
        return permission in self.permissions

    def grant_permission(self, permission: str) -> None:
        """권한 부여"""
        if permission not in self.permissions:
            self.permissions.append(permission)

    def revoke_permission(self, permission: str) -> bool:
        """권한 제거"""
        if permission in self.permissions:
            self.permissions.remove(permission)
            return True
        return False


class Product:
    """상품 모델 클래스"""

    def __init__(self, id: str, name: str, price: float, stock: int = 0):
        self.id = id
        self.name = name
        self.price = price
        self.stock = stock

    def __repr__(self) -> str:
        return f"Product(id={self.id}, name={self.name}, price={self.price})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Product):
            return False
        return self.id == other.id

    def apply_discount(self, percentage: float) -> None:
        """할인 적용"""
        if 0 <= percentage <= 100:
            self.price *= (1 - percentage / 100)

    def is_available(self) -> bool:
        """재고 확인"""
        return self.stock > 0


class Order:
    """주문 모델 클래스"""

    def __init__(self, order_id: str, user: User, products: list[Product]):
        self.order_id = order_id
        self.user = user
        self.products = products
        self.status: str = "pending"
        self.created_at = datetime.now()

    def calculate_total(self) -> float:
        """총액 계산"""
        return sum(product.price for product in self.products)

    def add_product(self, product: Product) -> None:
        """상품 추가"""
        self.products.append(product)

    def remove_product(self, product_id: str) -> bool:
        """상품 제거"""
        for i, product in enumerate(self.products):
            if product.id == product_id:
                self.products.pop(i)
                return True
        return False

