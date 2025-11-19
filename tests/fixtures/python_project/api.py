"""API 관련 코드"""

from functools import wraps
from typing import Any

from .models import Product, User
from .services import AuthService, OrderService, UserService


class APIError(Exception):
    """API 에러 클래스"""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def require_auth(func):
    """인증 필요 데코레이터"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "current_user") or self.current_user is None:
            raise APIError("인증이 필요합니다", 401)
        return func(self, *args, **kwargs)

    return wrapper


def require_admin(func):
    """관리자 권한 필요 데코레이터"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "current_user") or not isinstance(self.current_user, User):
            raise APIError("관리자 권한이 필요합니다", 403)
        return func(self, *args, **kwargs)

    return wrapper


class UserAPI:
    """사용자 API 클래스"""

    def __init__(self, user_service: UserService, auth_service: AuthService):
        self.user_service = user_service
        self.auth_service = auth_service
        self.current_user: User | None = None

    def login(self, username: str, password: str) -> dict[str, Any]:
        """로그인"""
        user = self.auth_service.authenticate(username, password)
        if user:
            self.current_user = user
            return {"success": True, "user": user.to_dict()}
        raise APIError("로그인 실패", 401)

    @require_auth
    def get_profile(self) -> dict[str, Any]:
        """프로필 조회"""
        return {"user": self.current_user.to_dict()}

    def create_user(self, name: str, age: int, email: str | None = None) -> dict[str, Any]:
        """사용자 생성"""
        user = self.user_service.create_user(name, age, email)
        return {"success": True, "user": user.to_dict()}

    @require_auth
    def update_email(self, email: str) -> dict[str, Any]:
        """이메일 업데이트"""
        if not self.current_user:
            raise APIError("사용자를 찾을 수 없습니다", 404)

        success = self.user_service.update_user_email(self.current_user.name, email)
        if success:
            return {"success": True, "message": "이메일이 업데이트되었습니다"}
        return {"success": False, "message": "업데이트 실패"}


class OrderAPI:
    """주문 API 클래스"""

    def __init__(self, order_service: OrderService, auth_service: AuthService):
        self.order_service = order_service
        self.auth_service = auth_service
        self.current_user: User | None = None

    @require_auth
    def create_order(self, order_id: str, products: list[dict[str, Any]]) -> dict[str, Any]:
        """주문 생성"""
        if not self.current_user:
            raise APIError("사용자를 찾을 수 없습니다", 404)

        product_objects = [
            Product(id=p["id"], name=p["name"], price=p["price"], stock=p.get("stock", 0))
            for p in products
        ]

        order = self.order_service.create_order(order_id, self.current_user.name, product_objects)

        if order:
            return {"success": True, "order_id": order.order_id, "total": order.calculate_total()}
        raise APIError("주문 생성 실패", 500)

    @require_auth
    def get_order(self, order_id: str) -> dict[str, Any]:
        """주문 조회"""
        order = self.order_service.get_order(order_id)
        if order:
            return {
                "order_id": order.order_id,
                "total": order.calculate_total(),
                "status": order.status,
            }
        raise APIError("주문을 찾을 수 없습니다", 404)

    @require_auth
    def get_order_total(self, order_id: str) -> dict[str, Any]:
        """주문 총액 조회"""
        total = self.order_service.calculate_order_total(order_id)
        if total is not None:
            return {"order_id": order_id, "total": total}
        raise APIError("주문을 찾을 수 없습니다", 404)


class APIRouter:
    """API 라우터 클래스"""

    def __init__(self):
        self.routes: dict[str, Any] = {}

    def register(self, path: str, handler: Any) -> None:
        """라우트 등록"""
        self.routes[path] = handler

    def get_handler(self, path: str) -> Any | None:
        """라우트 핸들러 조회"""
        return self.routes.get(path)

    def list_routes(self) -> list[str]:
        """등록된 라우트 목록"""
        return list(self.routes.keys())


def create_api_response(
    data: Any, success: bool = True, message: str | None = None
) -> dict[str, Any]:
    """API 응답 생성"""
    response = {"success": success, "data": data}
    if message:
        response["message"] = message
    return response


def handle_api_error(error: Exception) -> dict[str, Any]:
    """API 에러 처리"""
    if isinstance(error, APIError):
        return {"success": False, "error": error.message, "status_code": error.status_code}
    return {"success": False, "error": str(error), "status_code": 500}
