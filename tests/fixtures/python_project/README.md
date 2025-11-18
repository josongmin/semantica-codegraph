# Python Project

테스트용 Python 프로젝트입니다.

## 주요 모듈

### models.py
사용자, 관리자, 상품, 주문 데이터 모델을 정의합니다.

- `User`: 기본 사용자 클래스
- `Admin`: 관리자 권한을 가진 사용자
- `Product`: 상품 정보 (가격, 재고)
- `Order`: 주문 정보 및 총액 계산

### services.py
비즈니스 로직을 처리하는 서비스 레이어입니다.

- `UserService`: 사용자 생성, 조회, 업데이트
- `OrderService`: 주문 생성 및 총액 계산
- `AuthService`: 인증 및 권한 확인

### utils.py
공통 유틸리티 함수들입니다.

- `calculate_total`: 아이템 총액 계산
- `filter_by_price`: 가격 범위 필터링
- `Cache`: 간단한 캐시 구현

### api.py
REST API 엔드포인트 구현입니다.

- `UserAPI`: 사용자 관련 API
- `OrderAPI`: 주문 관련 API
- `require_auth`: 인증 데코레이터

## 디자인 패턴

### framework_patterns.py
실전 프레임워크 패턴 구현:

- Middleware Pattern: 요청/응답 처리 체인
- Router Pattern: URL 라우팅 및 매칭
- Dependency Injection: 서비스 컨테이너
- Event Bus: 이벤트 기반 아키텍처
- Command Pattern: 실행/취소 가능한 커맨드
- Repository Pattern: 데이터 접근 추상화
- Query Builder: SQL 쿼리 체이닝
- Specification Pattern: 비즈니스 규칙 조합

## 사용 예시

```python
# 사용자 생성
user = User("홍길동", 30, "hong@example.com")

# 주문 생성
order = Order("ORD-001", user, [product1, product2])
total = order.calculate_total()

# 인증
auth_service = AuthService(user_service)
authenticated_user = auth_service.authenticate("hong", "password123")
```

