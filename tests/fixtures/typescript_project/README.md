# TypeScript Project

테스트용 TypeScript 프로젝트입니다.

## Overview

이 프로젝트는 전자상거래 시스템의 핵심 도메인을 구현합니다.

## Modules

### models.ts
Core data models with TypeScript types and classes.

- `UserModel`: User entity with greeting functionality
- `AdminUser`: Extended user with permission management
- `ProductModel`: Product with discount and availability
- `OrderModel`: Order with total calculation

### services.ts
Business logic layer with async operations.

- `UserService`: User CRUD operations
- `OrderService`: Order management and calculations
- `AuthService`: Authentication and authorization
- `ValidationService`: Input validation

### utils.ts
Utility functions and helpers.

- `calculateTotal`: Calculate sum of item prices
- `filterByPrice`: Filter items by price range
- `debounce`: Delay function execution
- `throttle`: Limit function call frequency
- `Cache<T>`: Generic caching implementation
- `retry`: Async retry with exponential backoff

### api.ts
HTTP API endpoints and request handling.

- `UserAPI`: User management endpoints
- `OrderAPI`: Order processing endpoints
- Error handling and middleware

## Design Patterns

### Async/Await Pattern
All service methods use async/await for clean asynchronous code.

### Repository Pattern
`InMemoryUserRepository` implements data access interface.

### Generic Programming
`Cache<T>` demonstrates TypeScript generics usage.

### Validation Pattern
`ValidationService` provides centralized validation logic.

## Usage Example

```typescript
// Create user
const user = new UserModel("John", 25, "john@example.com");

// Create order
const order = new OrderModel("ORD-001", user.toObject(), products);
const total = order.calculateTotal();

// Authentication
const authService = new AuthService(userService);
const authenticatedUser = await authService.authenticate("john", "password");
```

## Testing

이 프로젝트는 검색 엔진의 TypeScript 파싱 및 인덱싱 테스트를 위해 사용됩니다.

