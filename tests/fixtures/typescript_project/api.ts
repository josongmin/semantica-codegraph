/**
 * API 관련 코드
 */

import { User, UserModel, AdminUser, Product, OrderModel } from "./models";
import { UserService, OrderService, AuthService } from "./services";

export class APIError extends Error {
  statusCode: number;

  constructor(message: string, statusCode: number = 400) {
    super(message);
    this.name = "APIError";
    this.statusCode = statusCode;
  }
}

export type APIResponse<T> = {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
};

export function requireAuth(
  target: any,
  propertyName: string,
  descriptor: PropertyDescriptor
): void {
  const method = descriptor.value;

  descriptor.value = function (...args: any[]) {
    if (!this.currentUser) {
      throw new APIError("인증이 필요합니다", 401);
    }
    return method.apply(this, args);
  };
}

export function requireAdmin(
  target: any,
  propertyName: string,
  descriptor: PropertyDescriptor
): void {
  const method = descriptor.value;

  descriptor.value = function (...args: any[]) {
    if (!this.currentUser || !(this.currentUser instanceof AdminUser)) {
      throw new APIError("관리자 권한이 필요합니다", 403);
    }
    return method.apply(this, args);
  };
}

export class UserAPI {
  currentUser?: UserModel;

  constructor(
    private userService: UserService,
    private authService: AuthService
  ) {}

  async login(
    username: string,
    password: string
  ): Promise<APIResponse<User>> {
    const user = await this.authService.authenticate(username, password);
    if (user) {
      this.currentUser = user;
      return {
        success: true,
        data: user.toObject()
      };
    }
    throw new APIError("로그인 실패", 401);
  }

  @requireAuth
  async getProfile(): Promise<APIResponse<User>> {
    if (!this.currentUser) {
      throw new APIError("사용자를 찾을 수 없습니다", 404);
    }
    return {
      success: true,
      data: this.currentUser.toObject()
    };
  }

  async createUser(
    name: string,
    age: number,
    email?: string
  ): Promise<APIResponse<User>> {
    const user = await this.userService.createUser(name, age, email);
    return {
      success: true,
      data: user.toObject()
    };
  }

  @requireAuth
  async updateEmail(email: string): Promise<APIResponse<void>> {
    if (!this.currentUser) {
      throw new APIError("사용자를 찾을 수 없습니다", 404);
    }

    const success = await this.userService.updateUserEmail(
      this.currentUser.name,
      email
    );
    if (success) {
      return {
        success: true,
        message: "이메일이 업데이트되었습니다"
      };
    }
    return {
      success: false,
      message: "업데이트 실패"
    };
  }
}

export class OrderAPI {
  currentUser?: UserModel;

  constructor(
    private orderService: OrderService,
    private authService: AuthService
  ) {}

  @requireAuth
  async createOrder(
    orderId: string,
    products: Product[]
  ): Promise<APIResponse<{ orderId: string; total: number }>> {
    if (!this.currentUser) {
      throw new APIError("사용자를 찾을 수 없습니다", 404);
    }

    const order = await this.orderService.createOrder(
      orderId,
      this.currentUser.name,
      products
    );

    if (order) {
      return {
        success: true,
        data: {
          orderId: order.orderId,
          total: order.calculateTotal()
        }
      };
    }
    throw new APIError("주문 생성 실패", 500);
  }

  @requireAuth
  async getOrder(orderId: string): Promise<APIResponse<OrderModel>> {
    const order = await this.orderService.getOrder(orderId);
    if (order) {
      return {
        success: true,
        data: order as any
      };
    }
    throw new APIError("주문을 찾을 수 없습니다", 404);
  }

  @requireAuth
  async getOrderTotal(
    orderId: string
  ): Promise<APIResponse<{ orderId: string; total: number }>> {
    const total = await this.orderService.calculateOrderTotal(orderId);
    if (total !== undefined) {
      return {
        success: true,
        data: {
          orderId,
          total
        }
      };
    }
    throw new APIError("주문을 찾을 수 없습니다", 404);
  }
}

export class APIRouter {
  private routes: Map<string, any> = new Map();

  register(path: string, handler: any): void {
    this.routes.set(path, handler);
  }

  getHandler(path: string): any | undefined {
    return this.routes.get(path);
  }

  listRoutes(): string[] {
    return Array.from(this.routes.keys());
  }

  removeRoute(path: string): boolean {
    return this.routes.delete(path);
  }
}

export function createAPIResponse<T>(
  data: T,
  success: boolean = true,
  message?: string
): APIResponse<T> {
  const response: APIResponse<T> = {
    success,
    data
  };
  if (message) {
    response.message = message;
  }
  return response;
}

export function handleAPIError(error: unknown): APIResponse<never> {
  if (error instanceof APIError) {
    return {
      success: false,
      error: error.message,
      message: `Status: ${error.statusCode}`
    };
  }
  return {
    success: false,
    error: error instanceof Error ? error.message : String(error)
  };
}

export interface RouteHandler {
  (req: any, res: any): Promise<void> | void;
}

export class Middleware {
  static async errorHandler(
    error: unknown,
    req: any,
    res: any,
    next: () => void
  ): Promise<void> {
    const response = handleAPIError(error);
    res.status(
      error instanceof APIError ? error.statusCode : 500
    ).json(response);
  }

  static async authMiddleware(
    req: any,
    res: any,
    next: () => void
  ): Promise<void> {
    // 실제 구현에서는 토큰 검증 로직이 들어감
    next();
  }
}

