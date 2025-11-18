/**
 * 서비스 레이어
 */

import { User, UserModel, AdminUser, Product, Order, OrderModel, OrderStatus } from "./models";

export interface IUserRepository {
  save(user: User): Promise<void>;
  findById(userId: string): Promise<User | undefined>;
  findAll(): Promise<User[]>;
}

export class InMemoryUserRepository implements IUserRepository {
  private users: Map<string, User> = new Map();

  async save(user: User): Promise<void> {
    this.users.set(user.name, user);
  }

  async findById(userId: string): Promise<User | undefined> {
    return this.users.get(userId);
  }

  async findAll(): Promise<User[]> {
    return Array.from(this.users.values());
  }

  async delete(userId: string): Promise<boolean> {
    return this.users.delete(userId);
  }

  async clear(): Promise<void> {
    this.users.clear();
  }
}

export abstract class ProductService {
  abstract getProduct(productId: string): Promise<Product | undefined>;
  abstract listProducts(): Promise<Product[]>;
}

export class UserService {
  constructor(private repository: IUserRepository) {}

  async createUser(name: string, age: number, email?: string): Promise<UserModel> {
    const user = new UserModel(name, age, email);
    await this.repository.save(user.toObject());
    return user;
  }

  async getUser(userId: string): Promise<UserModel | undefined> {
    const user = await this.repository.findById(userId);
    return user ? UserModel.fromObject(user) : undefined;
  }

  async listUsers(): Promise<UserModel[]> {
    const users = await this.repository.findAll();
    return users.map(u => UserModel.fromObject(u));
  }

  async updateUserEmail(userId: string, email: string): Promise<boolean> {
    const user = await this.repository.findById(userId);
    if (user) {
      user.email = email;
      await this.repository.save(user);
      return true;
    }
    return false;
  }

  async deleteUser(userId: string): Promise<boolean> {
    if (this.repository instanceof InMemoryUserRepository) {
      return await this.repository.delete(userId);
    }
    return false;
  }
}

export class OrderService {
  private orders: Map<string, OrderModel> = new Map();

  constructor(private userService: UserService) {}

  async createOrder(
    orderId: string,
    userId: string,
    products: Product[]
  ): Promise<OrderModel | undefined> {
    const user = await this.userService.getUser(userId);
    if (!user) {
      return undefined;
    }

    const order = new OrderModel(orderId, user.toObject(), products);
    this.orders.set(orderId, order);
    return order;
  }

  async getOrder(orderId: string): Promise<OrderModel | undefined> {
    return this.orders.get(orderId);
  }

  async calculateOrderTotal(orderId: string): Promise<number | undefined> {
    const order = await this.getOrder(orderId);
    return order?.calculateTotal();
  }

  async updateOrderStatus(orderId: string, status: OrderStatus): Promise<boolean> {
    const order = await this.getOrder(orderId);
    if (order) {
      order.updateStatus(status);
      return true;
    }
    return false;
  }

  async listOrders(): Promise<OrderModel[]> {
    return Array.from(this.orders.values());
  }
}

export class AuthService {
  constructor(private userService: UserService) {}

  async authenticate(username: string, password: string): Promise<UserModel | undefined> {
    const user = await this.userService.getUser(username);
    // 실제로는 비밀번호 검증 로직이 들어감
    return user;
  }

  async authorize(user: UserModel, permission: string): Promise<boolean> {
    if (user instanceof AdminUser) {
      return await user.checkPermission(permission);
    }
    return false;
  }

  async requireAuth(user: UserModel | undefined): Promise<UserModel> {
    if (!user) {
      throw new Error("인증이 필요합니다");
    }
    return user;
  }

  async requireAdmin(user: UserModel | undefined): Promise<AdminUser> {
    if (!user || !(user instanceof AdminUser)) {
      throw new Error("관리자 권한이 필요합니다");
    }
    return user;
  }
}

export function validateEmail(email: string): boolean {
  return email.includes("@") && email.split("@")[1].includes(".");
}

export class ValidationService {
  static validateUser(user: Partial<User>): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!user.name || user.name.trim().length === 0) {
      errors.push("이름은 필수입니다");
    }

    if (user.age !== undefined && (user.age < 0 || user.age > 150)) {
      errors.push("나이는 0-150 사이여야 합니다");
    }

    if (user.email && !validateEmail(user.email)) {
      errors.push("유효한 이메일 형식이 아닙니다");
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  static validateProduct(product: Partial<Product>): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!product.id || product.id.trim().length === 0) {
      errors.push("상품 ID는 필수입니다");
    }

    if (!product.name || product.name.trim().length === 0) {
      errors.push("상품명은 필수입니다");
    }

    if (product.price !== undefined && product.price < 0) {
      errors.push("가격은 0 이상이어야 합니다");
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

