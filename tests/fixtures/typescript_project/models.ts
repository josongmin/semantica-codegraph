/**
 * 데이터 모델 정의
 */

export enum UserRole {
  ADMIN = "admin",
  USER = "user",
  GUEST = "guest"
}

export interface Address {
  street: string;
  city: string;
  zipCode: string;
  country?: string;
}

export interface User {
  name: string;
  age: number;
  email?: string;
  role: UserRole;
  createdAt: Date;
  address?: Address;
}

export class UserModel implements User {
  name: string;
  age: number;
  email?: string;
  role: UserRole;
  createdAt: Date;
  address?: Address;

  constructor(name: string, age: number, email?: string) {
    this.name = name;
    this.age = age;
    this.email = email;
    this.role = UserRole.USER;
    this.createdAt = new Date();
  }

  greet(): string {
    return `Hello, ${this.name}!`;
  }

  static createDefault(): UserModel {
    return new UserModel("Anonymous", 0);
  }

  static fromObject(data: Partial<User>): UserModel {
    const user = new UserModel(
      data.name || "",
      data.age || 0,
      data.email
    );
    if (data.role) {
      user.role = data.role;
    }
    return user;
  }

  toObject(): User {
    return {
      name: this.name,
      age: this.age,
      email: this.email,
      role: this.role,
      createdAt: this.createdAt,
      address: this.address
    };
  }
}

export class AdminUser extends UserModel {
  private permissions: string[];

  constructor(name: string, age: number, permissions: string[], email?: string) {
    super(name, age, email);
    this.role = UserRole.ADMIN;
    this.permissions = permissions;
  }

  async checkPermission(permission: string): Promise<boolean> {
    return this.permissions.includes(permission);
  }

  grantPermission(permission: string): void {
    if (!this.permissions.includes(permission)) {
      this.permissions.push(permission);
    }
  }

  revokePermission(permission: string): boolean {
    const index = this.permissions.indexOf(permission);
    if (index > -1) {
      this.permissions.splice(index, 1);
      return true;
    }
    return false;
  }

  getPermissions(): string[] {
    return [...this.permissions];
  }
}

export interface Product {
  id: string;
  name: string;
  price: number;
  stock: number;
}

export class ProductModel implements Product {
  id: string;
  name: string;
  price: number;
  stock: number;

  constructor(id: string, name: string, price: number, stock: number = 0) {
    this.id = id;
    this.name = name;
    this.price = price;
    this.stock = stock;
  }

  toString(): string {
    return `Product(id=${this.id}, name=${this.name}, price=${this.price})`;
  }

  equals(other: ProductModel): boolean {
    return this.id === other.id;
  }

  applyDiscount(percentage: number): void {
    if (percentage >= 0 && percentage <= 100) {
      this.price *= (1 - percentage / 100);
    }
  }

  isAvailable(): boolean {
    return this.stock > 0;
  }
}

export interface Order {
  orderId: string;
  user: User;
  products: Product[];
  status: OrderStatus;
  createdAt: Date;
}

export enum OrderStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  CANCELLED = "cancelled"
}

export class OrderModel implements Order {
  orderId: string;
  user: User;
  products: Product[];
  status: OrderStatus;
  createdAt: Date;

  constructor(orderId: string, user: User, products: Product[]) {
    this.orderId = orderId;
    this.user = user;
    this.products = products;
    this.status = OrderStatus.PENDING;
    this.createdAt = new Date();
  }

  calculateTotal(): number {
    return this.products.reduce((sum, product) => sum + product.price, 0);
  }

  addProduct(product: Product): void {
    this.products.push(product);
  }

  removeProduct(productId: string): boolean {
    const index = this.products.findIndex(p => p.id === productId);
    if (index > -1) {
      this.products.splice(index, 1);
      return true;
    }
    return false;
  }

  updateStatus(status: OrderStatus): void {
    this.status = status;
  }
}

export type UserWithRole = User & { role: UserRole };

export type ProductWithCategory = Product & { category: string };

