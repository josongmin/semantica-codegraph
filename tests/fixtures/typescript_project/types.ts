/**
 * 타입 정의
 */

export type UserRole = "admin" | "user" | "guest";

export type ID = string | number;

export type Timestamp = number | Date;

export interface BaseEntity {
  id: ID;
  createdAt: Timestamp;
  updatedAt?: Timestamp;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface QueryParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  filter?: Record<string, any>;
}

export type EventHandler<T = any> = (event: T) => void | Promise<void>;

export interface EventEmitter<T extends Record<string, any>> {
  on<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
  off<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
  emit<K extends keyof T>(event: K, data: T[K]): void;
}

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type ReadonlyDeep<T> = {
  readonly [P in keyof T]: T[P] extends object ? ReadonlyDeep<T[P]> : T[P];
};

export interface Repository<T extends BaseEntity> {
  findById(id: ID): Promise<T | undefined>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: ID): Promise<boolean>;
}

export interface Service<T extends BaseEntity> {
  get(id: ID): Promise<T | undefined>;
  list(params?: QueryParams): Promise<PaginatedResponse<T>>;
  create(data: Partial<T>): Promise<T>;
  update(id: ID, data: Partial<T>): Promise<T>;
  delete(id: ID): Promise<boolean>;
}

export type AsyncFunction<T extends any[], R> = (...args: T) => Promise<R>;

export type SyncFunction<T extends any[], R> = (...args: T) => R;

export type FunctionType<T extends any[], R> =
  | AsyncFunction<T, R>
  | SyncFunction<T, R>;

export interface Validator<T> {
  validate(value: T): { valid: boolean; errors: string[] };
}

export type Constructor<T = {}> = new (...args: any[]) => T;

export type Mixin<T extends Constructor> = InstanceType<T>;

export function createMixin<T extends Constructor>(
  Base: T,
  ...mixins: Constructor[]
): T {
  return mixins.reduce(
    (acc, mixin) => {
      Object.getOwnPropertyNames(mixin.prototype).forEach(name => {
        if (name !== "constructor") {
          Object.defineProperty(
            acc.prototype,
            name,
            Object.getOwnPropertyDescriptor(mixin.prototype, name) ||
              Object.create(null)
          );
        }
      });
      return acc;
    },
    Base
  ) as T;
}
