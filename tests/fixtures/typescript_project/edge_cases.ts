/**
 * 엣지 케이스 및 이상한 패턴들
 */

// 복잡한 타입 정의
type NestedDict = {
  [key: string]: string | number | NestedDict | any[];
};

type ComplexType =
  | Array<{ [key: string]: number | string | null }>
  | { [key: string]: number[] };

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

// 조건부 타입
type NonNullable<T> = T extends null | undefined ? never : T;

type FunctionReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type ArrayElementType<T> = T extends (infer U)[] ? U : never;

// 복잡한 제네릭
interface Container<T> {
  value: T;
  get(): T;
  set(value: T): void;
  map<U>(fn: (value: T) => U): Container<U>;
}

class GenericContainer<T> implements Container<T> {
  constructor(public value: T) {}

  get(): T {
    return this.value;
  }

  set(value: T): void {
    this.value = value;
  }

  map<U>(fn: (value: T) => U): Container<U> {
    return new GenericContainer(fn(this.value));
  }
}

interface MultiContainer<T, K> {
  first: T;
  second: K;
  swap(): MultiContainer<K, T>;
}

// 오버로드
class Calculator {
  add(x: number, y: number): number;
  add(x: string, y: string): string;
  add(x: number[], y: number[]): number[];
  add(
    x: number | string | number[],
    y: number | string | number[]
  ): number | string | number[] {
    if (typeof x === "number" && typeof y === "number") {
      return x + y;
    }
    if (typeof x === "string" && typeof y === "string") {
      return x + y;
    }
    if (Array.isArray(x) && Array.isArray(y)) {
      return x.map((val, idx) => val + y[idx]);
    }
    throw new Error("Unsupported types");
  }
}

// 싱글톤 패턴
class DatabaseConnection {
  private static instance: DatabaseConnection;
  private connected: boolean = false;

  private constructor() {}

  static getInstance(): DatabaseConnection {
    if (!DatabaseConnection.instance) {
      DatabaseConnection.instance = new DatabaseConnection();
    }
    return DatabaseConnection.instance;
  }

  connect(): void {
    this.connected = true;
  }

  disconnect(): void {
    this.connected = false;
  }

  isConnected(): boolean {
    return this.connected;
  }
}

// 복잡한 데코레이터 패턴 (TypeScript 데코레이터)
function validateInput(validator: (value: any) => boolean) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
      for (const arg of args) {
        if (!validator(arg)) {
          throw new Error(`Invalid input: ${arg}`);
        }
      }
      return originalMethod.apply(this, args);
    };

    return descriptor;
  };
}

function logExecution(
  target: any,
  propertyKey: string,
  descriptor: PropertyDescriptor
) {
  const originalMethod = descriptor.value;

  descriptor.value = function (...args: any[]) {
    console.log(`Executing ${propertyKey} with args:`, args);
    const result = originalMethod.apply(this, args);
    console.log(`Result:`, result);
    return result;
  };

  return descriptor;
}

class ProcessedClass {
  @validateInput((x: any) => typeof x === "number" && x > 0)
  @logExecution
  process(value: number): number {
    return value * 2;
  }
}

// 동적 속성 클래스
class DynamicClass {
  [key: string]: any;

  constructor(initialData: Record<string, any> = {}) {
    Object.assign(this, initialData);
  }

  getProperty(name: string): any {
    return this[name] ?? `<default_${name}>`;
  }

  setProperty(name: string, value: any): void {
    if (!name.startsWith("_")) {
      this[name] = value;
    }
  }
}

// 복잡한 예외 처리
class CustomError extends Error {
  code: number;

  constructor(message: string, code: number = 0) {
    super(message);
    this.name = "CustomError";
    this.code = code;
  }
}

class ValidationError extends CustomError {
  constructor(message: string) {
    super(message, 1001);
    this.name = "ValidationError";
  }
}

class ProcessingError extends CustomError {
  constructor(message: string) {
    super(message, 2001);
    this.name = "ProcessingError";
  }
}

function riskyOperation(data: any): any {
  if (!data) {
    throw new ValidationError("Data is required");
  }

  if (typeof data === "string" && data.length > 1000) {
    throw new ProcessingError("Data too large");
  }

  try {
    const result = typeof data === "string" ? parseInt(data) : data;
    return result * 2;
  } catch (error) {
    throw new ProcessingError(`Failed to process: ${error}`);
  }
}

// 복잡한 함수형 패턴
type FilterFunction = (items: number[]) => number[];

function createFilterFunction(threshold: number): FilterFunction {
  return (items: number[]) => items.filter(x => x > threshold);
}

type ComposeFunction = <T>(...fns: Array<(x: T) => T>) => (x: T) => T;

const compose: ComposeFunction = <T>(...fns: Array<(x: T) => T>) => {
  return (x: T) => fns.reduceRight((acc, fn) => fn(acc), x);
};

// 복잡한 프로토콜/인터페이스
interface Drawable {
  draw(): void;
  getArea(): number;
}

interface Movable {
  move(x: number, y: number): void;
}

abstract class Shape implements Drawable {
  constructor(protected x: number, protected y: number) {}

  abstract getArea(): number;

  draw(): void {
    console.log(`Drawing shape at (${this.x}, ${this.y})`);
  }
}

class Circle extends Shape implements Movable {
  constructor(x: number, y: number, private radius: number) {
    super(x, y);
  }

  getArea(): number {
    return Math.PI * this.radius ** 2;
  }

  move(x: number, y: number): void {
    this.x += x;
    this.y += y;
  }
}

// 복잡한 네스팅
class OuterClass {
  static InnerClass = class {
    static NestedClass = class {
      static deeplyNestedMethod(): string {
        return "deeply nested";
      }
    };

    innerMethod(): string {
      return "inner";
    }
  };

  outerMethod(): OuterClass.InnerClass {
    return new OuterClass.InnerClass();
  }
}

// 이상한 네이밍 패턴
class _PrivateClass {
  private __privateAttr: string = "secret";
}

class __DunderClass__ {
  private __private_attr: string = "secret";
}

class CamelCaseClass {
  mixedCaseMethod(): void {}
}

class snake_case_class {
  method_name(): void {}
}

// 연산자 오버로딩 (TypeScript에서는 직접 지원하지 않지만 패턴으로 구현)
class Vector {
  constructor(public x: number, public y: number) {}

  add(other: Vector): Vector {
    return new Vector(this.x + other.x, this.y + other.y);
  }

  subtract(other: Vector): Vector {
    return new Vector(this.x - other.x, this.y - other.y);
  }

  multiply(scalar: number): Vector {
    return new Vector(this.x * scalar, this.y * scalar);
  }

  toString(): string {
    return `Vector(${this.x}, ${this.y})`;
  }

  equals(other: Vector): boolean {
    return this.x === other.x && this.y === other.y;
  }

  get length(): number {
    return Math.sqrt(this.x ** 2 + this.y ** 2);
  }

  [Symbol.iterator](): Iterator<number> {
    let index = 0;
    const values = [this.x, this.y];
    return {
      next(): IteratorResult<number> {
        if (index < values.length) {
          return { value: values[index++], done: false };
        }
        return { value: undefined, done: true };
      }
    };
  }
}

// 복잡한 비동기 패턴
class AsyncProcessor {
  async processItem(item: any): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 100));
    return item * 2;
  }

  async processBatch(items: any[]): Promise<any[]> {
    const promises = items.map(item => this.processItem(item));
    return Promise.all(promises);
  }

  async processWithRetry(
    item: any,
    maxRetries: number = 3
  ): Promise<any> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.processItem(item);
      } catch (error) {
        if (attempt === maxRetries - 1) {
          throw error;
        }
        await new Promise(resolve =>
          setTimeout(resolve, 100 * (attempt + 1))
        );
      }
    }
    throw new Error("Max retries exceeded");
  }
}

// 이벤트 핸들러 패턴
type EventHandler<T = any> = (event: T) => void | Promise<void>;

interface EventEmitter<T extends Record<string, any>> {
  on<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
  off<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
  emit<K extends keyof T>(event: K, data: T[K]): void;
}

class SimpleEventEmitter<T extends Record<string, any>>
  implements EventEmitter<T>
{
  private handlers: Map<keyof T, EventHandler[]> = new Map();

  on<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event)!.push(handler);
  }

  off<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void {
    const handlers = this.handlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    const handlers = this.handlers.get(event);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }
}

// 타입 가드
function isString(value: any): value is string {
  return typeof value === "string";
}

function isNumber(value: any): value is number {
  return typeof value === "number";
}

function isArray<T>(value: any): value is T[] {
  return Array.isArray(value);
}

function isValidUser(obj: any): obj is { name: string; age: number } {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "name" in obj &&
    "age" in obj &&
    typeof obj.name === "string" &&
    typeof obj.age === "number"
  );
}

function processUserData(
  data: any
): { processed: boolean; user?: { name: string; age: number } } | null {
  if (isValidUser(data)) {
    return { processed: true, user: data };
  }
  return null;
}

// 복잡한 유니온 타입
type Status = "pending" | "processing" | "completed" | "failed" | "cancelled";

function getStatusColor(status: Status): string {
  const colors: Record<Status, string> = {
    pending: "yellow",
    processing: "blue",
    completed: "green",
    failed: "red",
    cancelled: "gray"
  };
  return colors[status];
}

// 복잡한 함수 시그니처
function complexFunctionSignature(
  required: string,
  optional?: number,
  ...rest: string[]
): {
  required: string;
  optional?: number;
  rest: string[];
} {
  return {
    required,
    optional,
    rest
  };
}

// 제네릭 제약 조건
interface HasId {
  id: string | number;
}

function findById<T extends HasId>(items: T[], id: string | number): T | undefined {
  return items.find(item => item.id === id);
}

// 매핑된 타입
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

type Partial<T> = {
  [P in keyof T]?: T[P];
};

type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};

type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// 조건부 타입을 사용한 유틸리티
type NonNullable<T> = T extends null | undefined ? never : T;

type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

// 복잡한 주석 패턴
class CommentedClass {
  /**
   * 매우 긴 독스트링을 가진 클래스.
   *
   * 이 클래스는 여러 줄에 걸친 설명을 가지고 있으며,
   * 복잡한 기능을 수행합니다.
   *
   * @param value - 초기값
   * @returns CommentedClass 인스턴스
   * @throws {Error} 잘못된 값이 전달될 경우
   *
   * @example
   * ```typescript
   * const obj = new CommentedClass(42);
   * console.log(obj.value); // 42
   * ```
   */
  constructor(public value: number) {
    // 인라인 주석
    this.value = value; // 값 설정
  }

  /**
   * 메서드 독스트링
   * @todo 구현 필요
   * @fixme 버그 수정 필요
   * @note 중요 참고사항
   */
  method(): void {
    // TODO: 구현 필요
    // FIXME: 버그 수정 필요
    // NOTE: 중요 참고사항
    // XXX: 위험한 코드
  }
}

// 믹스인 패턴
type Constructor<T = {}> = new (...args: any[]) => T;

function Timestamped<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    timestamp = new Date();
  };
}

function Activatable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    isActive = false;

    activate() {
      this.isActive = true;
    }

    deactivate() {
      this.isActive = false;
    }
  };
}

class User {
  constructor(public name: string) {}
}

const TimestampedActivatableUser = Timestamped(Activatable(User));

// 프로퍼티 데코레이터
function readonly(target: any, propertyKey: string) {
  Object.defineProperty(target, propertyKey, {
    writable: false,
    configurable: false
  });
}

class ReadonlyClass {
  @readonly
  readonlyProperty: string = "cannot be changed";
}

// 파라미터 데코레이터
function validate(target: any, propertyKey: string, parameterIndex: number) {
  // 파라미터 검증 로직
}

class ValidatedClass {
  method(@validate value: number): void {
    // 메서드 구현
  }
}

// 복잡한 제네릭 함수
function identity<T>(arg: T): T {
  return arg;
}

function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

function setProperty<T, K extends keyof T>(
  obj: T,
  key: K,
  value: T[K]
): void {
  obj[key] = value;
}

// 재귀적 타입
type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

// 브랜드 타입
type Brand<T, B> = T & { __brand: B };

type UserId = Brand<string, "UserId">;
type ProductId = Brand<string, "ProductId">;

function createUserId(id: string): UserId {
  return id as UserId;
}

function createProductId(id: string): ProductId {
  return id as ProductId;
}

// 템플릿 리터럴 타입
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";
type ApiEndpoint = `/api/${string}`;
type FullEndpoint = `${HttpMethod} ${ApiEndpoint}`;

// 인덱스 시그니처
interface StringDictionary {
  [key: string]: string;
}

interface NumberDictionary {
  [key: number]: string;
}

// 복잡한 조건부 타입
type ExtractArrayType<T> = T extends (infer U)[] ? U : never;

type ExtractPromiseType<T> = T extends Promise<infer U> ? U : never;

type Flatten<T> = T extends (infer U)[] ? U : T;

// 유니온에서 교집합 추출
type UnionToIntersection<U> = (U extends any ? (x: U) => void : never) extends (
  x: infer I
) => void
  ? I
  : never;

