/**
 * 고급 패턴 및 실전 케이스
 */

// 고급 제네릭 패턴
interface Monad<T> {
  bind<R>(fn: (value: T) => Monad<R>): Monad<R>;
  map<R>(fn: (value: T) => R): Monad<R>;
  value: T;
}

class MonadImpl<T> implements Monad<T> {
  constructor(public value: T) {}

  bind<R>(fn: (value: T) => Monad<R>): Monad<R> {
    return fn(this.value);
  }

  map<R>(fn: (value: T) => R): Monad<R> {
    return new MonadImpl(fn(this.value));
  }
}

// Maybe 모나드
class Maybe<T> {
  private constructor(private value: T | null) {}

  static just<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  isJust(): boolean {
    return this.value !== null;
  }

  isNothing(): boolean {
    return this.value === null;
  }

  map<R>(fn: (value: T) => R): Maybe<R> {
    if (this.isJust()) {
      return Maybe.just(fn(this.value!));
    }
    return Maybe.nothing<R>();
  }

  bind<R>(fn: (value: T) => Maybe<R>): Maybe<R> {
    if (this.isJust()) {
      return fn(this.value!);
    }
    return Maybe.nothing<R>();
  }

  getOrElse(defaultValue: T): T {
    return this.isJust() ? this.value! : defaultValue;
  }
}

// 고급 데코레이터 패턴
function memoize<T extends (...args: any[]) => any>(
  fn: T,
  maxSize?: number
): T {
  const cache = new Map<string, ReturnType<T>>();

  return ((...args: any[]) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    if (maxSize === undefined || cache.size < maxSize) {
      cache.set(key, result);
    }
    return result;
  }) as T;
}

function rateLimit(
  calls: number,
  period: number
): MethodDecorator {
  const callHistory: number[] = [];

  return function (
    target: any,
    propertyKey: string | symbol,
    descriptor: PropertyDescriptor
  ) {
    const original = descriptor.value;

    descriptor.value = function (...args: any[]) {
      const now = Date.now();
      // 오래된 호출 제거
      while (callHistory.length > 0 && callHistory[0] < now - period) {
        callHistory.shift();
      }

      if (callHistory.length >= calls) {
        throw new Error("Rate limit exceeded");
      }

      callHistory.push(now);
      return original.apply(this, args);
    };

    return descriptor;
  };
}

// 옵저버 패턴
interface Observer {
  update(event: string, data: any): void;
}

class Observable {
  private observers: Observer[] = [];

  attach(observer: Observer): void {
    if (!this.observers.includes(observer)) {
      this.observers.push(observer);
    }
  }

  detach(observer: Observer): void {
    const index = this.observers.indexOf(observer);
    if (index > -1) {
      this.observers.splice(index, 1);
    }
  }

  notify(event: string, data: any): void {
    const observers = [...this.observers];
    for (const observer of observers) {
      try {
        observer.update(event, data);
      } catch (error) {
        // 에러 처리
      }
    }
  }
}

// 전략 패턴
interface Strategy {
  execute(data: any): any;
}

class Context {
  constructor(private strategy: Strategy) {}

  setStrategy(strategy: Strategy): void {
    this.strategy = strategy;
  }

  execute(data: any): any {
    return this.strategy.execute(data);
  }
}

class ConcreteStrategyA implements Strategy {
  execute(data: any): any {
    return `Strategy A: ${data}`;
  }
}

class ConcreteStrategyB implements Strategy {
  execute(data: any): any {
    return `Strategy B: ${data}`;
  }
}

// 팩토리 패턴
interface Product {
  operation(): string;
}

class ConcreteProductA implements Product {
  operation(): string {
    return "Product A";
  }
}

class ConcreteProductB implements Product {
  operation(): string {
    return "Product B";
  }
}

class ProductFactory {
  private static products: Map<string, new () => Product> = new Map([
    ["A", ConcreteProductA],
    ["B", ConcreteProductB]
  ]);

  static create(productType: string): Product {
    const ProductClass = this.products.get(productType);
    if (ProductClass) {
      return new ProductClass();
    }
    throw new Error(`Unknown product type: ${productType}`);
  }

  static register(productType: string, ProductClass: new () => Product): void {
    this.products.set(productType, ProductClass);
  }
}

// 빌더 패턴
interface Query {
  select: string[];
  fromTable?: string;
  where: string[];
  orderBy: string[];
  limit?: number;
}

class QueryBuilder {
  private query: Query = {
    select: [],
    where: [],
    orderBy: []
  };

  select(...columns: string[]): this {
    this.query.select.push(...columns);
    return this;
  }

  fromTable(table: string): this {
    this.query.fromTable = table;
    return this;
  }

  where(condition: string): this {
    this.query.where.push(condition);
    return this;
  }

  orderBy(...columns: string[]): this {
    this.query.orderBy.push(...columns);
    return this;
  }

  limit(n: number): this {
    this.query.limit = n;
    return this;
  }

  build(): Query {
    return { ...this.query };
  }
}

// 체인 오브 리스폰시빌리티 패턴
abstract class Handler {
  protected next?: Handler;

  setNext(handler: Handler): Handler {
    this.next = handler;
    return handler;
  }

  abstract handle(request: any): any;

  protected handleNext(request: any): any {
    if (this.next) {
      return this.next.handle(request);
    }
    return null;
  }
}

class ConcreteHandlerA extends Handler {
  handle(request: any): any {
    if (typeof request === "string" && request.startsWith("A")) {
      return `Handler A processed: ${request}`;
    }
    return this.handleNext(request);
  }
}

class ConcreteHandlerB extends Handler {
  handle(request: any): any {
    if (typeof request === "string" && request.startsWith("B")) {
      return `Handler B processed: ${request}`;
    }
    return this.handleNext(request);
  }
}

// 비동기 패턴
class AsyncQueue<T> {
  private queue: T[] = [];
  private resolvers: Array<(value: T) => void> = [];
  private maxSize: number;

  constructor(maxSize: number = 0) {
    this.maxSize = maxSize;
  }

  async put(item: T): Promise<void> {
    if (this.resolvers.length > 0) {
      const resolve = this.resolvers.shift()!;
      resolve(item);
    } else {
      if (this.maxSize > 0 && this.queue.length >= this.maxSize) {
        throw new Error("Queue is full");
      }
      this.queue.push(item);
    }
  }

  async get(): Promise<T> {
    if (this.queue.length > 0) {
      return this.queue.shift()!;
    }
    return new Promise<T>(resolve => {
      this.resolvers.push(resolve);
    });
  }
}

class AsyncWorker<T> {
  private running: boolean = false;

  constructor(
    private queue: AsyncQueue<T>,
    private workerId: number
  ) {}

  async start(handler: (item: T) => Promise<void>): Promise<void> {
    this.running = true;
    while (this.running) {
      try {
        const item = await Promise.race([
          this.queue.get(),
          new Promise<T>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 1000)
          )
        ]);
        await handler(item);
      } catch (error) {
        if (error instanceof Error && error.message !== "Timeout") {
          throw error;
        }
      }
    }
  }

  stop(): void {
    this.running = false;
  }
}

// 이터레이터 패턴
class TreeNode<T> {
  value: T;
  children: TreeNode<T>[] = [];

  constructor(value: T) {
    this.value = value;
  }

  addChild(child: TreeNode<T>): void {
    this.children.push(child);
  }

  *[Symbol.iterator](): Generator<T> {
    yield this.value;
    for (const child of this.children) {
      yield* child;
    }
  }

  *depthFirst(): Generator<T> {
    const stack: TreeNode<T>[] = [this];
    while (stack.length > 0) {
      const node = stack.pop()!;
      yield node.value;
      stack.push(...node.children.reverse());
    }
  }

  *breadthFirst(): Generator<T> {
    const queue: TreeNode<T>[] = [this];
    while (queue.length > 0) {
      const node = queue.shift()!;
      yield node.value;
      queue.push(...node.children);
    }
  }
}

// 어댑터 패턴
class OldInterface {
  oldMethod(x: number, y: number): number {
    return x + y;
  }
}

interface NewInterface {
  newMethod(data: { x: number; y: number }): number;
}

class Adapter implements NewInterface {
  constructor(private oldInterface: OldInterface) {}

  newMethod(data: { x: number; y: number }): number {
    return this.oldInterface.oldMethod(data.x, data.y);
  }
}

// 프록시 패턴
interface Subject {
  request(): string;
}

class RealSubject implements Subject {
  request(): string {
    return "RealSubject: Handling request";
  }
}

class Proxy implements Subject {
  private cache?: string;

  constructor(private realSubject: RealSubject) {}

  request(): string {
    if (!this.cache) {
      this.cache = this.realSubject.request();
    }
    return `Proxy: ${this.cache}`;
  }
}

// 데코레이터 패턴 (구조적)
interface Component {
  operation(): string;
}

class ConcreteComponent implements Component {
  operation(): string {
    return "ConcreteComponent";
  }
}

class Decorator implements Component {
  constructor(protected component: Component) {}

  operation(): string {
    return this.component.operation();
  }
}

class ConcreteDecoratorA extends Decorator {
  operation(): string {
    return `ConcreteDecoratorA(${super.operation()})`;
  }
}

class ConcreteDecoratorB extends Decorator {
  operation(): string {
    return `ConcreteDecoratorB(${super.operation()})`;
  }
}

// 상태 패턴
interface State {
  handle(context: StateContext): void;
}

class ConcreteStateA implements State {
  handle(context: StateContext): void {
    console.log("State A handling");
    context.setState(new ConcreteStateB());
  }
}

class ConcreteStateB implements State {
  handle(context: StateContext): void {
    console.log("State B handling");
    context.setState(new ConcreteStateA());
  }
}

class StateContext {
  constructor(private state: State) {}

  setState(state: State): void {
    this.state = state;
  }

  request(): void {
    this.state.handle(this);
  }
}

// 비지터 패턴
interface Visitor {
  visitElementA(element: ElementA): void;
  visitElementB(element: ElementB): void;
}

interface Element {
  accept(visitor: Visitor): void;
}

class ElementA implements Element {
  accept(visitor: Visitor): void {
    visitor.visitElementA(this);
  }

  operationA(): string {
    return "ElementA";
  }
}

class ElementB implements Element {
  accept(visitor: Visitor): void {
    visitor.visitElementB(this);
  }

  operationB(): string {
    return "ElementB";
  }
}

class ConcreteVisitor implements Visitor {
  visitElementA(element: ElementA): void {
    console.log(`Visiting ${element.operationA()}`);
  }

  visitElementB(element: ElementB): void {
    console.log(`Visiting ${element.operationB()}`);
  }
}

// 템플릿 메서드 패턴
abstract class AbstractClass {
  templateMethod(): string {
    return `${this.primitiveOperation1()}-${this.primitiveOperation2()}`;
  }

  abstract primitiveOperation1(): string;
  abstract primitiveOperation2(): string;
}

class ConcreteClass extends AbstractClass {
  primitiveOperation1(): string {
    return "Operation1";
  }

  primitiveOperation2(): string {
    return "Operation2";
  }
}

// 고급 함수형 패턴
function compose<T>(...fns: Array<(x: T) => T>): (x: T) => T {
  return (x: T) => fns.reduceRight((acc, fn) => fn(acc), x);
}

function pipe<T>(value: T, ...fns: Array<(x: T) => T>): T {
  return fns.reduce((acc, fn) => fn(acc), value);
}

type Curried<A extends any[], R> = <P extends Partial<A>>(
  ...args: P
) => P extends A
  ? R
  : A extends [...P, ...infer Rest]
  ? Rest extends []
    ? R
    : Curried<Rest, R>
  : never;

function curry<A extends any[], R>(
  fn: (...args: A) => R
): Curried<A, R> {
  return ((...args: any[]) => {
    if (args.length >= fn.length) {
      return fn(...(args as A));
    }
    return (...args2: any[]) =>
      (curry(fn) as any)(...args, ...args2);
  }) as Curried<A, R>;
}

// 레지스트리 패턴
class Registry {
  private static registry: Map<string, new () => any> = new Map();

  static register(name: string) {
    return function <T extends new () => any>(constructor: T): T {
      Registry.registry.set(name, constructor);
      return constructor;
    };
  }

  static get(name: string): new () => any | undefined {
    return Registry.registry.get(name);
  }

  static listAll(): string[] {
    return Array.from(Registry.registry.keys());
  }
}

@Registry.register("type_a")
class RegisteredTypeA {}

@Registry.register("type_b")
class RegisteredTypeB {}

// 고급 타입 패턴
type PositiveInt = number & { __brand: "PositiveInt" };

function createPositiveInt(value: number): PositiveInt {
  if (value <= 0) {
    throw new Error("Value must be positive");
  }
  return value as PositiveInt;
}

type NonEmptyString = string & { __brand: "NonEmptyString" };

function createNonEmptyString(value: string): NonEmptyString {
  if (value.length === 0) {
    throw new Error("String must not be empty");
  }
  return value as NonEmptyString;
}

// 믹스인 패턴 확장
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

function Loggable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    log(message: string) {
      console.log(`[${this.constructor.name}] ${message}`);
    }
  };
}

class BaseUser {
  constructor(public name: string) {}
}

const EnhancedUser = Loggable(Activatable(Timestamped(BaseUser)));

// 이벤트 에미터 고급 패턴
type EventMap = {
  user_created: { id: string; name: string };
  user_updated: { id: string; changes: Record<string, any> };
  user_deleted: { id: string };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private handlers: Map<keyof T, Array<(data: T[keyof T]) => void>> = new Map();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event)!.push(handler);
  }

  off<K extends keyof T>(event: K, handler: (data: T[K]) => void): void {
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

// 리액티브 프로그래밍 패턴
class Observable<T> {
  private subscribers: Array<(value: T) => void> = [];

  subscribe(fn: (value: T) => void): () => void {
    this.subscribers.push(fn);
    return () => {
      const index = this.subscribers.indexOf(fn);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }
    };
  }

  next(value: T): void {
    this.subscribers.forEach(fn => fn(value));
  }

  map<R>(fn: (value: T) => R): Observable<R> {
    const mapped = new Observable<R>();
    this.subscribe(value => mapped.next(fn(value)));
    return mapped;
  }

  filter(predicate: (value: T) => boolean): Observable<T> {
    const filtered = new Observable<T>();
    this.subscribe(value => {
      if (predicate(value)) {
        filtered.next(value);
      }
    });
    return filtered;
  }
}

// 의존성 주입 패턴
interface DependencyContainer {
  register<T>(key: string, factory: () => T): void;
  resolve<T>(key: string): T;
}

class SimpleContainer implements DependencyContainer {
  private services: Map<string, () => any> = new Map();
  private singletons: Map<string, any> = new Map();

  register<T>(key: string, factory: () => T, singleton: boolean = false): void {
    if (singleton) {
      this.services.set(key, () => {
        if (!this.singletons.has(key)) {
          this.singletons.set(key, factory());
        }
        return this.singletons.get(key);
      });
    } else {
      this.services.set(key, factory);
    }
  }

  resolve<T>(key: string): T {
    const factory = this.services.get(key);
    if (!factory) {
      throw new Error(`Service ${key} not found`);
    }
    return factory();
  }
}




