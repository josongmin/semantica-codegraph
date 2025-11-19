/**
 * 프레임워크 스타일 패턴 및 실전 케이스
 */

// 미들웨어 패턴
interface Middleware {
  handle(request: Record<string, any>): Record<string, any>;
  setNext(middleware: Middleware): void;
}

abstract class BaseMiddleware implements Middleware {
  protected next?: Middleware;

  setNext(middleware: Middleware): void {
    this.next = middleware;
  }

  handle(request: Record<string, any>): Record<string, any> {
    request = this.processRequest(request);
    const response = this.next ? this.next.handle(request) : {};
    return this.processResponse(response);
  }

  protected processRequest(request: Record<string, any>): Record<string, any> {
    return request;
  }

  protected processResponse(response: Record<string, any>): Record<string, any> {
    return response;
  }
}

class AuthMiddleware extends BaseMiddleware {
  protected processRequest(request: Record<string, any>): Record<string, any> {
    const token = request.token;
    if (!token) {
      throw new Error("Authentication required");
    }
    request.user = this.authenticate(token);
    return request;
  }

  private authenticate(token: string): Record<string, any> {
    return { id: "user123", name: "Test User" };
  }
}

class LoggingMiddleware extends BaseMiddleware {
  protected processRequest(request: Record<string, any>): Record<string, any> {
    console.log(`[REQUEST] ${new Date()}: ${request.path || "unknown"}`);
    return request;
  }

  protected processResponse(response: Record<string, any>): Record<string, any> {
    console.log(`[RESPONSE] ${new Date()}: ${response.status || "unknown"}`);
    return response;
  }
}

class ValidationMiddleware extends BaseMiddleware {
  protected processRequest(request: Record<string, any>): Record<string, any> {
    if (!request.data) {
      throw new Error("Data is required");
    }
    return request;
  }
}

// 라우터 패턴
interface Route {
  path: string;
  handler: (...args: any[]) => any;
  methods: string[];
  middleware?: Middleware[];
}

class Router {
  private routes: Route[] = [];

  addRoute(
    path: string,
    handler: (...args: any[]) => any,
    methods: string[] = ["GET"],
    middleware?: Middleware[]
  ): void {
    this.routes.push({ path, handler, methods, middleware });
  }

  get(path: string, handler: (...args: any[]) => any, middleware?: Middleware[]): void {
    this.addRoute(path, handler, ["GET"], middleware);
  }

  post(path: string, handler: (...args: any[]) => any, middleware?: Middleware[]): void {
    this.addRoute(path, handler, ["POST"], middleware);
  }

  findRoute(path: string, method: string): { route: Route; params: Record<string, string> } | null {
    for (const route of this.routes) {
      if (route.methods.includes(method)) {
        const params = this.matchPath(route.path, path);
        if (params !== null) {
          return { route, params };
        }
      }
    }
    return null;
  }

  private matchPath(pattern: string, path: string): Record<string, string> | null {
    const patternParts = pattern.split("/");
    const pathParts = path.split("/");

    if (patternParts.length !== pathParts.length) {
      return null;
    }

    const params: Record<string, string> = {};
    for (let i = 0; i < patternParts.length; i++) {
      if (patternParts[i].startsWith("{")) {
        const paramName = patternParts[i].slice(1, -1);
        params[paramName] = pathParts[i];
      } else if (patternParts[i] !== pathParts[i]) {
        return null;
      }
    }
    return params;
  }
}

// 의존성 주입 컨테이너
class ServiceContainer {
  private services: Map<string, any> = new Map();
  private factories: Map<string, () => any> = new Map();
  private singletons: Map<string, any> = new Map();

  register(name: string, service: any, singleton: boolean = false): void {
    if (singleton) {
      this.singletons.set(name, service);
    } else {
      this.services.set(name, service);
    }
  }

  registerFactory(name: string, factory: () => any, singleton: boolean = false): void {
    if (singleton) {
      this.factories.set(name, () => {
        if (!this.singletons.has(name)) {
          this.singletons.set(name, factory());
        }
        return this.singletons.get(name);
      });
    } else {
      this.factories.set(name, factory);
    }
  }

  get<T>(name: string): T {
    if (this.services.has(name)) {
      return this.services.get(name);
    }
    if (this.factories.has(name)) {
      return this.factories.get(name)!();
    }
    if (this.singletons.has(name)) {
      return this.singletons.get(name);
    }
    throw new Error(`Service ${name} not found`);
  }
}

// 이벤트 버스 패턴
class EventBus {
  private handlers: Map<string, Array<(data: any) => void>> = new Map();
  private asyncHandlers: Map<string, Array<(data: any) => Promise<void>>> = new Map();

  subscribe(eventType: string, handler: (data: any) => void | Promise<void>): void {
    if (handler.constructor.name === "AsyncFunction") {
      if (!this.asyncHandlers.has(eventType)) {
        this.asyncHandlers.set(eventType, []);
      }
      this.asyncHandlers.get(eventType)!.push(handler as (data: any) => Promise<void>);
    } else {
      if (!this.handlers.has(eventType)) {
        this.handlers.set(eventType, []);
      }
      this.handlers.get(eventType)!.push(handler as (data: any) => void);
    }
  }

  unsubscribe(eventType: string, handler: (data: any) => void | Promise<void>): void {
    const syncHandlers = this.handlers.get(eventType);
    if (syncHandlers) {
      const index = syncHandlers.indexOf(handler as (data: any) => void);
      if (index > -1) {
        syncHandlers.splice(index, 1);
      }
    }

    const asyncHandlers = this.asyncHandlers.get(eventType);
    if (asyncHandlers) {
      const index = asyncHandlers.indexOf(handler as (data: any) => Promise<void>);
      if (index > -1) {
        asyncHandlers.splice(index, 1);
      }
    }
  }

  publish(eventType: string, data: any): void {
    const syncHandlers = this.handlers.get(eventType);
    if (syncHandlers) {
      syncHandlers.forEach(handler => handler(data));
    }

    const asyncHandlers = this.asyncHandlers.get(eventType);
    if (asyncHandlers) {
      asyncHandlers.forEach(handler => {
        handler(data).catch(console.error);
      });
    }
  }
}

// 커맨드 패턴
interface Command {
  execute(): any;
  undo(): any;
}

class CommandInvoker {
  private history: Command[] = [];

  execute(command: Command): any {
    const result = command.execute();
    this.history.push(command);
    return result;
  }

  undo(): any {
    if (this.history.length > 0) {
      const command = this.history.pop()!;
      return command.undo();
    }
    return null;
  }
}

class AddCommand implements Command {
  constructor(
    private target: number[],
    private value: number
  ) {}

  execute(): any {
    this.target.push(this.value);
    return this.target.length;
  }

  undo(): any {
    if (this.target.length > 0 && this.target[this.target.length - 1] === this.value) {
      this.target.pop();
    }
    return this.target.length;
  }
}

// 메멘토 패턴
class Memento {
  constructor(private state: Record<string, any>) {}

  getState(): Record<string, any> {
    return { ...this.state };
  }
}

class Originator {
  private state: Record<string, any> = {};

  setState(state: Record<string, any>): void {
    this.state = state;
  }

  save(): Memento {
    return new Memento(this.state);
  }

  restore(memento: Memento): void {
    this.state = memento.getState();
  }

  getState(): Record<string, any> {
    return { ...this.state };
  }
}

class Caretaker {
  private history: Memento[] = [];

  constructor(private originator: Originator) {}

  save(): void {
    this.history.push(this.originator.save());
  }

  restore(index: number = -1): void {
    if (index >= 0 && index < this.history.length) {
      this.originator.restore(this.history[index]);
    } else if (index < 0 && Math.abs(index) <= this.history.length) {
      this.originator.restore(this.history[this.history.length + index]);
    }
  }
}

// 플라이웨이트 패턴
class Flyweight {
  constructor(private intrinsicState: string) {}

  operation(extrinsicState: string): string {
    return `${this.intrinsicState}-${extrinsicState}`;
  }
}

class FlyweightFactory {
  private flyweights: Map<string, Flyweight> = new Map();

  getFlyweight(key: string): Flyweight {
    if (!this.flyweights.has(key)) {
      this.flyweights.set(key, new Flyweight(key));
    }
    return this.flyweights.get(key)!;
  }

  count(): number {
    return this.flyweights.size;
  }
}

// 인터프리터 패턴
interface Expression {
  interpret(context: Record<string, any>): any;
}

class TerminalExpression implements Expression {
  constructor(private value: string) {}

  interpret(context: Record<string, any>): any {
    return context[this.value] ?? this.value;
  }
}

abstract class NonTerminalExpression implements Expression {
  constructor(
    protected left: Expression,
    protected right: Expression
  ) {}

  interpret(context: Record<string, any>): any {
    const leftValue = this.left.interpret(context);
    const rightValue = this.right.interpret(context);
    return this.operate(leftValue, rightValue);
  }

  abstract operate(left: any, right: any): any;
}

class AddExpression extends NonTerminalExpression {
  operate(left: any, right: any): any {
    return left + right;
  }
}

class MultiplyExpression extends NonTerminalExpression {
  operate(left: any, right: any): any {
    return left * right;
  }
}

// 레포지토리 패턴
interface Repository<T> {
  findById(id: string): T | undefined;
  findAll(): T[];
  save(entity: T): T;
  delete(id: string): boolean;
}

class InMemoryRepository<T extends { id?: string }> implements Repository<T> {
  private entities: Map<string, T> = new Map();

  findById(id: string): T | undefined {
    return this.entities.get(id);
  }

  findAll(): T[] {
    return Array.from(this.entities.values());
  }

  save(entity: T): T {
    const id = entity.id ?? String(Math.random());
    this.entities.set(id, { ...entity, id });
    return entity;
  }

  delete(id: string): boolean {
    return this.entities.delete(id);
  }
}

// 유닛 오브 워크 패턴
class UnitOfWork<T> {
  private newEntities: T[] = [];
  private modifiedEntities: T[] = [];
  private deletedEntities: T[] = [];

  registerNew(entity: T): void {
    this.newEntities.push(entity);
  }

  registerModified(entity: T): void {
    if (!this.modifiedEntities.includes(entity)) {
      this.modifiedEntities.push(entity);
    }
  }

  registerDeleted(entity: T): void {
    this.deletedEntities.push(entity);
  }

  commit(): void {
    this.newEntities.forEach(entity => console.log(`Inserting ${entity}`));
    this.modifiedEntities.forEach(entity => console.log(`Updating ${entity}`));
    this.deletedEntities.forEach(entity => console.log(`Deleting ${entity}`));

    this.newEntities = [];
    this.modifiedEntities = [];
    this.deletedEntities = [];
  }

  rollback(): void {
    this.newEntities = [];
    this.modifiedEntities = [];
    this.deletedEntities = [];
  }
}

// 스펙ification 패턴
interface Specification<T> {
  isSatisfiedBy(candidate: T): boolean;
  and(other: Specification<T>): Specification<T>;
  or(other: Specification<T>): Specification<T>;
  not(): Specification<T>;
}

abstract class BaseSpecification<T> implements Specification<T> {
  abstract isSatisfiedBy(candidate: T): boolean;

  and(other: Specification<T>): Specification<T> {
    return new AndSpecification(this, other);
  }

  or(other: Specification<T>): Specification<T> {
    return new OrSpecification(this, other);
  }

  not(): Specification<T> {
    return new NotSpecification(this);
  }
}

class AndSpecification<T> extends BaseSpecification<T> {
  constructor(
    private spec1: Specification<T>,
    private spec2: Specification<T>
  ) {
    super();
  }

  isSatisfiedBy(candidate: T): boolean {
    return this.spec1.isSatisfiedBy(candidate) && this.spec2.isSatisfiedBy(candidate);
  }
}

class OrSpecification<T> extends BaseSpecification<T> {
  constructor(
    private spec1: Specification<T>,
    private spec2: Specification<T>
  ) {
    super();
  }

  isSatisfiedBy(candidate: T): boolean {
    return this.spec1.isSatisfiedBy(candidate) || this.spec2.isSatisfiedBy(candidate);
  }
}

class NotSpecification<T> extends BaseSpecification<T> {
  constructor(private spec: Specification<T>) {
    super();
  }

  isSatisfiedBy(candidate: T): boolean {
    return !this.spec.isSatisfiedBy(candidate);
  }
}

class AgeSpecification extends BaseSpecification<{ age: number }> {
  constructor(private minAge: number) {
    super();
  }

  isSatisfiedBy(candidate: { age: number }): boolean {
    return candidate.age >= this.minAge;
  }
}

// 파이프라인 패턴
class Pipeline<T> {
  private stages: Array<(data: T) => T> = [];

  addStage(stage: (data: T) => T): this {
    this.stages.push(stage);
    return this;
  }

  execute(data: T): T {
    return this.stages.reduce((acc, stage) => stage(acc), data);
  }
}

// 체이닝 빌더 패턴
class QueryBuilder {
  private select: string[] = [];
  private fromTable?: string;
  private where: string[] = [];
  private orderBy: string[] = [];
  private limit?: number;

  selectColumns(...columns: string[]): this {
    this.select.push(...columns);
    return this;
  }

  from(table: string): this {
    this.fromTable = table;
    return this;
  }

  whereCondition(condition: string): this {
    this.where.push(condition);
    return this;
  }

  orderByColumns(...columns: string[]): this {
    this.orderBy.push(...columns);
    return this;
  }

  limitRows(n: number): this {
    this.limit = n;
    return this;
  }

  build(): string {
    const parts: string[] = [];
    if (this.select.length > 0) {
      parts.push(`SELECT ${this.select.join(", ")}`);
    }
    if (this.fromTable) {
      parts.push(`FROM ${this.fromTable}`);
    }
    if (this.where.length > 0) {
      parts.push(`WHERE ${this.where.join(" AND ")}`);
    }
    if (this.orderBy.length > 0) {
      parts.push(`ORDER BY ${this.orderBy.join(", ")}`);
    }
    if (this.limit !== undefined) {
      parts.push(`LIMIT ${this.limit}`);
    }
    return parts.join(" ");
  }
}
