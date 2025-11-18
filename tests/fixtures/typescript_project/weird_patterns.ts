/**
 * 이상한 패턴과 엣지 케이스
 */

// 전역 상태 남용
const globalState: Record<string, any> = {};
const moduleCache: Record<string, any> = {};

// 동적 속성 접근
class DynamicPropertyAccess {
  [key: string]: any;

  constructor(initialData: Record<string, any> = {}) {
    Object.assign(this, initialData);
  }

  getProperty(name: string): any {
    return this[name] ?? `<default_${name}>`;
  }
}

// 함수를 클래스로 변환
class FunctionAsClass {
  constructor(private func: Function) {}

  call(...args: any[]): any {
    return this.func(...args);
  }

  get name(): string {
    return this.func.name;
  }
}

// 클래스를 함수처럼 사용
class CallableClass {
  constructor(private multiplier: number = 1) {}

  call(value: number): number {
    return value * this.multiplier;
  }

  add(other: CallableClass): CallableClass {
    return new CallableClass(this.multiplier + other.multiplier);
  }

  multiply(factor: number): CallableClass {
    return new CallableClass(this.multiplier * factor);
  }
}

// 속성 접근이 함수 호출인 클래스
class FunctionAttribute {
  [key: string]: any;

  constructor() {
    return new Proxy(this, {
      get(target, prop: string) {
        if (prop.startsWith("_")) {
          return target[prop];
        }
        return (...args: any[]) =>
          `Called ${prop} with ${JSON.stringify(args)}`;
      }
    });
  }
}

// 무한 재귀 가능한 클래스
class RecursiveClass {
  value: any;
  selfRef?: RecursiveClass;

  constructor(value: any = null) {
    this.value = value;
    this.selfRef = this;
  }

  setRef(ref: RecursiveClass): void {
    this.selfRef = ref;
  }

  getChainLength(): number {
    if (!this.selfRef || this.selfRef === this) {
      return 1;
    }
    return 1 + this.selfRef.getChainLength();
  }
}

// 순환 참조
class NodeA {
  refB?: NodeB;
}

class NodeB {
  refA?: NodeA;
}

// 빌더 패턴
class Builder {
  private data: Record<string, any> = {};

  set<K extends string, V>(key: K, value: V): Builder & { [P in K]: V } {
    this.data[key] = value;
    return this as any;
  }

  build(): Record<string, any> {
    return { ...this.data };
  }
}

// 데코레이터 체이닝
function decorator1(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const original = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log("Decorator 1");
    return original.apply(this, args);
  };
}

function decorator2(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const original = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log("Decorator 2");
    return original.apply(this, args);
  };
}

function decorator3(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const original = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log("Decorator 3");
    return original.apply(this, args);
  };
}

class DecoratedClass {
  @decorator1
  @decorator2
  @decorator3
  heavilyDecoratedMethod(x: number): number {
    return x * 2;
  }
}

// 람다/화살표 함수 체이닝
const lambdaChain = [
  (x: number) => x + 1,
  (x: number) => x * 2,
  (x: number) => x ** 2,
  (x: number) => String(x)
];

function applyLambdaChain(value: number): string {
  return lambdaChain.reduce((acc, fn) => fn(acc), value);
}

// 클로저 남용
function createCounter(start: number = 0) {
  let count = start;

  return {
    increment: () => ++count,
    decrement: () => --count,
    get: () => count,
    reset: () => (count = start)
  };
}

// 부분 적용
function multiply(x: number, y: number, z: number): number {
  return x * y * z;
}

const multiplyBy2 = (y: number, z: number) => multiply(2, y, z);
const multiplyBy2And3 = (z: number) => multiplyBy2(3, z);

// 연산자 오버로딩 패턴 (TypeScript에서는 직접 지원하지 않지만 패턴으로)
class NumberLike {
  constructor(public value: number) {}

  add(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(this.value + val);
  }

  subtract(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(this.value - val);
  }

  multiply(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(this.value * val);
  }

  divide(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(Math.floor(this.value / val));
  }

  pow(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(this.value ** val);
  }

  mod(other: NumberLike | number): NumberLike {
    const val = other instanceof NumberLike ? other.value : other;
    return new NumberLike(this.value % val);
  }

  equals(other: NumberLike | number): boolean {
    const val = other instanceof NumberLike ? other.value : other;
    return this.value === val;
  }

  lessThan(other: NumberLike | number): boolean {
    const val = other instanceof NumberLike ? other.value : other;
    return this.value < val;
  }

  greaterThan(other: NumberLike | number): boolean {
    const val = other instanceof NumberLike ? other.value : other;
    return this.value > val;
  }

  toString(): string {
    return String(this.value);
  }

  valueOf(): number {
    return this.value;
  }
}

// 모든 특수 메서드를 구현한 클래스 (가능한 범위 내에서)
class EverythingClass {
  [key: string]: any;

  constructor(public value: any) {}

  get length(): number {
    return String(this.value).length;
  }

  toString(): string {
    return `EverythingClass(${this.value})`;
  }

  valueOf(): any {
    return this.value;
  }

  [Symbol.toPrimitive](hint: string): any {
    if (hint === "number") {
      return Number(this.value);
    }
    if (hint === "string") {
      return String(this.value);
    }
    return this.value;
  }

  [Symbol.iterator](): Iterator<any> {
    const str = String(this.value);
    let index = 0;
    return {
      next(): IteratorResult<string> {
        if (index < str.length) {
          return { value: str[index++], done: false };
        }
        return { value: undefined, done: true };
      }
    };
  }

  [Symbol.hasInstance](instance: any): boolean {
    return instance instanceof EverythingClass;
  }
}

// 동적 타입 생성
function createType(
  name: string,
  base: any = Object,
  props: Record<string, any> = {}
): any {
  class DynamicType extends base {
    constructor(...args: any[]) {
      super(...args);
      Object.assign(this, props);
    }
  }
  Object.defineProperty(DynamicType, "name", { value: name });
  return DynamicType;
}

const DynamicType = createType("DynamicType", Object, {
  value: 42,
  getValue() {
    return this.value;
  }
});

// 함수 시그니처 조작
function inspectAndModify<T extends (...args: any[]) => any>(
  func: T
): T {
  const wrapper = ((...args: any[]) => {
    console.log(`Function: ${func.name}`);
    console.log(`Args:`, args);
    return func(...args);
  }) as T;
  return wrapper;
}

const annotatedFunction = inspectAndModify((x: number, y: string = "default"): string => {
  return `${x}: ${y}`;
});

// 제너레이터 체이닝
function* generator1(n: number): Generator<number> {
  for (let i = 0; i < n; i++) {
    yield i * 2;
  }
}

function* generator2(gen: Generator<number>): Generator<number> {
  for (const value of gen) {
    yield value + 1;
  }
}

function* generator3(gen: Generator<number>): Generator<number> {
  for (const value of gen) {
    yield value ** 2;
  }
}

function* chainedGenerators(n: number): Generator<number> {
  const gen1 = generator1(n);
  const gen2 = generator2(gen1);
  yield* generator3(gen2);
}

// 컨텍스트 매니저 패턴 (TypeScript에서는 직접 지원하지 않지만 패턴으로)
class Context1 {
  enter(): this {
    console.log("Enter Context1");
    return this;
  }

  exit(): void {
    console.log("Exit Context1");
  }
}

class Context2 {
  enter(): this {
    console.log("Enter Context2");
    return this;
  }

  exit(): void {
    console.log("Exit Context2");
  }
}

class Context3 {
  enter(): this {
    console.log("Enter Context3");
    return this;
  }

  exit(): void {
    console.log("Exit Context3");
  }
}

function nestedContexts(): void {
  const ctx1 = new Context1().enter();
  try {
    const ctx2 = new Context2().enter();
    try {
      const ctx3 = new Context3().enter();
      try {
        // 작업 수행
      } finally {
        ctx3.exit();
      }
    } finally {
      ctx2.exit();
    }
  } finally {
    ctx1.exit();
  }
}

// 예외 체이닝
function exceptionChain(): never {
  try {
    try {
      throw new Error("Inner error");
    } catch (e) {
      throw new Error("Outer error");
    }
  } catch (e) {
    throw new Error("Final error");
  }
}

// 얕은/깊은 복사
class CopyableClass {
  constructor(public data: Record<string, any>) {}

  shallowCopy(): CopyableClass {
    return new CopyableClass({ ...this.data });
  }

  deepCopy(): CopyableClass {
    return new CopyableClass(JSON.parse(JSON.stringify(this.data)));
  }
}

// 프로퍼티 체이닝
class PropertyChain {
  private _value: number = 0;

  get value(): PropertyChain {
    return this;
  }

  set value(val: number) {
    this._value = val;
  }

  add(n: number): PropertyChain {
    this._value += n;
    return this;
  }

  multiply(n: number): PropertyChain {
    this._value *= n;
    return this;
  }

  get(): number {
    return this._value;
  }
}

// 데코레이터 팩토리의 팩토리
function decoratorFactoryFactory(baseName: string) {
  return function decoratorFactory(prefix: string = "") {
    return function decorator(
      target: any,
      propertyKey: string,
      descriptor: PropertyDescriptor
    ) {
      const original = descriptor.value;
      descriptor.value = function (...args: any[]) {
        console.log(`${prefix}${baseName}: ${propertyKey}`);
        return original.apply(this, args);
      };
    };
  };
}

const myDecoratorFactory = decoratorFactoryFactory("LOG");
const myDecorator = myDecoratorFactory("[INFO] ");

class DecoratedWithFactory {
  @myDecorator
  decoratedMethod(): string {
    return "result";
  }
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

// 프록시 패턴
class ProxyHandler implements ProxyHandler<any> {
  get(target: any, prop: string | symbol): any {
    if (typeof prop === "string" && prop.startsWith("_")) {
      return target[prop];
    }
    if (prop in target) {
      return target[prop];
    }
    return (...args: any[]) => `Called ${String(prop)} with ${JSON.stringify(args)}`;
  }

  set(target: any, prop: string | symbol, value: any): boolean {
    target[prop] = value;
    return true;
  }

  has(target: any, prop: string | symbol): boolean {
    return prop in target;
  }

  deleteProperty(target: any, prop: string | symbol): boolean {
    delete target[prop];
    return true;
  }
}

class ProxiedClass {
  [key: string]: any;

  constructor() {
    return new Proxy(this, new ProxyHandler());
  }
}

// 심볼 사용
const PRIVATE_SYMBOL = Symbol("private");
const METADATA_SYMBOL = Symbol("metadata");

class SymbolClass {
  [PRIVATE_SYMBOL]: string = "secret";
  [METADATA_SYMBOL]: Record<string, any> = {};

  getPrivate(): string {
    return this[PRIVATE_SYMBOL];
  }

  setMetadata(key: string, value: any): void {
    this[METADATA_SYMBOL][key] = value;
  }

  getMetadata(key: string): any {
    return this[METADATA_SYMBOL][key];
  }
}

// WeakMap/WeakSet 사용
const privateData = new WeakMap<object, any>();

class WeakMapClass {
  constructor() {
    privateData.set(this, { secret: "data" });
  }

  getSecret(): any {
    return privateData.get(this);
  }
}

// 리플렉션 패턴
class ReflectiveClass {
  name: string = "ReflectiveClass";
  value: number = 42;

  getPropertyNames(): string[] {
    return Object.getOwnPropertyNames(this);
  }

  getPropertyDescriptors(): Record<string, PropertyDescriptor> {
    const descriptors: Record<string, PropertyDescriptor> = {};
    for (const prop of Object.getOwnPropertyNames(this)) {
      descriptors[prop] = Object.getOwnPropertyDescriptor(this, prop)!;
    }
    return descriptors;
  }

  invokeMethod(methodName: string, ...args: any[]): any {
    const method = (this as any)[methodName];
    if (typeof method === "function") {
      return method.apply(this, args);
    }
    throw new Error(`Method ${methodName} not found`);
  }
}

