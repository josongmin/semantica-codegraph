/**
 * 유틸리티 함수들
 */

export interface Item {
  price: number;
  category?: string;
  [key: string]: any;
}

export function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + (item.price || 0), 0);
}

export function filterByPrice(
  items: Item[],
  minPrice: number,
  maxPrice: number
): Item[] {
  return items.filter(
    item => item.price >= minPrice && item.price <= maxPrice
  );
}

export function groupByCategory(items: Item[]): Record<string, Item[]> {
  const grouped: Record<string, Item[]> = {};
  for (const item of items) {
    const category = item.category || "uncategorized";
    if (!grouped[category]) {
      grouped[category] = [];
    }
    grouped[category].push(item);
  }
  return grouped;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false;

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

export async function fetchData(url: string): Promise<any> {
  // 실제 구현에서는 HTTP 요청을 수행함
  return { url, data: [] };
}

export async function fetchMultiple(urls: string[]): Promise<any[]> {
  const results = [];
  for (const url of urls) {
    const result = await fetchData(url);
    results.push(result);
  }
  return results;
}

export function serializeToJson(obj: any): string {
  return JSON.stringify(obj);
}

export function deserializeFromJson<T>(jsonStr: string): T {
  return JSON.parse(jsonStr) as T;
}

export class Cache<T> {
  private cache: Map<string, T> = new Map();
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  get(key: string): T | undefined {
    return this.cache.get(key);
  }

  set(key: string, value: T): void {
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }
}

export function findMax<T>(
  items: T[],
  key?: (item: T) => any
): T | undefined {
  if (items.length === 0) {
    return undefined;
  }
  if (key) {
    return items.reduce((max, item) =>
      key(item) > key(max) ? item : max
    );
  }
  return items.reduce((max, item) => (item > max ? item : max));
}

export function findMin<T>(
  items: T[],
  key?: (item: T) => any
): T | undefined {
  if (items.length === 0) {
    return undefined;
  }
  if (key) {
    return items.reduce((min, item) =>
      key(item) < key(min) ? item : min
    );
  }
  return items.reduce((min, item) => (item < min ? item : min));
}

export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array));
}

export function flatten<T>(arrays: T[][]): T[] {
  return arrays.reduce((acc, arr) => acc.concat(arr), []);
}

export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj)) as T;
}

export function formatCurrency(amount: number, currency: string = "KRW"): string {
  return new Intl.NumberFormat("ko-KR", {
    style: "currency",
    currency
  }).format(amount);
}

export function formatDate(date: Date, format: string = "YYYY-MM-DD"): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");

  return format
    .replace("YYYY", String(year))
    .replace("MM", month)
    .replace("DD", day);
}

export type AsyncFunction<T extends any[], R> = (...args: T) => Promise<R>;

export async function retry<T extends any[], R>(
  fn: AsyncFunction<T, R>,
  maxAttempts: number = 3,
  delay: number = 1000
): Promise<R> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (attempt < maxAttempts - 1) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error("재시도 실패");
}
