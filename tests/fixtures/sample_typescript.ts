/**
 * Sample TypeScript file for testing parser
 */

export interface User {
  name: string;
  age: number;
  email?: string;
}

export class UserService {
  private users: User[] = [];

  constructor(initialUsers?: User[]) {
    if (initialUsers) {
      this.users = initialUsers;
    }
  }

  public addUser(user: User): void {
    this.users.push(user);
  }

  public getUser(name: string): User | undefined {
    return this.users.find(u => u.name === name);
  }

  static createDefault(): UserService {
    return new UserService([]);
  }
}

export abstract class BaseRepository<T> {
  protected items: T[] = [];

  abstract save(item: T): Promise<void>;
  abstract findById(id: string): Promise<T | undefined>;
}

export class UserRepository extends BaseRepository<User> implements Repository {
  async save(user: User): Promise<void> {
    this.items.push(user);
  }

  async findById(id: string): Promise<User | undefined> {
    return this.items.find(u => u.name === id);
  }
}

export function calculateAge(birthYear: number): number {
  return new Date().getFullYear() - birthYear;
}

export async function fetchUsers(): Promise<User[]> {
  // Implementation here
  return [];
}

type UserRole = "admin" | "user" | "guest";

export type UserWithRole = User & { role: UserRole };
