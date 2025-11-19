/**
 * 설정 관련 코드
 */

export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
}

export interface APIConfig {
  host: string;
  port: number;
  debug: boolean;
  corsEnabled: boolean;
  allowedOrigins: string[];
}

export interface CacheConfig {
  enabled: boolean;
  backend: string;
  host: string;
  port: number;
  ttl: number;
}

export interface AppConfig {
  appName: string;
  version: string;
  environment: string;
  database: DatabaseConfig;
  api: APIConfig;
  cache: CacheConfig;
}

export class DatabaseConfigBuilder {
  private config: Partial<DatabaseConfig> = {
    host: "localhost",
    port: 5432,
    database: "mydb",
    username: "user",
    password: "password"
  };

  host(host: string): this {
    this.config.host = host;
    return this;
  }

  port(port: number): this {
    this.config.port = port;
    return this;
  }

  database(database: string): this {
    this.config.database = database;
    return this;
  }

  username(username: string): this {
    this.config.username = username;
    return this;
  }

  password(password: string): this {
    this.config.password = password;
    return this;
  }

  build(): DatabaseConfig {
    return this.config as DatabaseConfig;
  }
}

export class AppConfigBuilder {
  private config: Partial<AppConfig> = {
    appName: "MyApp",
    version: "1.0.0",
    environment: "development"
  };

  appName(name: string): this {
    this.config.appName = name;
    return this;
  }

  version(version: string): this {
    this.config.version = version;
    return this;
  }

  environment(env: string): this {
    this.config.environment = env;
    return this;
  }

  database(config: DatabaseConfig | DatabaseConfigBuilder): this {
    this.config.database =
      config instanceof DatabaseConfigBuilder
        ? config.build()
        : config;
    return this;
  }

  api(config: Partial<APIConfig>): this {
    this.config.api = {
      host: config.host || "0.0.0.0",
      port: config.port || 8000,
      debug: config.debug || false,
      corsEnabled: config.corsEnabled !== undefined ? config.corsEnabled : true,
      allowedOrigins: config.allowedOrigins || ["*"]
    };
    return this;
  }

  cache(config: Partial<CacheConfig>): this {
    this.config.cache = {
      enabled: config.enabled !== undefined ? config.enabled : true,
      backend: config.backend || "redis",
      host: config.host || "localhost",
      port: config.port || 6379,
      ttl: config.ttl || 3600
    };
    return this;
  }

  build(): AppConfig {
    return this.config as AppConfig;
  }
}

export function createDefaultConfig(): AppConfig {
  return new AppConfigBuilder()
    .database(
      new DatabaseConfigBuilder()
        .host("localhost")
        .port(5432)
        .database("mydb")
    )
    .api({
      host: "0.0.0.0",
      port: 8000,
      debug: false
    })
    .cache({
      enabled: true,
      backend: "redis"
    })
    .build();
}

export function createConfigFromEnv(): AppConfig {
  const dbConfig = new DatabaseConfigBuilder()
    .host(process.env.DB_HOST || "localhost")
    .port(parseInt(process.env.DB_PORT || "5432"))
    .database(process.env.DB_NAME || "mydb")
    .username(process.env.DB_USER || "user")
    .password(process.env.DB_PASSWORD || "password")
    .build();

  return new AppConfigBuilder()
    .appName(process.env.APP_NAME || "MyApp")
    .version(process.env.APP_VERSION || "1.0.0")
    .environment(process.env.ENVIRONMENT || "development")
    .database(dbConfig)
    .api({
      host: process.env.API_HOST || "0.0.0.0",
      port: parseInt(process.env.API_PORT || "8000"),
      debug: process.env.DEBUG === "true"
    })
    .build();
}

export class ConfigManager {
  private config: AppConfig;

  constructor(config?: AppConfig) {
    this.config = config || createDefaultConfig();
  }

  getConfig(): AppConfig {
    return this.config;
  }

  update(updates: Partial<AppConfig>): void {
    this.config = {
      ...this.config,
      ...updates,
      database: { ...this.config.database, ...updates.database },
      api: { ...this.config.api, ...updates.api },
      cache: { ...this.config.cache, ...updates.cache }
    };
  }

  get<K extends keyof AppConfig>(key: K): AppConfig[K] {
    return this.config[key];
  }

  getNested(key: string): any {
    const keys = key.split(".");
    let value: any = this.config;
    for (const k of keys) {
      if (value && typeof value === "object" && k in value) {
        value = value[k as keyof typeof value];
      } else {
        return undefined;
      }
    }
    return value;
  }

  reload(): void {
    this.config = createConfigFromEnv();
  }

  toJSON(): string {
    return JSON.stringify(this.config, null, 2);
  }
}

export const configManager = new ConfigManager();
