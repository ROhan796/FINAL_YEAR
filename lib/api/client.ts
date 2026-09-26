/**
 * Centralized HTTP client for the CKGAI FastAPI backend.
 *
 * All API modules import from this file so request logic, auth headers,
 * error handling, and base URL configuration live in one place.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Token storage (in-memory + localStorage persistence) ────────────────

let accessToken: string | null = null;
let refreshToken: string | null = null;

if (typeof window !== 'undefined') {
  accessToken = localStorage.getItem('ckgai_access_token');
  refreshToken = localStorage.getItem('ckgai_refresh_token');
}

export function setTokens(access: string, refresh: string) {
  accessToken = access;
  refreshToken = refresh;
  if (typeof window !== 'undefined') {
    localStorage.setItem('ckgai_access_token', access);
    localStorage.setItem('ckgai_refresh_token', refresh);
  }
}

export function clearTokens() {
  accessToken = null;
  refreshToken = null;
  if (typeof window !== 'undefined') {
    localStorage.removeItem('ckgai_access_token');
    localStorage.removeItem('ckgai_refresh_token');
  }
}

export function getAccessToken(): string | null {
  return accessToken;
}

export function getRefreshToken(): string | null {
  return refreshToken;
}

// ── Error class ─────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

// ── Core request function ───────────────────────────────────────────────

interface RequestOptions extends Omit<RequestInit, 'method' | 'body'> {
  params?: Record<string, string | number | boolean | undefined | null>;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const { params, ...fetchOptions } = options;

  // Build URL with query params
  let url = `${API_BASE}${path}`;
  if (params) {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value != null && value !== '') {
        searchParams.set(key, String(value));
      }
    }
    const qs = searchParams.toString();
    if (qs) url += `?${qs}`;
  }

  // Auth header
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(fetchOptions.headers as Record<string, string> || {}),
  };
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  const res = await fetch(url, {
    method,
    headers,
    body: body != null ? JSON.stringify(body) : undefined,
    ...fetchOptions,
  });

  // Handle 401 — try token refresh once
  if (res.status === 401 && refreshToken) {
    try {
      const refreshRes = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken }),
      });
      if (refreshRes.ok) {
        const data = await refreshRes.json();
        setTokens(data.accessToken, refreshToken);
        // Retry original request
        headers['Authorization'] = `Bearer ${data.accessToken}`;
        const retryRes = await fetch(url, { method, headers, body: body != null ? JSON.stringify(body) : undefined });
        if (!retryRes.ok) {
          const errBody = await retryRes.json().catch(() => null);
          throw new ApiError(retryRes.status, `Request failed: ${retryRes.status}`, errBody);
        }
        return retryRes.json();
      }
    } catch {
      clearTokens();
    }
  }

  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    const msg = errBody?.detail || `Request failed: ${res.status}`;
    throw new ApiError(res.status, typeof msg === 'string' ? msg : JSON.stringify(msg), errBody);
  }

  // 204 No Content
  if (res.status === 204) return undefined as T;

  return res.json();
}

// ── HTTP verbs ──────────────────────────────────────────────────────────

export const api = {
  get: <T>(path: string, opts?: RequestOptions) => request<T>('GET', path, undefined, opts),
  post: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('POST', path, body, opts),
  put: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PUT', path, body, opts),
  patch: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PATCH', path, body, opts),
  delete: <T>(path: string, opts?: RequestOptions) => request<T>('DELETE', path, undefined, opts),
};

export { API_BASE };
