export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://arkon-be.azurewebsites.net/";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("arkon_token");
}

export function setToken(t: string) {
  localStorage.setItem("arkon_token", t);
}

export function clearToken() {
  localStorage.removeItem("arkon_token");
}

export async function api<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || res.statusText);
  }
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}
