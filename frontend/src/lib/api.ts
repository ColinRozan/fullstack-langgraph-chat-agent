const API_BASE = import.meta.env.DEV ? "" : "http://localhost:8123";

export interface ThreadMeta {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface ThreadListResponse {
  threads: ThreadMeta[];
}

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem("api_token");
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  const apiKey = localStorage.getItem("api_key");
  if (apiKey) {
    return { "X-API-Key": apiKey };
  }
  return {};
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "Unknown error");
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

export async function listThreads(): Promise<ThreadMeta[]> {
  const data = await fetchJson<ThreadListResponse>(`${API_BASE}/api/threads`);
  return data.threads;
}

export async function getThread(threadId: string): Promise<ThreadMeta | null> {
  try {
    return await fetchJson<ThreadMeta>(`${API_BASE}/api/threads/${threadId}`);
  } catch {
    return null;
  }
}

export async function updateThread(
  threadId: string,
  title: string
): Promise<void> {
  await fetchJson(`${API_BASE}/api/threads/${threadId}`, {
    method: "PUT",
    body: JSON.stringify({ title }),
  });
}

export async function deleteThread(threadId: string): Promise<void> {
  await fetchJson(`${API_BASE}/api/threads/${threadId}`, {
    method: "DELETE",
  });
}
