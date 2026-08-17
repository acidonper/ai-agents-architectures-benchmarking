import type {
  AppConfig,
  ChatResponse,
  ChatSettings,
  CreateVectorStoreParams,
  HealthResponse,
  McpServerConfig,
  ModelInfo,
  VectorStoreInfo,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData
        ? {}
        : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? JSON.stringify(payload)
    } catch {
      // keep statusText
    }
    throw new Error(detail)
  }

  return response.json() as Promise<T>
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health')
}

export function fetchConfig(): Promise<AppConfig> {
  return request<AppConfig>('/api/config')
}

export function fetchModels(
  modelType: 'llm' | 'embedding' | 'all' = 'all',
): Promise<ModelInfo[]> {
  const query = modelType === 'all' ? '' : `?model_type=${modelType}`
  return request<ModelInfo[]>(`/api/models${query}`)
}

export function fetchVectorStores(): Promise<VectorStoreInfo[]> {
  return request<VectorStoreInfo[]>('/api/rag/vector-stores')
}

export function createVectorStore(
  params: CreateVectorStoreParams,
): Promise<VectorStoreInfo> {
  return request<VectorStoreInfo>('/api/rag/vector-stores', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export async function uploadRagFile(
  vectorStoreId: string,
  file: File,
): Promise<unknown> {
  const form = new FormData()
  form.append('file', file)
  return request(`/api/rag/vector-stores/${vectorStoreId}/files`, {
    method: 'POST',
    body: form,
  })
}

export function fetchMcpDefaults(): Promise<{ servers: McpServerConfig[] }> {
  return request<{ servers: McpServerConfig[] }>('/api/mcp/defaults')
}

export function sendChat(params: {
  message: string
  history: { role: string; content: string }[]
  previousResponseId?: string | null
  settings: ChatSettings
}): Promise<ChatResponse> {
  const { message, history, previousResponseId, settings } = params
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      history,
      previous_response_id: previousResponseId ?? undefined,
      model: settings.model || undefined,
      instructions: settings.instructions || undefined,
      enable_rag: settings.enableRag,
      vector_store_ids: settings.vectorStoreIds,
      enable_mcp: settings.enableMcp,
      mcp_servers: settings.mcpServers,
      stream: false,
      temperature: settings.temperature,
    }),
  })
}

export type StreamHandlers = {
  onDelta?: (text: string) => void
  onTool?: (data: unknown) => void
  onMessage?: (data: ChatResponse) => void
  onDone?: (data: unknown) => void
  onError?: (message: string) => void
}

export async function streamChat(
  params: {
    message: string
    history: { role: string; content: string }[]
    previousResponseId?: string | null
    settings: ChatSettings
  },
  handlers: StreamHandlers,
): Promise<void> {
  const { message, history, previousResponseId, settings } = params
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      history,
      previous_response_id: previousResponseId ?? undefined,
      model: settings.model || undefined,
      instructions: settings.instructions || undefined,
      enable_rag: settings.enableRag,
      vector_store_ids: settings.vectorStoreIds,
      enable_mcp: settings.enableMcp,
      mcp_servers: settings.mcpServers,
      stream: true,
      temperature: settings.temperature,
    }),
  })

  if (!response.ok || !response.body) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? detail
    } catch {
      // ignore
    }
    handlers.onError?.(detail)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''

    for (const chunk of chunks) {
      const lines = chunk.split('\n')
      let event = 'message'
      let data = ''
      for (const line of lines) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (!data) continue

      try {
        const parsed = JSON.parse(data)
        if (event === 'delta') handlers.onDelta?.(parsed.text ?? '')
        else if (event === 'tool') handlers.onTool?.(parsed)
        else if (event === 'message') handlers.onMessage?.(parsed)
        else if (event === 'done') handlers.onDone?.(parsed)
        else if (event === 'error') handlers.onError?.(parsed.detail ?? 'Stream error')
      } catch {
        // ignore malformed SSE chunks
      }
    }
  }
}
