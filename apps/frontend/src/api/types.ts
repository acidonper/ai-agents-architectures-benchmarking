export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  toolCalls?: ToolCallSummary[]
  pending?: boolean
}

export interface ToolCallSummary {
  type: string
  name?: string | null
  status?: string | null
  detail?: Record<string, unknown> | null
}

export interface McpServerConfig {
  server_label: string
  server_url: string
  allowed_tools?: string[] | null
}

export interface ModelInfo {
  identifier: string
  model_type?: string | null
  provider_id?: string | null
  embedding_dimension?: number | null
  metadata?: Record<string, unknown>
}

export interface VectorStoreInfo {
  id: string
  name?: string | null
  status?: string | null
  embedding_model?: string | null
  embedding_dimension?: number | null
  provider_id?: string | null
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  llama_stack: 'reachable' | 'unreachable'
  detail?: string | null
}

export interface AppConfig {
  llama_stack_base_url: string
  default_model?: string | null
  default_embedding_model?: string | null
  default_embedding_dimension?: number | null
  default_vector_store_provider?: string | null
  default_vector_store_ids?: string[]
  default_mcp_servers?: McpServerConfig[]
}

export interface CreateVectorStoreParams {
  name: string
  embedding_model?: string
  embedding_dimension?: number
  provider_id?: string
}

export interface ChatSettings {
  model: string
  instructions: string
  enableRag: boolean
  vectorStoreIds: string[]
  enableMcp: boolean
  mcpServers: McpServerConfig[]
  stream: boolean
  temperature: number
}

export interface ChatResponse {
  response_id?: string | null
  model?: string | null
  output_text: string
  tool_calls?: ToolCallSummary[]
}
