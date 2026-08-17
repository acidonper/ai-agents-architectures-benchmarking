import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createVectorStore,
  fetchConfig,
  fetchHealth,
  fetchMcpDefaults,
  fetchModels,
  fetchVectorStores,
  sendChat,
  streamChat,
  uploadRagFile,
} from './api/client'
import type {
  AppConfig,
  ChatMessage,
  ChatSettings,
  CreateVectorStoreParams,
  HealthResponse,
  McpServerConfig,
  ModelInfo,
  ToolCallSummary,
  VectorStoreInfo,
} from './api/types'
import { ChatWindow } from './components/ChatWindow'
import { Composer } from './components/Composer'
import { Sidebar } from './components/Sidebar'

function uid(): string {
  return crypto.randomUUID()
}

const DEFAULT_SETTINGS: ChatSettings = {
  model: '',
  instructions:
    'You are a helpful assistant running on Llama Stack. Use available tools when they improve the answer.',
  enableRag: false,
  vectorStoreIds: [],
  enableMcp: false,
  mcpServers: [],
  stream: true,
  temperature: 0.7,
}

export default function App() {
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<ModelInfo[]>([])
  const [vectorStores, setVectorStores] = useState<VectorStoreInfo[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const previousResponseId = useRef<string | null>(null)

  const refreshResources = async () => {
    try {
      const [healthRes, configRes, modelRes, embedRes, storeRes, mcpRes] =
        await Promise.all([
          fetchHealth(),
          fetchConfig().catch(() => null),
          fetchModels('llm').catch(() => [] as ModelInfo[]),
          fetchModels('embedding').catch(() => [] as ModelInfo[]),
          fetchVectorStores().catch(() => [] as VectorStoreInfo[]),
          fetchMcpDefaults().catch(() => ({ servers: [] as McpServerConfig[] })),
        ])
      setHealth(healthRes)
      setAppConfig(configRes)
      setModels(modelRes)
      setEmbeddingModels(embedRes)
      setVectorStores(storeRes)
      setSettings((prev) => {
        const defaultIds = configRes?.default_vector_store_ids ?? []
        const nextIds =
          prev.vectorStoreIds.length > 0
            ? prev.vectorStoreIds
            : defaultIds.filter(Boolean)
        return {
          ...prev,
          model:
            prev.model ||
            configRes?.default_model ||
            modelRes[0]?.identifier ||
            '',
          vectorStoreIds: nextIds,
          enableRag: prev.enableRag || nextIds.length > 0,
          mcpServers:
            prev.mcpServers.length > 0 ? prev.mcpServers : mcpRes.servers,
        }
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load resources')
    }
  }

  useEffect(() => {
    void refreshResources()
  }, [])

  const selectedStoreNames = useMemo(() => {
    return settings.vectorStoreIds.map((id) => {
      const match = vectorStores.find((store) => store.id === id)
      return match?.name ? `${match.name} (${id})` : id
    })
  }, [settings.vectorStoreIds, vectorStores])

  const handleSend = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || busy) return

    if (settings.enableRag && settings.vectorStoreIds.length === 0) {
      setError('Enable RAG requires at least one vector store.')
      return
    }

    setError(null)
    setBusy(true)

    const userMessage: ChatMessage = {
      id: uid(),
      role: 'user',
      content: trimmed,
    }
    const assistantId = uid()
    const pendingAssistant: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      pending: true,
      toolCalls: [],
    }

    const history = messages
      .filter((m) => !m.pending)
      .map((m) => ({ role: m.role, content: m.content }))

    setMessages((prev) => [...prev, userMessage, pendingAssistant])

    const updateAssistant = (patch: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)),
      )
    }

    try {
      if (settings.stream) {
        let accumulated = ''
        const toolCalls: ToolCallSummary[] = []

        await streamChat(
          {
            message: trimmed,
            history,
            previousResponseId: previousResponseId.current,
            settings,
          },
          {
            onDelta: (delta) => {
              accumulated += delta
              updateAssistant({ content: accumulated, pending: true })
            },
            onTool: (data) => {
              const entry = data as ToolCallSummary
              toolCalls.push(entry)
              updateAssistant({ toolCalls: [...toolCalls] })
            },
            onMessage: (data) => {
              if (data.response_id) previousResponseId.current = data.response_id
              accumulated = data.output_text || accumulated
              updateAssistant({
                content: accumulated,
                toolCalls: data.tool_calls ?? toolCalls,
                pending: false,
              })
            },
            onDone: (data) => {
              const payload = data as {
                response_id?: string
                output_text?: string
                tool_calls?: ToolCallSummary[]
              }
              if (payload.response_id) {
                previousResponseId.current = payload.response_id
              }
              updateAssistant({
                content: payload.output_text || accumulated || '(empty response)',
                toolCalls: payload.tool_calls ?? toolCalls,
                pending: false,
              })
            },
            onError: (message) => {
              setError(message)
              updateAssistant({
                content: accumulated || `Error: ${message}`,
                pending: false,
              })
            },
          },
        )
      } else {
        const response = await sendChat({
          message: trimmed,
          history,
          previousResponseId: previousResponseId.current,
          settings,
        })
        if (response.response_id) {
          previousResponseId.current = response.response_id
        }
        updateAssistant({
          content: response.output_text || '(empty response)',
          toolCalls: response.tool_calls,
          pending: false,
        })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Chat failed'
      setError(message)
      updateAssistant({ content: `Error: ${message}`, pending: false })
    } finally {
      setBusy(false)
    }
  }

  const handleClear = () => {
    setMessages([])
    previousResponseId.current = null
    setError(null)
  }

  return (
    <div className="app-shell">
      <Sidebar
        health={health}
        models={models}
        embeddingModels={embeddingModels}
        vectorStores={vectorStores}
        settings={settings}
        selectedStoreNames={selectedStoreNames}
        defaultEmbeddingModel={
          appConfig?.default_embedding_model ||
          embeddingModels[0]?.identifier ||
          ''
        }
        defaultEmbeddingDimension={
          appConfig?.default_embedding_dimension ??
          embeddingModels[0]?.embedding_dimension ??
          768
        }
        onChange={setSettings}
        onRefresh={() => void refreshResources()}
        onCreateStore={async (params: CreateVectorStoreParams) => {
          const created = await createVectorStore(params)
          setVectorStores((prev) => [created, ...prev])
          setSettings((prev) => ({
            ...prev,
            vectorStoreIds: [...prev.vectorStoreIds, created.id],
            enableRag: true,
          }))
        }}
        onUpload={async (storeId, file) => {
          await uploadRagFile(storeId, file)
          await refreshResources()
        }}
      />
      <main className="main">
        <header className="main-header">
          <div>
            <h1>Conversation</h1>
            <p>
              LLM{settings.enableRag ? ' · RAG' : ''}
              {settings.enableMcp ? ' · MCP' : ''} via Llama Stack Responses API
            </p>
          </div>
          <button className="btn" type="button" onClick={handleClear}>
            New chat
          </button>
        </header>
        {error ? <div className="error-banner">{error}</div> : null}
        <ChatWindow messages={messages} />
        <Composer busy={busy} onSend={(text) => void handleSend(text)} />
      </main>
    </div>
  )
}
