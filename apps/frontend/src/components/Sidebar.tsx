import { useEffect, useMemo, useState } from 'react'
import type {
  ChatSettings,
  CreateVectorStoreParams,
  HealthResponse,
  McpServerConfig,
  ModelInfo,
  VectorStoreInfo,
} from '../api/types'

type Props = {
  health: HealthResponse | null
  models: ModelInfo[]
  embeddingModels: ModelInfo[]
  vectorStores: VectorStoreInfo[]
  settings: ChatSettings
  selectedStoreNames: string[]
  defaultEmbeddingModel: string
  defaultEmbeddingDimension: number | null
  onChange: (settings: ChatSettings) => void
  onRefresh: () => void
  onCreateStore: (params: CreateVectorStoreParams) => Promise<void>
  onUpload: (storeId: string, file: File) => Promise<void>
}

export function Sidebar({
  health,
  models,
  embeddingModels,
  vectorStores,
  settings,
  selectedStoreNames,
  defaultEmbeddingModel,
  defaultEmbeddingDimension,
  onChange,
  onRefresh,
  onCreateStore,
  onUpload,
}: Props) {
  const [storeName, setStoreName] = useState('knowledge-base')
  const [embeddingModel, setEmbeddingModel] = useState(defaultEmbeddingModel)
  const [embeddingDimension, setEmbeddingDimension] = useState(
    defaultEmbeddingDimension != null ? String(defaultEmbeddingDimension) : '768',
  )
  const [uploadStoreId, setUploadStoreId] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!embeddingModel && defaultEmbeddingModel) {
      setEmbeddingModel(defaultEmbeddingModel)
    }
  }, [defaultEmbeddingModel, embeddingModel])

  useEffect(() => {
    if (
      (!embeddingDimension || embeddingDimension === '768') &&
      defaultEmbeddingDimension != null
    ) {
      setEmbeddingDimension(String(defaultEmbeddingDimension))
    }
  }, [defaultEmbeddingDimension, embeddingDimension])

  const llmModels = useMemo(
    () => models.filter((m) => (m.model_type || 'llm').toLowerCase() === 'llm'),
    [models],
  )

  const resolvedEmbedding = embeddingModel || defaultEmbeddingModel || embeddingModels[0]?.identifier || ''

  const patch = (partial: Partial<ChatSettings>) => {
    onChange({ ...settings, ...partial })
  }

  const updateMcp = (index: number, partial: Partial<McpServerConfig>) => {
    const next = settings.mcpServers.map((server, i) =>
      i === index ? { ...server, ...partial } : server,
    )
    patch({ mcpServers: next })
  }

  const onEmbeddingChange = (id: string) => {
    setEmbeddingModel(id)
    const match = embeddingModels.find((m) => m.identifier === id)
    if (match?.embedding_dimension) {
      setEmbeddingDimension(String(match.embedding_dimension))
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          Stack<span>Chat</span>
        </div>
        <p className="brand-sub">
          Chat against Llama Stack with LLM, RAG, and MCP in one workflow.
        </p>
        <div className="status-pill">
          <span
            className={`status-dot ${health?.llama_stack === 'reachable' ? 'ok' : ''}`}
          />
          {health
            ? health.llama_stack === 'reachable'
              ? 'Llama Stack connected'
              : 'Llama Stack unreachable'
            : 'Checking…'}
        </div>
      </div>

      <section className="panel">
        <h2>Model</h2>
        <p className="panel-desc">Inference model registered in Llama Stack.</p>
        <div className="field">
          <label htmlFor="model">Model id</label>
          <select
            id="model"
            value={settings.model}
            onChange={(event) => patch({ model: event.target.value })}
          >
            <option value="">Select a model</option>
            {llmModels.map((model) => (
              <option key={model.identifier} value={model.identifier}>
                {model.identifier}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="temperature">Temperature ({settings.temperature})</label>
          <input
            id="temperature"
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={settings.temperature}
            onChange={(event) =>
              patch({ temperature: Number(event.target.value) })
            }
          />
        </div>
        <div className="field">
          <label htmlFor="instructions">Instructions</label>
          <textarea
            id="instructions"
            value={settings.instructions}
            onChange={(event) => patch({ instructions: event.target.value })}
          />
        </div>
        <div className="toggle-row">
          <span>Stream responses</span>
          <button
            type="button"
            className={`toggle ${settings.stream ? 'on' : ''}`}
            aria-pressed={settings.stream}
            onClick={() => patch({ stream: !settings.stream })}
          />
        </div>
        <button className="btn" type="button" onClick={onRefresh}>
          Refresh models & stores
        </button>
      </section>

      <section className="panel">
        <h2>RAG</h2>
        <p className="panel-desc">
          Create a vector store with an embedding model, upload docs, then enable
          file_search retrieval.
        </p>
        <div className="toggle-row">
          <span>Enable RAG</span>
          <button
            type="button"
            className={`toggle ${settings.enableRag ? 'on' : ''}`}
            aria-pressed={settings.enableRag}
            onClick={() => patch({ enableRag: !settings.enableRag })}
          />
        </div>
        {settings.enableRag && settings.vectorStoreIds.length === 0 ? (
          <p className="panel-desc" style={{ color: 'var(--danger, #c00)' }}>
            Select or create a vector store before chatting with RAG enabled.
          </p>
        ) : null}
        <div className="field">
          <label htmlFor="vector-store">Vector stores</label>
          <select
            id="vector-store"
            value=""
            onChange={(event) => {
              const id = event.target.value
              if (!id || settings.vectorStoreIds.includes(id)) return
              patch({
                vectorStoreIds: [...settings.vectorStoreIds, id],
                enableRag: true,
              })
            }}
          >
            <option value="">Add store…</option>
            {vectorStores.map((store) => (
              <option key={store.id} value={store.id}>
                {store.name || store.id}
                {store.embedding_model ? ` · ${store.embedding_model}` : ''}
              </option>
            ))}
          </select>
        </div>
        <div className="chip-list">
          {selectedStoreNames.map((name, index) => (
            <span className="chip" key={settings.vectorStoreIds[index]}>
              {name}
              <button
                type="button"
                aria-label="Remove store"
                onClick={() =>
                  patch({
                    vectorStoreIds: settings.vectorStoreIds.filter(
                      (_, i) => i !== index,
                    ),
                  })
                }
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="field">
          <label htmlFor="embedding-model">Embedding model</label>
          <select
            id="embedding-model"
            value={resolvedEmbedding}
            onChange={(event) => onEmbeddingChange(event.target.value)}
          >
            <option value="">Select embedding model…</option>
            {embeddingModels.map((model) => (
              <option key={model.identifier} value={model.identifier}>
                {model.identifier}
                {model.embedding_dimension ? ` (${model.embedding_dimension}d)` : ''}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="embedding-dimension">Embedding dimension</label>
          <input
            id="embedding-dimension"
            type="number"
            min={1}
            value={embeddingDimension}
            onChange={(event) => setEmbeddingDimension(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="new-store">Create store</label>
          <input
            id="new-store"
            value={storeName}
            onChange={(event) => setStoreName(event.target.value)}
          />
        </div>
        <div className="btn-row">
          <button
            className="btn"
            type="button"
            disabled={busy || !storeName.trim() || !resolvedEmbedding}
            onClick={async () => {
              setBusy(true)
              try {
                const dim = Number(embeddingDimension)
                await onCreateStore({
                  name: storeName.trim(),
                  embedding_model: resolvedEmbedding,
                  embedding_dimension: Number.isFinite(dim) && dim > 0 ? dim : undefined,
                })
              } finally {
                setBusy(false)
              }
            }}
          >
            Create
          </button>
        </div>
        <div className="field">
          <label htmlFor="upload-store">Upload document</label>
          <select
            id="upload-store"
            value={uploadStoreId || settings.vectorStoreIds[0] || ''}
            onChange={(event) => setUploadStoreId(event.target.value)}
          >
            <option value="">Select store…</option>
            {vectorStores.map((store) => (
              <option key={store.id} value={store.id}>
                {store.name || store.id}
              </option>
            ))}
          </select>
          <input
            className="file-input"
            type="file"
            onChange={async (event) => {
              const file = event.target.files?.[0]
              const storeId = uploadStoreId || settings.vectorStoreIds[0]
              if (!file || !storeId) return
              setBusy(true)
              try {
                await onUpload(storeId, file)
              } finally {
                setBusy(false)
                event.target.value = ''
              }
            }}
          />
        </div>
      </section>

      <section className="panel">
        <h2>MCP</h2>
        <p className="panel-desc">
          Connect Model Context Protocol servers for tool calling.
        </p>
        <div className="toggle-row">
          <span>Enable MCP</span>
          <button
            type="button"
            className={`toggle ${settings.enableMcp ? 'on' : ''}`}
            aria-pressed={settings.enableMcp}
            onClick={() => patch({ enableMcp: !settings.enableMcp })}
          />
        </div>
        <div className="mcp-list">
          {settings.mcpServers.map((server, index) => (
            <div className="mcp-item" key={`mcp-${index}`}>
              <input
                placeholder="server_label"
                value={server.server_label}
                onChange={(event) =>
                  updateMcp(index, { server_label: event.target.value })
                }
              />
              <input
                placeholder="server_url (e.g. http://localhost:3000/sse)"
                value={server.server_url}
                onChange={(event) =>
                  updateMcp(index, { server_url: event.target.value })
                }
              />
              <button
                className="btn"
                type="button"
                onClick={() =>
                  patch({
                    mcpServers: settings.mcpServers.filter((_, i) => i !== index),
                  })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
        <button
          className="btn"
          type="button"
          onClick={() =>
            patch({
              mcpServers: [
                ...settings.mcpServers,
                { server_label: 'mcp-server', server_url: 'http://localhost:3000/sse' },
              ],
              enableMcp: true,
            })
          }
        >
          Add MCP server
        </button>
      </section>
    </aside>
  )
}
