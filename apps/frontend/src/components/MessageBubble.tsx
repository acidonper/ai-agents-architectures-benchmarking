import type { ChatMessage } from '../api/types'

type Props = {
  message: ChatMessage
}

export function MessageBubble({ message }: Props) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-meta">
        <span>{message.role}</span>
        {message.pending ? <span>thinking…</span> : null}
      </div>
      <div className="bubble">
        {message.content || (message.pending ? '…' : '')}
      </div>
      {message.toolCalls && message.toolCalls.length > 0 ? (
        <div className="tool-trace">
          {message.toolCalls.map((tool, index) => (
            <span className="tool-chip" key={`${tool.type}-${index}`}>
              {tool.type}
              {tool.name ? ` · ${tool.name}` : ''}
              {tool.status ? ` · ${tool.status}` : ''}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  )
}
