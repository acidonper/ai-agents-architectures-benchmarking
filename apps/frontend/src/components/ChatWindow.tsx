import type { ChatMessage } from '../api/types'
import { MessageBubble } from './MessageBubble'

type Props = {
  messages: ChatMessage[]
}

export function ChatWindow({ messages }: Props) {
  if (messages.length === 0) {
    return (
      <div className="messages">
        <div className="empty-state">
          <h2>Ask anything</h2>
          <p>
            Enable RAG to ground answers in your vector stores, or MCP to let
            the model call external tools through Llama Stack.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="messages">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </div>
  )
}
