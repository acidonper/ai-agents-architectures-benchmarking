import { useEffect, useRef, useState } from 'react'

type Props = {
  busy: boolean
  onSend: (text: string) => void
}

export function Composer({ busy, onSend }: Props) {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [text])

  const submit = () => {
    if (!text.trim() || busy) return
    onSend(text)
    setText('')
  }

  return (
    <div className="composer">
      <div className="composer-inner">
        <textarea
          ref={textareaRef}
          value={text}
          placeholder="Message Llama Stack…"
          rows={2}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          disabled={busy}
        />
        <button
          className="btn btn-primary"
          type="button"
          disabled={busy || !text.trim()}
          onClick={submit}
        >
          {busy ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
