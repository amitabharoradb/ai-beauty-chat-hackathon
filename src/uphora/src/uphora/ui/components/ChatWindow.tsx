// src/uphora/src/uphora/ui/components/ChatWindow.tsx
import { useRef, useEffect, useState, KeyboardEvent } from "react";
import { useChat } from "../hooks/useChat";
import { ProductCard } from "./ProductCard";
import "../styles/uphora.css";

interface Props {
  customerId: string;
}

export function ChatWindow({ customerId }: Props) {
  const { messages, products, isStreaming, sendMessage } = useChat(customerId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    sendMessage(trimmed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 64px)" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px", display: "flex", flexDirection: "column", gap: 16 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: "center", marginTop: 80, color: "#8a8075" }}>
            <div style={{ fontSize: "2rem", marginBottom: 12 }}>✨</div>
            <div className="uphora-label" style={{ display: "block", marginBottom: 8 }}>Uphora Beauty Advisor</div>
            <div style={{ fontSize: "0.9rem" }}>
              {customerId
                ? "Ask me anything about your skincare, makeup, or haircare routine."
                : "Select a customer above to begin."}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}>
            {msg.role === "assistant" && (
              <span className="uphora-label" style={{ marginBottom: 4, marginLeft: 4 }}>AVA</span>
            )}
            <div className={msg.role === "user" ? "bubble-user" : "bubble-bot"}>
              {msg.content}
              {isStreaming && i === messages.length - 1 && msg.role === "assistant" && (
                <span style={{ color: "#c9a96e", marginLeft: 4 }}>▌</span>
              )}
            </div>
          </div>
        ))}

        {products.length > 0 && (
          <div style={{ display: "flex", gap: 12, overflowX: "auto", paddingBottom: 4 }}>
            {products.map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div
        style={{
          borderTop: "1px solid #e8ddd0",
          padding: "16px 32px",
          display: "flex",
          gap: 8,
          background: "#faf8f5",
        }}
      >
        <input
          className="uphora-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={customerId ? "Ask about your skin, products, or routine…" : "Select a customer first"}
          disabled={!customerId || isStreaming}
        />
        <button
          className="uphora-btn uphora-btn-gold"
          onClick={handleSend}
          disabled={!customerId || isStreaming || !input.trim()}
        >
          {isStreaming ? "…" : "SEND"}
        </button>
      </div>
    </div>
  );
}
