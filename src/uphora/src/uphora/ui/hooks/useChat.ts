// src/uphora/src/uphora/ui/hooks/useChat.ts
import { useCallback, useState } from "react";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  key_ingredients: string[];
  benefits: string[];
  tags: string[];
}

export function useChat(customerId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!customerId || isStreaming) return;

      const userMsg: Message = { role: "user", content };
      const updatedHistory = [...messages, userMsg];
      setMessages(updatedHistory);
      setIsStreaming(true);
      setProducts([]);

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            customer_id: customerId,
            message: content,
            history: messages.map((m) => ({ role: m.role, content: m.content })),
          }),
        });

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          for (const line of text.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") break;

            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                setMessages((prev) => [
                  ...prev.slice(0, -1),
                  { role: "assistant", content: prev[prev.length - 1].content + parsed.text },
                ]);
              }
              if (parsed.products) {
                setProducts(parsed.products);
              }
            } catch {
              // ignore parse errors on partial chunks
            }
          }
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [customerId, messages, isStreaming]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setProducts([]);
  }, []);

  return { messages, products, isStreaming, sendMessage, clearChat };
}
