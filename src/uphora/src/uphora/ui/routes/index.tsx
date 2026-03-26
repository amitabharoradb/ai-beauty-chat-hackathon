// src/uphora/src/uphora/ui/routes/index.tsx
import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { CustomerSelector } from "../components/CustomerSelector";
import { ChatWindow } from "../components/ChatWindow";
import "../styles/uphora.css";

export const Route = createFileRoute("/")({
  component: IndexPage,
});

function IndexPage() {
  const [customerId, setCustomerId] = useState("");

  return (
    <div style={{ minHeight: "100vh", background: "#faf8f5" }}>
      <header className="uphora-header">
        <span className="uphora-logo">UPHORA</span>
        <CustomerSelector selectedId={customerId} onSelect={setCustomerId} />
      </header>
      <ChatWindow customerId={customerId} />
    </div>
  );
}
