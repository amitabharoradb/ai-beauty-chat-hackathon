// src/uphora/src/uphora/ui/components/CustomerSelector.tsx
import { Suspense } from "react";
import { useListCustomersSuspense } from "@/lib/api";
import selector from "@/lib/selector";
import "../styles/uphora.css";

interface Props {
  selectedId: string;
  onSelect: (id: string) => void;
}

function CustomerDropdown({ selectedId, onSelect }: Props) {
  const { data: customers } = useListCustomersSuspense(selector());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="uphora-label">Browsing as</div>
      <select
        className="uphora-select"
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        style={{ minWidth: 220 }}
      >
        <option value="" disabled>Select customer…</option>
        {customers.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name} — {c.skin_type} skin
          </option>
        ))}
      </select>
    </div>
  );
}

export function CustomerSelector({ selectedId, onSelect }: Props) {
  return (
    <Suspense fallback={<div className="uphora-label">Loading customers…</div>}>
      <CustomerDropdown selectedId={selectedId} onSelect={onSelect} />
    </Suspense>
  );
}
