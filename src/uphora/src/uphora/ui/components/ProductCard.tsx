// src/uphora/src/uphora/ui/components/ProductCard.tsx
interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  key_ingredients: string[];
  benefits: string[];
  tags: string[];
}

interface Props {
  product: Product;
}

export function ProductCard({ product }: Props) {
  return (
    <div
      style={{
        border: "1px solid #e8ddd0",
        borderRadius: 4,
        padding: "12px 14px",
        background: "#ffffff",
        width: 180,
        flexShrink: 0,
      }}
    >
      <div
        style={{
          background: "#f5f0eb",
          height: 80,
          borderRadius: 2,
          marginBottom: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: "1.5rem" }}>✨</span>
      </div>

      <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#2d2d2d", marginBottom: 2 }}>
        {product.name}
      </div>
      <div style={{ fontSize: "0.7rem", color: "#c9a96e", marginBottom: 6 }}>
        ${product.price.toFixed(2)}
      </div>
      <div style={{ fontSize: "0.65rem", color: "#8a8075", marginBottom: 8, lineHeight: 1.4 }}>
        {product.description.length > 60
          ? product.description.slice(0, 60) + "…"
          : product.description}
      </div>

      <a
        href="#"
        style={{
          display: "block",
          textAlign: "center",
          fontSize: "0.6rem",
          letterSpacing: "1.5px",
          textTransform: "uppercase",
          color: "#2d2d2d",
          textDecoration: "none",
          border: "1px solid #2d2d2d",
          padding: "5px 0",
        }}
      >
        View on Uphora
      </a>
    </div>
  );
}
