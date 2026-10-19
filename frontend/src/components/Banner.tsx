import type { ReactNode } from "react";

import { AlertIcon, CrossIcon, InfoIcon } from "./icons";

export function Banner({
  tone,
  title,
  children,
  action,
}: {
  tone: "info" | "warning" | "critical";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const icon =
    tone === "critical" ? <CrossIcon size={16} /> : tone === "warning" ? <AlertIcon size={16} /> : <InfoIcon size={16} />;
  return (
    <div className={`banner banner-${tone}`} role={tone === "critical" ? "alert" : "status"}>
      <span className="banner-icon">{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p className="banner-title">{title}</p>
        {children ? <div className="small secondary">{children}</div> : null}
      </div>
      {action}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="small secondary">
      <span className="spinner" /> {label}…
    </div>
  );
}
