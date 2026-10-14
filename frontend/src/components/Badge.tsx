import type { ReactNode } from "react";

import type { AnswerStatus, ClaimStatus, DocumentStatus, Grounding, SupportLabel, Verdict } from "../api/types";
import { AlertIcon, CheckIcon, ClockIcon, CrossIcon, InfoIcon, MinusIcon } from "./icons";

type Tone = "good" | "warning" | "critical" | "info" | "neutral";

const ICONS: Record<Tone, ReactNode> = {
  good: <CheckIcon size={12} />,
  warning: <AlertIcon size={12} />,
  critical: <CrossIcon size={12} />,
  info: <InfoIcon size={12} />,
  neutral: <MinusIcon size={12} />,
};

/** A status colour never appears alone: every badge carries an icon and a text label. */
export function Badge({ tone, children, icon }: { tone: Tone; children: ReactNode; icon?: ReactNode }) {
  const className = tone === "neutral" ? "badge" : `badge badge-${tone}`;
  return (
    <span className={className}>
      {icon ?? ICONS[tone]}
      {children}
    </span>
  );
}

export function DocumentStatusBadge({ status }: { status: DocumentStatus }) {
  switch (status) {
    case "ready":
      return <Badge tone="good">Ready</Badge>;
    case "failed":
      return <Badge tone="critical">Failed</Badge>;
    case "processing":
      return (
        <Badge tone="info" icon={<span className="spinner" style={{ width: 10, height: 10 }} />}>
          Processing
        </Badge>
      );
    default:
      return (
        <Badge tone="neutral" icon={<ClockIcon size={12} />}>
          Pending
        </Badge>
      );
  }
}

export function AnswerStatusBadge({ status, grounding }: { status: AnswerStatus; grounding: Grounding | null }) {
  if (status === "insufficient_evidence") return <Badge tone="info">Not enough evidence</Badge>;
  if (status === "invalid_output") return <Badge tone="critical">Invalid model output</Badge>;
  if (grounding === "grounded") return <Badge tone="good">Grounded in sources</Badge>;
  if (grounding === "partially_grounded") return <Badge tone="warning">Partly grounded</Badge>;
  return <Badge tone="critical">Not grounded</Badge>;
}

/** Whether the answer's statements were found in the retrieved passages (claim check). */
export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  switch (verdict) {
    case "verified":
      return <Badge tone="good">Every statement found in the sources</Badge>;
    case "partially_verified":
      return <Badge tone="warning">Some statements not found in the sources</Badge>;
    case "unverified":
      return <Badge tone="critical">Not found in the sources</Badge>;
    default:
      return <Badge tone="neutral">No statements to check</Badge>;
  }
}

export function ClaimStatusBadge({ status }: { status: ClaimStatus }) {
  switch (status) {
    case "supported":
      return <Badge tone="good">Supported by the cited passage</Badge>;
    case "wrong_source":
      return <Badge tone="warning">Cited the wrong passage</Badge>;
    case "added_source":
      return <Badge tone="info">Found in a passage, not cited</Badge>;
    case "partial":
      return <Badge tone="warning">Partly found</Badge>;
    default:
      return <Badge tone="critical">Not found in the sources</Badge>;
  }
}

export function SupportBadge({ support }: { support: SupportLabel }) {
  switch (support) {
    case "supported":
      return <Badge tone="good">Supported</Badge>;
    case "partial":
      return <Badge tone="warning">Partly supported</Badge>;
    case "unsupported":
      return <Badge tone="critical">Not supported</Badge>;
    default:
      return <Badge tone="neutral">No citation</Badge>;
  }
}
