import type { ReactNode } from "react";

import type {
  AddressesQuestion,
  AnswerStatus,
  CitationVerdict,
  ClaimStatus,
  DocumentStatus,
  EvidenceVerdict,
  Grounding,
  SupportLabel,
  Verdict,
  VerificationOut,
} from "../api/types";
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

const EVIDENCE_TEXT: Record<EvidenceVerdict, [Tone, string]> = {
  all_found: ["good", "Every statement found"],
  some_uncertain: ["warning", "Some statements only partly found"],
  some_not_found: ["warning", "Some statements not found"],
  none_found: ["critical", "No statement found"],
  not_applicable: ["neutral", "No statements to check"],
};
const CITATION_TEXT: Record<CitationVerdict, [Tone, string]> = {
  all_correct: ["good", "Citations correct"],
  some_wrong: ["critical", "Some citations wrong"],
  some_unconfirmed: ["warning", "Some citations not confirmed"],
  none_given: ["neutral", "No citations given"],
  not_applicable: ["neutral", "No citations to check"],
};
const ADDRESSES_TEXT: Record<AddressesQuestion, [Tone, string]> = {
  yes: ["good", "Gives what was asked"],
  no: ["warning", "May not give what was asked"],
  declined: ["neutral", "Says the documents do not give it"],
  not_checked: ["neutral", "Relevance not checked"],
};

/** claims-v2: evidence, citations and relevance as three badges. Answers checked by an older
 * version (or recorded before it) have none of these fields and show the single verdict badge. */
export function VerificationBadges({ verification }: { verification: VerificationOut }) {
  const { evidence, citations, addresses_question: addresses } = verification;
  if (!evidence || !citations || !addresses) return <VerdictBadge verdict={verification.verdict} />;
  return (
    <span className="badge-row" aria-label="Answer checks">
      {[EVIDENCE_TEXT[evidence], CITATION_TEXT[citations], ADDRESSES_TEXT[addresses]].map(([tone, text]) => (
        <Badge key={text} tone={tone}>
          {text}
        </Badge>
      ))}
    </span>
  );
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
