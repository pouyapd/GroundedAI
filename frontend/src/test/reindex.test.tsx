import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "../App";
import type { ReindexJob, ReindexStatus } from "../api/types";
import { ReindexPanel } from "../components/ReindexPanel";
import { documentPage, fakeApi, readyDocument } from "./fixtures";

const OUTDATED: ReindexStatus = {
  ready: 3,
  current: 0,
  outdated: 3,
  outdated_without_source: 1,
  differences: [{ setting: "text clean-up", indexed: "v1", current: "v2", documents: 3 }],
  job: null,
};

function job(overrides: Partial<ReindexJob>): ReindexJob {
  return {
    id: "job1",
    state: "running",
    started_at: "2026-11-15T10:00:00Z",
    finished_at: null,
    total: 2,
    done: 0,
    reindexed: 0,
    already_current: 0,
    skipped: [],
    failed: [],
    processing_id: "doc_a",
    processing_title: "Pump manual",
    ...overrides,
  };
}

describe("ReindexPanel", () => {
  it("lists what differs, re-indexes, shows progress and reports a failure", async () => {
    const running: ReindexStatus = { ...OUTDATED, job: job({ done: 1 }) };
    const finished: ReindexStatus = {
      ...OUTDATED,
      current: 1,
      outdated: 2,
      job: job({
        state: "finished",
        done: 2,
        reindexed: 1,
        processing_id: null,
        processing_title: null,
        failed: [
          {
            document_id: "doc_b",
            title: "Boiler guide",
            result: "failed",
            detail: "The PDF has no extractable text. The previous index is still used.",
            code: "empty_document",
          },
        ],
      }),
    };
    const reindexStatus = vi.fn().mockResolvedValueOnce(OUTDATED).mockResolvedValueOnce(running).mockResolvedValue(finished);
    const startReindex = vi.fn().mockResolvedValue(running);
    const onFinished = vi.fn();
    render(<ReindexPanel api={fakeApi({ reindexStatus, startReindex })} reloadToken={0} readOnly={false} onFinished={onFinished} />);

    expect(await screen.findByText("3 documents indexed with older settings")).toBeInTheDocument();
    expect(screen.getByText("text clean-up v1 → v2 (3 documents)")).toBeInTheDocument();
    expect(screen.getByText(/1 document cannot be re-indexed because the original file was not kept/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Re-index 2 documents" }));
    expect(startReindex).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("Re-indexing: 1 of 2 documents done")).toBeInTheDocument();
    expect(screen.getByText("Now: Pump manual.")).toBeInTheDocument();

    expect(await screen.findByText(/1 failed \(they keep their previous index\)/, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText(/Boiler guide: The PDF has no extractable text/)).toBeInTheDocument();
    expect(screen.getByText("2 documents indexed with older settings")).toBeInTheDocument();
    expect(onFinished).toHaveBeenCalled();
  });

  it("shows nothing when every document is current and no job ran here", async () => {
    const reindexStatus = vi.fn().mockResolvedValue({ ...OUTDATED, outdated: 0, current: 3, outdated_without_source: 0, differences: [], job: job({ state: "finished" }) });
    const { container } = render(
      <ReindexPanel api={fakeApi({ reindexStatus })} reloadToken={0} readOnly={false} onFinished={() => undefined} />,
    );
    await vi.waitFor(() => expect(reindexStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("shows nothing for a server without the endpoint, and no button when read-only", async () => {
    const missing = vi.fn().mockRejectedValue(new Error("422"));
    const { container, unmount } = render(
      <ReindexPanel api={fakeApi({ reindexStatus: missing })} reloadToken={0} readOnly={false} onFinished={() => undefined} />,
    );
    await vi.waitFor(() => expect(missing).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
    unmount();
    render(
      <ReindexPanel api={fakeApi({ reindexStatus: vi.fn().mockResolvedValue(OUTDATED) })} reloadToken={0} readOnly onFinished={() => undefined} />,
    );
    expect(await screen.findByText("3 documents indexed with older settings")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Re-index/ })).not.toBeInTheDocument();
  });
});

describe("Documents page", () => {
  it("marks a document indexed with older settings", async () => {
    const old = {
      ...readyDocument("doc_00000000000000000001", "Pump manual", "pump.pdf"),
      index_differences: [{ setting: "text clean-up", indexed: "v1", current: "v2" }],
    };
    const api = fakeApi({
      listDocuments: vi.fn().mockResolvedValue(documentPage([old])),
      reindexStatus: vi.fn().mockResolvedValue({ ...OUTDATED, ready: 1, outdated: 1, outdated_without_source: 0 }),
    });
    render(<App api={api} />);
    await userEvent.click(await screen.findByRole("button", { name: "Documents" }));
    expect(await screen.findByText("Indexed with older settings: text clean-up v1 → v2")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Re-index 1 document" })).toBeInTheDocument();
  });
});
