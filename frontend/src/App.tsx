import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";

import { ApiError, getStoredApiKey, LiveApi, storeApiKey, type Api } from "./api/client";
import { StaticDemoApi, type DemoManifest } from "./api/demo";
import type { SystemStatus } from "./api/types";
import { Banner } from "./components/Banner";
import { AppContext } from "./context";
import { AskPage } from "./pages/AskPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { OperationsPage } from "./pages/OperationsPage";

const TABS = [
  { id: "ask", label: "Ask" },
  { id: "documents", label: "Documents" },
  { id: "evaluation", label: "Evaluation" },
  { id: "operations", label: "Operations" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function tabFromHash(): TabId {
  let value = "";
  try {
    value = window.location.hash.replace(/^#\/?/, "");
  } catch {
    // An embedding frame may hide the location; start on the first tab.
  }
  return (TABS.find((tab) => tab.id === value)?.id ?? "ask") as TabId;
}

/**
 * Records the tab in the address bar so it can be bookmarked and Back works. The view is
 * switched by React state before this runs, so a frame that blocks or ignores history
 * changes (as some embedding hosts do) cannot stop navigation.
 */
function rememberTab(id: TabId): void {
  try {
    if (window.location.hash !== `#${id}`) window.history.pushState(null, "", `#${id}`);
  } catch {
    // History is unavailable here; the tab still changes.
  }
}

export function createDefaultApi(): Api {
  if (import.meta.env.VITE_DEMO_MODE === "true") return new StaticDemoApi();
  return new LiveApi(import.meta.env.VITE_API_BASE_URL ?? "");
}

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem("groundedai.theme") ?? "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("groundedai.theme", theme);
    } catch {
      // ignore unavailable storage
    }
  }, [theme]);
  const cycle = () => setTheme((current) => (current === "system" ? "light" : current === "light" ? "dark" : "system"));
  return [theme, cycle];
}

export function App({ api: injected }: { api?: Api }) {
  const api = useMemo(() => injected ?? createDefaultApi(), [injected]);
  const [tab, setTab] = useState<TabId>(tabFromHash);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusError, setStatusError] = useState<ApiError | null>(null);
  const [manifest, setManifest] = useState<DemoManifest | null>(null);
  const [keyDialog, setKeyDialog] = useState(false);
  const [askScope, setAskScope] = useState("all");
  const [theme, cycleTheme] = useTheme();

  const askAbout = useCallback((documentId: string) => {
    setAskScope(documentId);
    setTab("ask");
    rememberTab("ask");
  }, []);

  const refreshStatus = useCallback(() => {
    api
      .systemStatus()
      .then((value) => {
        setStatus(value);
        setStatusError(null);
      })
      .catch((error: unknown) => {
        setStatusError(error instanceof ApiError ? error : new ApiError(0, "unknown", String(error)));
      });
  }, [api]);

  useEffect(() => {
    refreshStatus();
    if (api instanceof StaticDemoApi) api.manifest().then(setManifest).catch(() => setManifest(null));
  }, [api, refreshStatus]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    window.addEventListener("popstate", onHash);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("popstate", onHash);
    };
  }, []);

  const contextValue = useMemo(
    () => ({ api, status, statusError, refreshStatus, askScope, setAskScope, askAbout }),
    [api, status, statusError, refreshStatus, askScope, askAbout],
  );
  const needsKey = status?.auth_required === true && !getStoredApiKey();

  return (
    <AppContext.Provider value={contextValue}>
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            GroundedAI <small>grounded answers with checked citations</small>
          </div>
          <nav className="tabs" aria-label="Sections">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                className="tab"
                aria-current={tab === item.id ? "page" : undefined}
                onClick={() => {
                  setTab(item.id);
                  rememberTab(item.id);
                }}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="header-actions">
            {status?.auth_required ? (
              <button type="button" className="btn btn-small" onClick={() => setKeyDialog(true)}>
                API key
              </button>
            ) : null}
            <button
              type="button"
              className="btn btn-small"
              onClick={cycleTheme}
              aria-label={`Theme: ${theme}`}
              title="Switch between the system theme, light and dark"
            >
              Theme: {theme}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <ModeBanners api={api} status={status} statusError={statusError} manifest={manifest} onRetry={refreshStatus} />
        {needsKey ? (
          <Banner
            tone="warning"
            title="This server requires an API key"
            action={
              <button type="button" className="btn btn-small" onClick={() => setKeyDialog(true)}>
                Enter key
              </button>
            }
          >
            The key is kept in this browser tab only (session storage) and sent in the X-API-Key header.
          </Banner>
        ) : null}
        {/* Ask stays mounted so its answer and settings survive a visit to another tab. */}
        <div className="page" hidden={tab !== "ask"}>
          <AskPage active={tab === "ask"} />
        </div>
        {tab === "documents" ? <DocumentsPage /> : null}
        {tab === "evaluation" ? <EvaluationPage /> : null}
        {tab === "operations" ? <OperationsPage /> : null}
      </main>
      <footer className="footer">
        GroundedAI {status?.version ?? ""} · answers can be wrong even when cited; open the sources before relying on
        them.
      </footer>
      {keyDialog ? <ApiKeyDialog onClose={() => setKeyDialog(false)} onSaved={refreshStatus} /> : null}
    </AppContext.Provider>
  );
}

function ModeBanners({
  api,
  status,
  statusError,
  manifest,
  onRetry,
}: {
  api: Api;
  status: SystemStatus | null;
  statusError: ApiError | null;
  manifest: DemoManifest | null;
  onRetry: () => void;
}) {
  if (api.mode === "static-demo") {
    return (
      <Banner tone="info" title="Static demo: no backend and no live inference">
        {manifest?.note ??
          "Answers are replays of responses recorded from a local run, and benchmark numbers are real measurements."}{" "}
        {manifest ? `Exported ${manifest.exported_at}.` : ""}
      </Banner>
    );
  }
  if (statusError) {
    return (
      <Banner
        tone="critical"
        title="The GroundedAI API is not reachable"
        action={
          <button type="button" className="btn btn-small" onClick={onRetry}>
            Retry
          </button>
        }
      >
        Start it with <code>groundedai serve</code> or <code>docker compose up</code>. ({statusError.message})
      </Banner>
    );
  }
  if (!status) return null;
  if (status.generation.available === false) {
    return (
      <Banner tone="warning" title={`Model unavailable: ${status.generation.provider} / ${status.generation.model ?? ""}`}>
        {status.generation.detail} Questions will fail with “model unavailable” until this is fixed; the server does not
        switch to another model on its own.
      </Banner>
    );
  }
  if (status.demo_mode) {
    return (
      <Banner tone="info" title="Demo generator active (not an LLM)">
        Answers are built by selecting sentences from the retrieved passages. To use a local language model, set{" "}
        <code>GROUNDEDAI_GENERATION_PROVIDER=ollama</code> or <code>huggingface-local</code> (see README, “Local
        inference”).
      </Banner>
    );
  }
  return null;
}

function ApiKeyDialog({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [value, setValue] = useState(getStoredApiKey());
  function save(event: FormEvent) {
    event.preventDefault();
    storeApiKey(value.trim());
    onSaved();
    onClose();
  }
  return (
    <div className="dialog-backdrop" role="presentation" onClick={onClose}>
      <form
        className="card dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="key-title"
        onSubmit={save}
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="key-title">API key</h2>
        <div className="field">
          <label htmlFor="api-key">X-API-Key</label>
          <input id="api-key" type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
        </div>
        <p className="small muted">Stored in session storage for this tab only.</p>
        <div className="controls-row" style={{ justifyContent: "flex-end" }}>
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary">
            Save
          </button>
        </div>
      </form>
    </div>
  );
}
