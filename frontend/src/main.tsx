import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import type { Api } from "./api/client";
import { App } from "./App";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

/** The online demo loads its in-browser backend only in that build, so other builds stay small. */
async function demoApi(): Promise<Api | undefined> {
  if (import.meta.env.VITE_DEMO_MODE !== "browser") return undefined;
  const { BrowserApi } = await import("./browser/BrowserApi");
  return new BrowserApi();
}

void demoApi().then((api) =>
  createRoot(root).render(
    <StrictMode>
      <App api={api} />
    </StrictMode>,
  ),
);
