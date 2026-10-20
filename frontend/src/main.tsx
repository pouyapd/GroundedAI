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

demoApi().then(
  (api) =>
    createRoot(root).render(
      <StrictMode>
        <App api={api} />
      </StrictMode>,
    ),
  (error: unknown) => {
    // Never a blank page: say what failed and how to try again.
    const message = document.createElement("p");
    message.setAttribute("role", "alert");
    message.style.cssText = "max-width:60ch;margin:48px auto;padding:0 16px;font-family:system-ui,sans-serif";
    message.textContent = `The demo could not start (${error instanceof Error ? error.message : String(error)}). Reload the page to try again; it needs a current browser with JavaScript modules and Web Workers.`;
    root.replaceChildren(message);
  },
);
