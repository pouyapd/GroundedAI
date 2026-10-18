import { createContext, useContext } from "react";

import type { Api, ApiError } from "./api/client";
import type { SystemStatus } from "./api/types";

export interface AppContextValue {
  api: Api;
  status: SystemStatus | null;
  statusError: ApiError | null;
  refreshStatus: () => void;
  /** Which documents the Ask page searches: "all" or one document id. */
  askScope: string;
  setAskScope: (scope: string) => void;
  /** Opens the Ask page with one document selected. */
  askAbout: (documentId: string) => void;
}

export const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp must be used inside AppContext");
  return value;
}
