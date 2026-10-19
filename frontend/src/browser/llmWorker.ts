/// <reference lib="webworker" />
// Runs WebLLM in its own worker so that decoding does not block the page.
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (message: MessageEvent) => handler.onmessage(message);
