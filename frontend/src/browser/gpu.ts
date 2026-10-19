/** The in-page model of the online demo and the WebGPU check (no WebLLM import here). */

export const PAGE_MODEL = {
  id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  label: "Qwen2.5-1.5B-Instruct",
  downloadMB: 880,
  memoryMB: 1630,
} as const;

export interface GpuSupport {
  available: boolean;
  reason?: string;
}

/** Whether this browser offers WebGPU with an adapter (needed for the in-page model). */
export async function checkWebGpu(): Promise<GpuSupport> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return { available: false, reason: "This browser does not support WebGPU." };
  try {
    const adapter = await gpu.requestAdapter();
    return adapter ? { available: true } : { available: false, reason: "No WebGPU adapter is available." };
  } catch {
    return { available: false, reason: "WebGPU could not be started." };
  }
}
