const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

function url(path: string): string {
  return `${API_BASE}${path}`;
}

export type BendFunctionName =
  | "add_scalar"
  | "multiply_scalar"
  | "invert"
  | "reflect"
  | "rotate";

export interface BendSpec {
  name: string;
  function: BendFunctionName;
  params: Record<string, number | boolean>;
  steps: number[];
}

export interface GenerateRequest {
  prompt: string;
  seed: number;
  width: number;
  height: number;
  num_frames: number;
  frame_rate: number;
  enhance_prompt: boolean;
  streaming_prefetch_count: number | null;
  bending_ops: BendSpec[];
}

export interface GenerationStatus {
  state: "running" | "done" | "error";
  started_at: string;
  finished_at?: string | null;
  error?: string | null;
}

export interface Generation {
  id: string;
  config: Record<string, unknown>;
  status: GenerationStatus;
}

export async function listGenerations(): Promise<Generation[]> {
  const res = await fetch(url("/api/generations"), { cache: "no-store" });
  if (!res.ok) throw new Error(`listGenerations: ${res.status}`);
  return res.json();
}

export async function getGeneration(id: string): Promise<Generation> {
  const res = await fetch(url(`/api/generations/${encodeURIComponent(id)}`), { cache: "no-store" });
  if (!res.ok) throw new Error(`getGeneration: ${res.status}`);
  return res.json();
}

export async function generate(req: GenerateRequest): Promise<{ id: string; status: string; video_url: string }> {
  const res = await fetch(url("/api/generate"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export function videoUrl(id: string): string {
  return url(`/api/generations/${encodeURIComponent(id)}/video`);
}
