"use client";

import { useEffect, useState } from "react";
import { Generation, GenerationStatus, videoUrl } from "../lib/api";

function formatPhase(status: GenerationStatus): string {
  if (!status.phase) return "starting…";
  if (status.step !== undefined && status.total !== undefined) {
    return `${status.phase} ${status.step + 1}/${status.total}`;
  }
  if (status.step !== undefined) {
    return `${status.phase} (${status.step})`;
  }
  return status.phase;
}

interface Props {
  generation: Generation | null;
}

export function GenerationViewer({ generation }: Props) {
  if (!generation) {
    return (
      <div className="flex-1 flex items-center justify-center text-zinc-500">
        Select a generation or create a new one.
      </div>
    );
  }

  const isDone = generation.status.state === "done";
  const isError = generation.status.state === "error";

  return (
    <div className="flex-1 flex flex-col p-3 sm:p-6 gap-4 overflow-y-auto">
      <div className="flex flex-col items-center gap-3">
        {isDone ? (
          <video
            key={generation.id}
            controls
            loop
            className="w-full max-h-[60vh] rounded bg-black"
            src={videoUrl(generation.id)}
          />
        ) : (
          <div className="h-[40vh] w-full max-w-2xl rounded bg-zinc-200 dark:bg-zinc-900 flex flex-col items-center justify-center gap-2 text-zinc-500">
            {isError ? (
              <span>Generation failed.</span>
            ) : (
              <>
                <span className="text-sm">Running…</span>
                <span className="text-xs font-mono">
                  {formatPhase(generation.status)}
                </span>
              </>
            )}
          </div>
        )}
        <div className="text-xs text-zinc-500 font-mono">{generation.id}</div>
        {generation.status.error && (
          <div className="text-sm text-red-600 max-w-2xl whitespace-pre-wrap">
            {generation.status.error}
          </div>
        )}
      </div>
      <section className="mt-2 flex flex-col gap-3">
        <JsonDetails title="Config" value={generation.config} />
        <JsonDetails title="Status" value={generation.status} />
      </section>
    </div>
  );
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    setOpen(window.matchMedia("(min-width: 768px)").matches);
  }, []);
  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="text-sm font-semibold mb-2 cursor-pointer select-none">
        {title}
      </summary>
      <pre className="text-xs bg-zinc-100 dark:bg-zinc-900 p-3 rounded overflow-x-auto">
{JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}
