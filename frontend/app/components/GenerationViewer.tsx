"use client";

import { Generation, videoUrl } from "../lib/api";

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
    <div className="flex-1 flex flex-col p-6 gap-4 overflow-y-auto">
      <div className="flex flex-col items-center gap-3">
        {isDone ? (
          <video
            key={generation.id}
            controls
            loop
            className="max-h-[60vh] w-auto rounded bg-black"
            src={videoUrl(generation.id)}
          />
        ) : (
          <div className="h-[40vh] w-full max-w-2xl rounded bg-zinc-200 dark:bg-zinc-900 flex items-center justify-center text-zinc-500">
            {isError ? "Generation failed." : "Running..."}
          </div>
        )}
        <div className="text-xs text-zinc-500 font-mono">{generation.id}</div>
        {generation.status.error && (
          <div className="text-sm text-red-600 max-w-2xl whitespace-pre-wrap">
            {generation.status.error}
          </div>
        )}
      </div>
      <section className="mt-2">
        <h2 className="text-sm font-semibold mb-2">Config</h2>
        <pre className="text-xs bg-zinc-100 dark:bg-zinc-900 p-3 rounded overflow-x-auto">
{JSON.stringify(generation.config, null, 2)}
        </pre>
        <h2 className="text-sm font-semibold mt-4 mb-2">Status</h2>
        <pre className="text-xs bg-zinc-100 dark:bg-zinc-900 p-3 rounded overflow-x-auto">
{JSON.stringify(generation.status, null, 2)}
        </pre>
      </section>
    </div>
  );
}
