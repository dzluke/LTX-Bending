"use client";

import { useCallback, useEffect, useState } from "react";
import { HistoryList } from "./components/HistoryList";
import { GenerationViewer } from "./components/GenerationViewer";
import { ParamForm } from "./components/ParamForm";
import { generate, Generation, GenerateRequest, listGenerations } from "./lib/api";

export default function Home() {
  const [items, setItems] = useState<Generation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listGenerations();
      setItems(list);
      setSelectedId((prev) => prev ?? list[0]?.id ?? null);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  async function onSubmit(req: GenerateRequest) {
    setRunning(true);
    setError(null);
    try {
      const result = await generate(req);
      await refresh();
      setSelectedId(result.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const selected = items.find((g) => g.id === selectedId) ?? null;

  return (
    <div className="flex flex-1 h-screen bg-zinc-50 dark:bg-black text-zinc-950 dark:text-zinc-50">
      <aside className="w-72 border-r border-zinc-200 dark:border-zinc-800 overflow-y-auto">
        <div className="p-3 font-semibold text-sm border-b border-zinc-200 dark:border-zinc-800">
          History
        </div>
        <HistoryList items={items} selectedId={selectedId} onSelect={setSelectedId} />
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        <GenerationViewer generation={selected} />
      </main>

      <aside className="w-96 border-l border-zinc-200 dark:border-zinc-800 overflow-y-auto">
        <div className="p-3 font-semibold text-sm border-b border-zinc-200 dark:border-zinc-800">
          Parameters
        </div>
        <ParamForm running={running} onSubmit={onSubmit} error={error} />
      </aside>
    </div>
  );
}
