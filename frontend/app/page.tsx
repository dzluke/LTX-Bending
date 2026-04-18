"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { HistoryList } from "./components/HistoryList";
import { GenerationViewer } from "./components/GenerationViewer";
import { ParamForm } from "./components/ParamForm";
import { generate, Generation, GenerateRequest, listGenerations } from "./lib/api";

export default function Home() {
  const [items, setItems] = useState<Generation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasRunning = useMemo(() => items.some((g) => g.status.state === "running"), [items]);

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
  }, [refresh]);

  useEffect(() => {
    const interval = hasRunning ? 1000 : 5000;
    const t = setInterval(refresh, interval);
    return () => clearInterval(t);
  }, [refresh, hasRunning]);

  async function onSubmit(req: GenerateRequest) {
    setSubmitting(true);
    setError(null);
    try {
      const result = await generate(req);
      await refresh();
      setSelectedId(result.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  const selected = items.find((g) => g.id === selectedId) ?? null;

  return (
    <div className="md:flex md:flex-row md:h-[100dvh] bg-zinc-50 dark:bg-black text-zinc-950 dark:text-zinc-50">
      <main className="md:order-2 md:flex md:flex-col md:flex-1 md:min-w-0 md:overflow-y-auto border-b md:border-b-0 border-zinc-200 dark:border-zinc-800">
        <GenerationViewer generation={selected} />
      </main>

      <aside className="md:order-3 md:flex md:flex-col md:w-96 md:flex-none md:overflow-y-auto md:border-l border-b md:border-b-0 border-zinc-200 dark:border-zinc-800">
        <div className="p-3 font-semibold text-sm border-b border-zinc-200 dark:border-zinc-800">
          Parameters
        </div>
        <ParamForm running={submitting} onSubmit={onSubmit} error={error} />
      </aside>

      <aside className="md:order-1 md:flex md:flex-col md:w-72 md:flex-none md:overflow-y-auto md:border-r border-zinc-200 dark:border-zinc-800">
        <div className="p-3 font-semibold text-sm border-b border-zinc-200 dark:border-zinc-800">
          History
        </div>
        <HistoryList items={items} selectedId={selectedId} onSelect={setSelectedId} />
      </aside>
    </div>
  );
}
