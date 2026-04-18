"use client";

import { Generation } from "../lib/api";

interface Props {
  items: Generation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function HistoryList({ items, selectedId, onSelect }: Props) {
  if (items.length === 0) {
    return (
      <div className="p-4 text-sm text-zinc-500">No generations yet.</div>
    );
  }
  return (
    <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
      {items.map((g) => {
        const prompt = typeof g.config.prompt === "string" ? g.config.prompt : "(no prompt)";
        const selected = g.id === selectedId;
        return (
          <li key={g.id}>
            <button
              onClick={() => onSelect(g.id)}
              className={`w-full text-left px-3 py-2 hover:bg-zinc-100 dark:hover:bg-zinc-900 ${
                selected ? "bg-zinc-100 dark:bg-zinc-900" : ""
              }`}
            >
              <div className="flex justify-between items-center gap-2">
                <span className="text-xs font-mono text-zinc-500">{shortTs(g.id)}</span>
                <StatusBadge state={g.status.state} />
              </div>
              <div className="text-sm mt-1 line-clamp-2">{prompt}</div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function shortTs(id: string): string {
  const [ts] = id.split("_");
  return ts?.replace("T", " ") ?? id;
}

function StatusBadge({ state }: { state: "running" | "done" | "error" | string }) {
  const color =
    state === "done"
      ? "bg-green-600"
      : state === "running"
      ? "bg-amber-500"
      : "bg-red-600";
  return (
    <span className={`text-[10px] uppercase text-white px-1.5 py-0.5 rounded ${color}`}>
      {state}
    </span>
  );
}
