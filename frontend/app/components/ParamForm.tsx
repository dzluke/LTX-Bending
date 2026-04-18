"use client";

import { useState } from "react";
import { BendFunctionName, BendSpec, GenerateRequest } from "../lib/api";

interface Props {
  running: boolean;
  onSubmit: (req: GenerateRequest) => void;
  error: string | null;
}

const FUNCTIONS: BendFunctionName[] = [
  "add_scalar",
  "multiply_scalar",
  "invert",
  "reflect",
  "rotate",
];

function defaultParams(fn: BendFunctionName): Record<string, number> {
  if (fn === "add_scalar") return { value: 1.0 };
  if (fn === "multiply_scalar") return { factor: 2.0 };
  if (fn === "reflect") return { dim: -1 };
  if (fn === "rotate") return { k: 1 };
  return {};
}

export function ParamForm({ running, onSubmit, error }: Props) {
  const [prompt, setPrompt] = useState("A cinematic portrait of a fox in a misty forest at sunrise");
  const [seed, setSeed] = useState(42);
  const [width, setWidth] = useState(768);
  const [height, setHeight] = useState(512);
  const [numFrames, setNumFrames] = useState(49);
  const [frameRate, setFrameRate] = useState(24);
  const [enhancePrompt, setEnhancePrompt] = useState(false);
  const [streamingPrefetch, setStreamingPrefetch] = useState<number>(1);
  const [specs, setSpecs] = useState<BendSpec[]>([]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      prompt,
      seed,
      width,
      height,
      num_frames: numFrames,
      frame_rate: frameRate,
      enhance_prompt: enhancePrompt,
      streaming_prefetch_count: streamingPrefetch,
      bending_ops: specs,
    });
  }

  function addSpec() {
    setSpecs((prev) => [
      ...prev,
      { name: "", function: "multiply_scalar", params: defaultParams("multiply_scalar"), steps: [4] },
    ]);
  }
  function removeSpec(i: number) {
    setSpecs((prev) => prev.filter((_, j) => j !== i));
  }
  function updateSpec(i: number, next: BendSpec) {
    setSpecs((prev) => prev.map((s, j) => (j === i ? next : s)));
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      <Field label="Prompt">
        <textarea
          className="input h-24"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Width"><NumInput value={width} onChange={setWidth} /></Field>
        <Field label="Height"><NumInput value={height} onChange={setHeight} /></Field>
        <Field label="Frames"><NumInput value={numFrames} onChange={setNumFrames} /></Field>
        <Field label="FPS"><NumInput value={frameRate} onChange={setFrameRate} step={1} /></Field>
        <Field label="Seed"><NumInput value={seed} onChange={setSeed} /></Field>
        <Field label="Prefetch">
          <NumInput value={streamingPrefetch} onChange={setStreamingPrefetch} />
        </Field>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={enhancePrompt}
          onChange={(e) => setEnhancePrompt(e.target.checked)}
        />
        Enhance prompt
      </label>

      <section>
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-semibold">Network bending</h3>
          <button type="button" onClick={addSpec} className="btn-sm">
            + add
          </button>
        </div>
        <div className="flex flex-col gap-2">
          {specs.map((spec, i) => (
            <SpecRow
              key={i}
              spec={spec}
              onChange={(s) => updateSpec(i, s)}
              onRemove={() => removeSpec(i)}
            />
          ))}
          {specs.length === 0 && (
            <div className="text-xs text-zinc-500">No ops; video generates unchanged.</div>
          )}
        </div>
      </section>

      {error && (
        <div className="text-sm text-red-600 whitespace-pre-wrap">{error}</div>
      )}

      <button
        type="submit"
        disabled={running}
        className="mt-2 px-3 py-2 rounded bg-zinc-900 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed dark:bg-zinc-100 dark:text-zinc-900"
      >
        {running ? "Generating..." : "Generate"}
      </button>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-zinc-600 dark:text-zinc-400">{label}</span>
      {children}
    </label>
  );
}

function NumInput({
  value,
  onChange,
  step,
}: {
  value: number;
  onChange: (n: number) => void;
  step?: number;
}) {
  return (
    <input
      type="number"
      className="input"
      value={value}
      step={step ?? 1}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

function SpecRow({
  spec,
  onChange,
  onRemove,
}: {
  spec: BendSpec;
  onChange: (s: BendSpec) => void;
  onRemove: () => void;
}) {
  function setFunction(fn: BendFunctionName) {
    onChange({ ...spec, function: fn, params: defaultParams(fn) });
  }
  function setParam(key: string, value: number) {
    onChange({ ...spec, params: { ...spec.params, [key]: value } });
  }
  function setSteps(csv: string) {
    const steps = csv
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s !== "")
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    onChange({ ...spec, steps });
  }

  return (
    <div className="border border-zinc-200 dark:border-zinc-800 rounded p-2 flex flex-col gap-2">
      <div className="flex gap-2 items-center">
        <select
          className="input"
          value={spec.function}
          onChange={(e) => setFunction(e.target.value as BendFunctionName)}
        >
          {FUNCTIONS.map((fn) => (
            <option key={fn} value={fn}>{fn}</option>
          ))}
        </select>
        <button type="button" onClick={onRemove} className="btn-sm ml-auto">✕</button>
      </div>
      <label className="text-xs flex items-center gap-1">
        steps
        <input
          type="text"
          className="input"
          placeholder="4, 6, 8"
          value={spec.steps.join(", ")}
          onChange={(e) => setSteps(e.target.value)}
        />
      </label>
      <ParamInputs spec={spec} onParam={setParam} />
    </div>
  );
}

function ParamInputs({
  spec,
  onParam,
}: {
  spec: BendSpec;
  onParam: (key: string, value: number) => void;
}) {
  const keys = Object.keys(spec.params);
  if (keys.length === 0) {
    return <div className="text-xs text-zinc-500">(no parameters)</div>;
  }
  return (
    <div className="flex gap-2 flex-wrap">
      {keys.map((key) => (
        <label key={key} className="text-xs flex items-center gap-1 flex-1 min-w-0">
          {key}
          <input
            type="number"
            step="0.1"
            className="input"
            value={Number(spec.params[key] ?? 0)}
            onChange={(e) => onParam(key, Number(e.target.value))}
          />
        </label>
      ))}
    </div>
  );
}
