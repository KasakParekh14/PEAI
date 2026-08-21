import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowLeft, CheckCircle2, Download, FileUp, Loader2, Play, Trophy, XCircle,
} from "lucide-react";
import { toast } from "sonner";
import {
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, Legend, BarChart, Bar, ReferenceLine, ReferenceArea, ZAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { parseFile, detectRange, cleanNumeric, type ParsedDataset } from "@/lib/dataset";
import {
  runPipeline, MODEL_CATALOG, REQUIRED_MODEL_KEYS, BASELINE_MODEL_KEYS, type PipelineResult,
} from "@/lib/ml/pipeline";

export const Route = createFileRoute("/app")({
  component: Dashboard,
  head: () => ({
    meta: [
      { title: "Benchmark Workspace — SpectraML" },
      { name: "description", content: "Train, benchmark and extrapolate scientific datasets with tree ensembles, SVR and neural regressors — with backtesting and research-grade export." },
      { property: "og:title", content: "Benchmark Workspace — SpectraML" },
      { property: "og:description", content: "Research-grade ML benchmarking, extrapolation backtesting and TIFF/PDF export for scientific datasets." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
});

const PALETTE = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)",
  "var(--chart-5)", "var(--primary)", "var(--accent)", "var(--chart-1)",
  "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)",
];

const sevIcon = {
  pass: <CheckCircle2 className="h-3.5 w-3.5 text-primary" />,
  warn: <AlertTriangle className="h-3.5 w-3.5 text-[var(--chart-4)]" />,
  fail: <XCircle className="h-3.5 w-3.5 text-destructive" />,
};

function Dashboard() {
  const [dataset, setDataset] = useState<ParsedDataset | null>(null);
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [predStart, setPredStart] = useState<number | "">("");
  const [predEnd, setPredEnd] = useState<number | "">("");
  const [trainPct, setTrainPct] = useState(80);
  const [selected, setSelected] = useState<string[]>([...REQUIRED_MODEL_KEYS]);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [detailKey, setDetailKey] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);
  const curvesRef = useRef<HTMLDivElement>(null);
  const scatterRef = useRef<HTMLDivElement>(null);
  const errorsRef = useRef<HTMLDivElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  const cleaned = useMemo(
    () => (dataset && xCol && yCol ? cleanNumeric(dataset.rows, xCol, yCol) : []),
    [dataset, xCol, yCol],
  );
  const range = useMemo(
    () => (cleaned.length ? { min: cleaned[0].x, max: cleaned[cleaned.length - 1].x, n: cleaned.length } : null),
    [cleaned],
  );

  async function handleUpload(f: File) {
    try {
      const ds = await parseFile(f);
      setDataset(ds);
      setResult(null);
      const x = ds.numericColumns[0] ?? ds.columns[0];
      const y = ds.numericColumns[1] ?? ds.columns[1];
      setXCol(x); setYCol(y);
      const r = detectRange(ds.rows, x);
      setPredStart(Number(r.max.toFixed(4)));
      setPredEnd(Number((r.max + (r.max - r.min) * 0.5).toFixed(4)));
      toast.success(`Loaded ${ds.rows.length} rows · X range ${r.min.toFixed(3)}–${r.max.toFixed(3)}`);
    } catch (e) {
      console.error(e);
      toast.error("Could not parse file");
    }
  }

  function toggle(key: string) {
    setSelected((s) => (s.includes(key) ? s.filter((k) => k !== key) : [...s, key]));
  }

  async function run() {
    if (cleaned.length < 12) { toast.error("Need at least 12 valid numeric rows"); return; }
    if (!selected.length) { toast.error("Select at least one model"); return; }
    if (typeof predStart === "number" && range && predStart < range.max - 1e-9) {
      toast.error("Extrapolation must start at or after the dataset maximum — interior ranges are validation, not extrapolation");
      return;
    }
    setRunning(true);
    await new Promise((r) => setTimeout(r, 30));
    try {
      const out = runPipeline({
        points: cleaned,
        modelKeys: selected,
        trainPct,
        predStart: typeof predStart === "number" ? predStart : null,
        predEnd: typeof predEnd === "number" ? predEnd : null,
      });
      setResult(out);
      setDetailKey(out.recommendation?.key ?? out.runs[0].key);
      toast.success(`Trained ${out.runs.length} models · backtested ${out.runs[0].backtest?.folds.length ?? 0} folds`);
    } catch (e: any) {
      console.error(e);
      toast.error("Training failed: " + (e?.message ?? e));
    } finally {
      setRunning(false);
    }
  }

  const ranked = useMemo(
    () => (result ? [...result.runs].sort((a, b) => b.validation.r2 - a.validation.r2) : []),
    [result],
  );
  const top = ranked.slice(0, 5);

  const curveData = useMemo(() => {
    if (!result) return [];
    const xs = Array.from(new Set([
      ...cleaned.map((p) => p.x),
      ...result.runs.flatMap((r) => r.extrapPred.map((p) => p.x)),
    ])).sort((a, b) => a - b);
    return xs.map((x) => {
      const row: Record<string, number | null> = { x, Measured: cleaned.find((p) => p.x === x)?.y ?? null };
      top.forEach((r) => {
        const f = r.fitCurve.find((p) => p.x === x)?.pred;
        const e = r.extrapPred.find((p) => Math.abs(p.x - x) < 1e-9)?.pred;
        row[r.name] = f ?? e ?? null;
      });
      return row;
    });
  }, [result, cleaned, top]);

  const detail = result?.runs.find((r) => r.key === detailKey) ?? null;

  // ---------- exports ----------
  function exportCSV() {
    if (!result) return;
    const names = result.runs.map((r) => r.name);
    const xs = Array.from(new Set([
      ...cleaned.map((p) => p.x),
      ...result.runs.flatMap((r) => r.extrapPred.map((p) => p.x)),
    ])).sort((a, b) => a - b);
    const lines = [`x,measured,region,${names.join(",")}`];
    for (const x of xs) {
      const measured = cleaned.find((p) => p.x === x)?.y ?? "";
      const region = range && x > range.max ? "extrapolation" : x >= result.trainRange[1] ? "test" : "train";
      const cols = result.runs.map((r) => {
        const f = r.fitCurve.find((p) => p.x === x)?.pred;
        const e = r.extrapPred.find((p) => Math.abs(p.x - x) < 1e-9)?.pred;
        return (f ?? e ?? "").toString();
      });
      lines.push(`${x},${measured},${region},${cols.join(",")}`);
    }
    lines.push("", "# Model metrics (Task A validation | Task B backtest)");
    lines.push("model,val_r2,val_mae,val_mse,val_rmse,val_evs,backtest_r2,backtest_rmse,extrapolation_risk");
    for (const r of result.runs) {
      lines.push([
        r.name, r.validation.r2, r.validation.mae, r.validation.mse, r.validation.rmse, r.validation.evs,
        r.backtest?.mean.r2 ?? "", r.backtest?.mean.rmse ?? "", r.extrapolationRisk?.level ?? "",
      ].join(","));
    }
    download(new Blob([lines.join("\n")], { type: "text/csv" }), `spectraml_results_${Date.now()}.csv`);
  }

  function download(blob: Blob, name: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  }

  async function exportTIFF(node: HTMLDivElement | null, label: string) {
    if (!node) return;
    try {
      const [{ default: html2canvas }, UTIF] = await Promise.all([import("html2canvas"), import("utif")]);
      const bg = getComputedStyle(document.body).backgroundColor || "#0b0f1a";
      const canvas = await html2canvas(node, { backgroundColor: bg, scale: 3 });
      const ctx = canvas.getContext("2d")!;
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const tiff: ArrayBuffer = (UTIF as any).encodeImage(img.data, canvas.width, canvas.height);
      download(new Blob([tiff], { type: "image/tiff" }), `spectraml_${label}_${Date.now()}.tiff`);
      toast.success(`${label} exported as 3× TIFF`);
    } catch (e: any) {
      toast.error("TIFF export failed: " + (e?.message ?? e));
    }
  }

  async function exportPDF() {
    if (!result) return;
    try {
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ unit: "pt", format: "a4" });
      let y = 48;
      const line = (t: string, size = 10, bold = false) => {
        doc.setFont("helvetica", bold ? "bold" : "normal");
        doc.setFontSize(size);
        for (const s of doc.splitTextToSize(t, 500)) {
          if (y > 780) { doc.addPage(); y = 48; }
          doc.text(s, 48, y); y += size + 4;
        }
      };
      line("SpectraML — Research Report", 16, true);
      line(`Generated ${new Date().toISOString()}`, 9);
      y += 8;
      line("1. Dataset & data quality", 12, true);
      line(`Columns: X = ${xCol}, Y = ${yCol}`);
      line(`Usable points: ${result.quality.n} · X ∈ [${result.quality.xMin}, ${result.quality.xMax}] · Y ∈ [${result.quality.yMin}, ${result.quality.yMax}]`);
      line(`Rows dropped (non-numeric/empty): ${result.quality.droppedRows} · duplicates averaged: ${result.quality.duplicates}`);
      result.quality.checks.forEach((c) => line(`  [${c.severity.toUpperCase()}] ${c.label} — ${c.detail}`, 9));
      y += 6;
      line("2. Protocol", 12, true);
      line(`Task A (validation): ordered ${trainPct}/${100 - trainPct} split — train X ∈ [${result.trainRange[0]}, ${result.trainRange[1]}], test X ∈ [${result.testRange?.[0]}, ${result.testRange?.[1]}]. No shuffling; no smoothing applied at any stage.`);
      line(`Task B (extrapolation): predictions over X ∈ [${result.extrapRange?.[0] ?? "—"}, ${result.extrapRange?.[1] ?? "—"}], strictly beyond the measured range. Honesty check performed by expanding-window ordered backtesting on held-out tail blocks.`);
      y += 6;
      line("3. Model benchmark", 12, true);
      result.runs.forEach((r) => {
        line(`${r.name} [${r.family}]`, 10, true);
        line(`  Validation: R²=${r.validation.r2.toFixed(4)} MAE=${r.validation.mae.toExponential(3)} MSE=${r.validation.mse.toExponential(3)} RMSE=${r.validation.rmse.toExponential(3)} EVS=${r.validation.evs.toFixed(4)}`, 9);
        line(`  Backtest: ${r.backtest ? `mean R²=${r.backtest.mean.r2.toFixed(4)} RMSE=${r.backtest.mean.rmse.toExponential(3)} over ${r.backtest.folds.length} folds` : "not available (dataset too small)"}`, 9);
        line(`  Extrapolation risk: ${r.extrapolationRisk?.level ?? "n/a"} — ${r.extrapolationRisk?.note ?? ""}`, 9);
        line(`  Hyperparameters: ${Object.entries(r.hyperparams).map(([k, v]) => `${k}=${v}`).join(", ")}`, 8);
        line(`  Fit time: ${r.trainSeconds.toFixed(2)} s`, 8);
      });
      y += 6;
      line("4. Recommendation", 12, true);
      line(result.recommendation ? `${result.recommendation.name} — ${result.recommendation.reason}` : "None");
      line("Caveat: extrapolation beyond the measured range is not validated by any held-out measurement. Tree and instance-based models saturate outside the training domain by construction; smooth models diverge according to their functional form. Treat extrapolated values as hypotheses, not measurements.", 9);
      doc.save(`spectraml_report_${Date.now()}.pdf`);
      toast.success("PDF report exported");
    } catch (e: any) {
      toast.error("PDF export failed: " + (e?.message ?? e));
    }
  }

  return (
    <div className="min-h-screen">
      <nav className="flex items-center justify-between px-6 py-4 border-b border-border/50">
        <Link to="/" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> <Activity className="h-4 w-4 text-primary" />
          <span className="font-mono">SpectraML</span>
        </Link>
        {result?.recommendation && (
          <div className="flex items-center gap-2 text-xs">
            <Badge variant="secondary" className="gap-1">
              <Trophy className="h-3 w-3 text-primary" /> Recommended: {result.recommendation.name}
            </Badge>
            <Button size="sm" variant="outline" onClick={exportPDF}>
              <Download className="h-3 w-3 mr-1" /> PDF report
            </Button>
          </div>
        )}
      </nav>

      <div className="grid lg:grid-cols-[350px_1fr] gap-6 p-6 max-w-[1700px] mx-auto">
        <aside className="space-y-4">
          <Card className="p-5 space-y-4">
            <div className="font-medium text-sm">1 · Dataset</div>
            <input ref={fileRef} type="file" hidden accept=".csv,.txt,.xlsx,.xls"
              onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])} />
            <Button variant="secondary" className="w-full" onClick={() => fileRef.current?.click()}>
              <FileUp className="h-4 w-4 mr-2" /> Upload CSV / Excel / TXT
            </Button>
            {dataset && (
              <>
                <p className="text-xs text-muted-foreground font-mono">
                  {dataset.rows.length} rows · {dataset.columns.length} columns
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label className="text-xs">X (independent)</Label>
                    <Select value={xCol} onValueChange={setXCol}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {dataset.numericColumns.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs">Y (target)</Label>
                    <Select value={yCol} onValueChange={setYCol}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {dataset.numericColumns.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {range && (
                  <div className="text-xs font-mono text-muted-foreground bg-muted/40 rounded px-2 py-1.5">
                    Measured range {range.min.toFixed(3)} → {range.max.toFixed(3)} ({range.n} pts)
                  </div>
                )}
              </>
            )}
          </Card>

          {dataset && (
            <Card className="p-5 space-y-3">
              <div className="font-medium text-sm">2 · Extrapolation window (Task B)</div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label className="text-xs">Start</Label>
                  <Input type="number" value={predStart}
                    onChange={(e) => setPredStart(e.target.value === "" ? "" : Number(e.target.value))} />
                </div>
                <div>
                  <Label className="text-xs">End</Label>
                  <Input type="number" value={predEnd}
                    onChange={(e) => setPredEnd(e.target.value === "" ? "" : Number(e.target.value))} />
                </div>
              </div>
              {range && (
                <div className="flex flex-wrap gap-1">
                  {[0.25, 0.5, 1, 2].map((m) => (
                    <Button key={m} size="sm" variant="outline" className="h-7 text-xs"
                      onClick={() => {
                        setPredStart(Number(range.max.toFixed(4)));
                        setPredEnd(Number((range.max + (range.max - range.min) * m).toFixed(4)));
                      }}>
                      +{m * 100}%
                    </Button>
                  ))}
                </div>
              )}
              <p className="text-[11px] text-muted-foreground leading-snug">
                Must begin at or after the measured maximum. Interior predictions are validation, not extrapolation.
              </p>
            </Card>
          )}

          {dataset && (
            <Card className="p-5 space-y-3">
              <div className="font-medium text-sm">3 · Ordered split (Task A)</div>
              <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
                <span>Train {trainPct}%</span><span>Test {100 - trainPct}%</span>
              </div>
              <Slider value={[trainPct]} min={50} max={90} step={5} onValueChange={(v) => setTrainPct(v[0])} />
              <p className="text-[11px] text-muted-foreground leading-snug">
                Split is ordered by X — never shuffled — so the test block is always the unseen upper region.
              </p>
            </Card>
          )}

          {dataset && (
            <Card className="p-5 space-y-3">
              <div className="font-medium text-sm">4 · Models</div>
              <div className="space-y-1.5">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Required suite</p>
                {MODEL_CATALOG.filter((m) => m.family === "required").map((m) => (
                  <label key={m.key} className="flex items-center gap-2 text-xs cursor-pointer">
                    <Checkbox checked={selected.includes(m.key)} onCheckedChange={() => toggle(m.key)} />
                    {m.name}
                  </label>
                ))}
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground pt-2">Optional baselines</p>
                {MODEL_CATALOG.filter((m) => m.family === "baseline").map((m) => (
                  <label key={m.key} className="flex items-center gap-2 text-xs cursor-pointer">
                    <Checkbox checked={selected.includes(m.key)} onCheckedChange={() => toggle(m.key)} />
                    {m.name}
                  </label>
                ))}
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="ghost" className="h-7 text-xs"
                  onClick={() => setSelected([...REQUIRED_MODEL_KEYS])}>Required only</Button>
                <Button size="sm" variant="ghost" className="h-7 text-xs"
                  onClick={() => setSelected([...REQUIRED_MODEL_KEYS, ...BASELINE_MODEL_KEYS])}>All</Button>
              </div>
            </Card>
          )}

          {dataset && (
            <Button onClick={run} disabled={running || !cleaned.length} className="w-full">
              {running
                ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Training {selected.length} models…</>
                : <><Play className="h-4 w-4 mr-2" /> Train, benchmark & backtest</>}
            </Button>
          )}
        </aside>

        <main className="space-y-4 min-w-0">
          {!dataset && (
            <Card className="p-16 text-center border-dashed">
              <FileUp className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
              <p className="text-muted-foreground">Upload a dataset to begin.</p>
              <p className="text-xs text-muted-foreground mt-1">CSV, TXT or Excel · first row must be column headers</p>
            </Card>
          )}

          {dataset && !result && (
            <Card className="p-5">
              <div className="font-medium text-sm mb-3">Raw measurements (no smoothing applied)</div>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" stroke="var(--muted-foreground)" fontSize={11} />
                  <YAxis type="number" dataKey="y" stroke="var(--muted-foreground)" fontSize={11} />
                  <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }} />
                  <Scatter data={cleaned} fill="var(--primary)" />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
          )}

          {result && (
            <>
              <Card className="p-4" ref={reportRef}>
                <div className="flex items-center justify-between mb-3">
                  <div className="font-medium text-sm">Data quality report</div>
                  <Badge variant={result.quality.usable ? "secondary" : "destructive"}>
                    {result.quality.usable ? "Usable for modelling" : "Quality issues — interpret with caution"}
                  </Badge>
                </div>
                <div className="grid sm:grid-cols-2 gap-x-6 gap-y-1.5">
                  {result.quality.checks.map((c) => (
                    <div key={c.id} className="flex items-start gap-2 text-xs">
                      {sevIcon[c.severity]}
                      <div>
                        <span className="font-medium">{c.label}</span>
                        <span className="text-muted-foreground"> — {c.detail}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-border/50 text-[11px] font-mono text-muted-foreground grid sm:grid-cols-3 gap-1">
                  <span>train X ∈ [{result.trainRange[0].toFixed(3)}, {result.trainRange[1].toFixed(3)}] · n={result.nTrain}</span>
                  <span>test X ∈ [{result.testRange?.[0].toFixed(3) ?? "—"}, {result.testRange?.[1].toFixed(3) ?? "—"}] · n={result.nTest}</span>
                  <span>extrapolate X ∈ [{result.extrapRange?.[0].toFixed(3) ?? "—"}, {result.extrapRange?.[1].toFixed(3) ?? "—"}]</span>
                </div>
              </Card>

              <Tabs defaultValue="curves">
                <TabsList className="flex-wrap h-auto">
                  <TabsTrigger value="curves">Predicted curves</TabsTrigger>
                  <TabsTrigger value="scatter">Actual vs Predicted</TabsTrigger>
                  <TabsTrigger value="ranking">Model ranking</TabsTrigger>
                  <TabsTrigger value="errors">Error comparison</TabsTrigger>
                  <TabsTrigger value="reliability">Extrapolation reliability</TabsTrigger>
                  <TabsTrigger value="details">Model details</TabsTrigger>
                </TabsList>

                {/* 1 — curves */}
                <TabsContent value="curves">
                  <Card className="p-5" ref={curvesRef}>
                    <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
                      <div>
                        <div className="font-medium text-sm">Measured data, model fits and extrapolation</div>
                        <p className="text-[11px] text-muted-foreground">Top 5 models by validation R². Raw points only — no smoothing.</p>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={exportCSV}><Download className="h-3 w-3 mr-1" /> CSV</Button>
                        <Button size="sm" variant="outline" onClick={() => exportTIFF(curvesRef.current, "curves")}><Download className="h-3 w-3 mr-1" /> TIFF</Button>
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={440}>
                      <LineChart data={curveData}>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} stroke="var(--muted-foreground)" fontSize={11} />
                        <YAxis stroke="var(--muted-foreground)" fontSize={11} />
                        <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {result.testRange && (
                          <ReferenceArea x1={result.testRange[0]} x2={result.testRange[1]} fill="var(--muted)" fillOpacity={0.25} />
                        )}
                        {range && <ReferenceLine x={range.max} stroke="var(--accent)" strokeDasharray="4 4"
                          label={{ value: "measured limit → extrapolation", fill: "var(--accent)", fontSize: 10 }} />}
                        <Line type="monotone" dataKey="Measured" stroke="var(--foreground)" dot={false} strokeWidth={2} connectNulls={false} />
                        {top.map((r, i) => (
                          <Line key={r.key} type="monotone" dataKey={r.name} stroke={PALETTE[i]} dot={false} strokeWidth={1.5} connectNulls />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </TabsContent>

                {/* 2 — actual vs predicted */}
                <TabsContent value="scatter">
                  <Card className="p-5" ref={scatterRef}>
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <div className="font-medium text-sm">Actual vs predicted (validation block only)</div>
                        <p className="text-[11px] text-muted-foreground">Diagonal = perfect prediction. Extrapolated points are excluded — they have no ground truth.</p>
                      </div>
                      <Button size="sm" variant="outline" onClick={() => exportTIFF(scatterRef.current, "actual_vs_predicted")}>
                        <Download className="h-3 w-3 mr-1" /> TIFF
                      </Button>
                    </div>
                    <ResponsiveContainer width="100%" height={420}>
                      <ScatterChart>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--muted-foreground)" fontSize={11} />
                        <YAxis type="number" dataKey="pred" name="Predicted" stroke="var(--muted-foreground)" fontSize={11} />
                        <ZAxis range={[24, 24]} />
                        <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {top.map((r, i) => (
                          <Scatter key={r.key} name={r.name} data={r.testPred} fill={PALETTE[i]} fillOpacity={0.7} />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Card>
                </TabsContent>

                {/* 3 — ranking */}
                <TabsContent value="ranking">
                  <Card className="p-5 overflow-x-auto">
                    <div className="font-medium text-sm mb-1">Benchmark table</div>
                    <p className="text-[11px] text-muted-foreground mb-3">
                      Task A = held-out validation inside the measured range. Task B = mean over ordered extrapolation backtest folds.
                    </p>
                    <table className="w-full text-xs font-mono">
                      <thead className="text-muted-foreground">
                        <tr className="border-b border-border/60">
                          <th className="text-left py-2 pr-3">#</th>
                          <th className="text-left pr-3">Model</th>
                          <th className="text-right pr-3">R²</th>
                          <th className="text-right pr-3">MAE</th>
                          <th className="text-right pr-3">MSE</th>
                          <th className="text-right pr-3">RMSE</th>
                          <th className="text-right pr-3">EVS</th>
                          <th className="text-right pr-3">Backtest R²</th>
                          <th className="text-right pr-3">Backtest RMSE</th>
                          <th className="text-right">Fit (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ranked.map((r, i) => (
                          <tr key={r.key} className="border-b border-border/30">
                            <td className="py-1.5 pr-3">{i + 1}</td>
                            <td className="pr-3 whitespace-nowrap">
                              {r.name}{" "}
                              {r.family === "baseline" && <span className="text-muted-foreground">(baseline)</span>}
                            </td>
                            <td className="text-right pr-3">{r.validation.r2.toFixed(4)}</td>
                            <td className="text-right pr-3">{r.validation.mae.toExponential(2)}</td>
                            <td className="text-right pr-3">{r.validation.mse.toExponential(2)}</td>
                            <td className="text-right pr-3">{r.validation.rmse.toExponential(2)}</td>
                            <td className="text-right pr-3">{r.validation.evs.toFixed(4)}</td>
                            <td className="text-right pr-3">{r.backtest ? r.backtest.mean.r2.toFixed(4) : "—"}</td>
                            <td className="text-right pr-3">{r.backtest ? r.backtest.mean.rmse.toExponential(2) : "—"}</td>
                            <td className="text-right">{r.trainSeconds.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {result.recommendation && (
                      <div className="mt-4 text-xs bg-muted/40 rounded p-3">
                        <span className="font-medium">Recommended for extrapolation: {result.recommendation.name}</span>
                        <p className="text-muted-foreground mt-1">{result.recommendation.reason}</p>
                      </div>
                    )}
                  </Card>
                </TabsContent>

                {/* 4 — errors */}
                <TabsContent value="errors">
                  <Card className="p-5" ref={errorsRef}>
                    <div className="flex items-center justify-between mb-3">
                      <div className="font-medium text-sm">Error comparison across models</div>
                      <Button size="sm" variant="outline" onClick={() => exportTIFF(errorsRef.current, "errors")}>
                        <Download className="h-3 w-3 mr-1" /> TIFF
                      </Button>
                    </div>
                    <ResponsiveContainer width="100%" height={360}>
                      <BarChart data={ranked.map((r) => ({
                        name: r.name, RMSE: r.validation.rmse, MAE: r.validation.mae,
                        "Backtest RMSE": r.backtest?.mean.rmse ?? 0,
                      }))}>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={10} angle={-25} textAnchor="end" height={80} interval={0} />
                        <YAxis stroke="var(--muted-foreground)" fontSize={11} />
                        <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Bar dataKey="MAE" fill="var(--chart-2)" />
                        <Bar dataKey="RMSE" fill="var(--chart-1)" />
                        <Bar dataKey="Backtest RMSE" fill="var(--chart-4)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </TabsContent>

                {/* 5 — reliability */}
                <TabsContent value="reliability">
                  <Card className="p-5 space-y-4">
                    <div>
                      <div className="font-medium text-sm">Extrapolation reliability</div>
                      <p className="text-[11px] text-muted-foreground">
                        No metric can validate predictions outside the measured range. These diagnostics quantify how far each
                        model departs from observed behaviour, plus its accuracy on ordered backtest folds that simulate extrapolation.
                      </p>
                    </div>
                    <div className="grid gap-2">
                      {ranked.map((r) => (
                        <div key={r.key} className="flex items-start gap-3 text-xs border border-border/50 rounded p-3">
                          <Badge variant={
                            r.extrapolationRisk?.level === "low" ? "secondary"
                              : r.extrapolationRisk?.level === "moderate" ? "outline" : "destructive"
                          }>
                            {r.extrapolationRisk?.level ?? "n/a"}
                          </Badge>
                          <div className="min-w-0">
                            <div className="font-medium">{r.name}</div>
                            <div className="font-mono text-muted-foreground">
                              drift {(100 * (r.extrapolationRisk?.driftRatio ?? 0)).toFixed(1)}% of measured amplitude ·
                              slope change ×{(r.extrapolationRisk?.slopeChange ?? 0).toFixed(2)} ·
                              backtest R² {r.backtest ? r.backtest.mean.r2.toFixed(4) : "—"}
                            </div>
                            <p className="text-muted-foreground mt-1">{r.extrapolationRisk?.note}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    {ranked[0]?.backtest && (
                      <div>
                        <div className="font-medium text-sm mb-2">Backtest fold accuracy (per model)</div>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={(ranked[0].backtest.folds).map((f, i) => {
                            const row: Record<string, number | string> = { fold: `fold ${f.fold} · X≤${f.trainEnd.toFixed(2)}` };
                            ranked.slice(0, 6).forEach((r) => { row[r.name] = r.backtest?.folds[i]?.metrics.r2 ?? 0; });
                            return row;
                          })}>
                            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                            <XAxis dataKey="fold" stroke="var(--muted-foreground)" fontSize={10} />
                            <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[-1, 1]} />
                            <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }} />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                            {ranked.slice(0, 6).map((r, i) => <Bar key={r.key} dataKey={r.name} fill={PALETTE[i]} />)}
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Card>
                </TabsContent>

                {/* 6 — details */}
                <TabsContent value="details">
                  <Card className="p-5 space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="font-medium text-sm">Model details</div>
                      <Select value={detailKey} onValueChange={setDetailKey}>
                        <SelectTrigger className="w-64"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {result.runs.map((r) => <SelectItem key={r.key} value={r.key}>{r.name}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </div>
                    {detail && (
                      <div className="grid md:grid-cols-2 gap-4 text-xs">
                        <div>
                          <div className="font-medium mb-2">Hyperparameters</div>
                          <table className="w-full font-mono">
                            <tbody>
                              {Object.entries(detail.hyperparams).map(([k, v]) => (
                                <tr key={k} className="border-b border-border/30">
                                  <td className="py-1 text-muted-foreground">{k}</td>
                                  <td className="py-1 text-right">{String(v)}</td>
                                </tr>
                              ))}
                              <tr><td className="py-1 text-muted-foreground">fit time</td><td className="py-1 text-right">{detail.trainSeconds.toFixed(3)} s</td></tr>
                            </tbody>
                          </table>
                        </div>
                        <div>
                          <div className="font-medium mb-2">Metrics</div>
                          <table className="w-full font-mono">
                            <tbody>
                              {(["r2", "mae", "mse", "rmse", "evs"] as const).map((k) => (
                                <tr key={k} className="border-b border-border/30">
                                  <td className="py-1 text-muted-foreground">{k.toUpperCase()}</td>
                                  <td className="py-1 text-right">{detail.validation[k].toExponential(4)}</td>
                                  <td className="py-1 text-right">{detail.backtest ? detail.backtest.mean[k].toExponential(4) : "—"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          <p className="text-[11px] text-muted-foreground mt-1">Columns: validation · backtest mean</p>
                        </div>
                        <div className="md:col-span-2 text-muted-foreground leading-snug">
                          <span className="font-medium text-foreground">Extrapolation behaviour: </span>{detail.extrapolationRisk?.note ?? "No extrapolation window configured."}
                        </div>
                      </div>
                    )}
                  </Card>
                </TabsContent>
              </Tabs>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
