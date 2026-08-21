import { computeMetrics, type Metrics } from "@/lib/metrics";
import { analyzeQuality, type DataQualityReport, type Point } from "./quality";
import { createEstimator, MODEL_CATALOG, REQUIRED_MODEL_KEYS, BASELINE_MODEL_KEYS } from "./estimators";

export type { Point, DataQualityReport };
export { MODEL_CATALOG, REQUIRED_MODEL_KEYS, BASELINE_MODEL_KEYS };

export interface Scaler {
  mean: number;
  sd: number;
}
function fitScaler(v: number[]): Scaler {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length) || 1;
  return { mean, sd };
}
const apply = (s: Scaler, v: number) => (v - s.mean) / s.sd;
const invert = (s: Scaler, v: number) => v * s.sd + s.mean;

export interface BacktestFold {
  fold: number;
  trainEnd: number;
  metrics: Metrics;
}

export interface ModelRun {
  key: string;
  name: string;
  family: "required" | "baseline";
  hyperparams: Record<string, string | number>;
  /** Task A — interpolation/validation metrics on the held-out interior test split */
  validation: Metrics;
  /** Task B — metrics from ordered extrapolation backtesting (held-out tail blocks) */
  backtest: { folds: BacktestFold[]; mean: Metrics } | null;
  testPred: { x: number; actual: number; pred: number }[];
  fitCurve: { x: number; pred: number }[];
  extrapPred: { x: number; pred: number }[];
  /** relative drift of prediction vs. observed data amplitude in the extrapolated band */
  extrapolationRisk: {
    level: "low" | "moderate" | "high";
    driftRatio: number;
    slopeChange: number;
    note: string;
  } | null;
  trainSeconds: number;
}

export interface PipelineConfig {
  points: Point[];
  modelKeys: string[];
  trainPct: number;
  predStart: number | null;
  predEnd: number | null;
  extrapSteps?: number;
  backtestFolds?: number;
}

export interface PipelineResult {
  quality: DataQualityReport;
  runs: ModelRun[];
  trainRange: [number, number];
  testRange: [number, number] | null;
  extrapRange: [number, number] | null;
  nTrain: number;
  nTest: number;
  rankedByValidation: string[];
  rankedByBacktest: string[];
  recommendation: { key: string; name: string; reason: string } | null;
}

function meanMetrics(list: Metrics[]): Metrics {
  const n = list.length || 1;
  const s = (f: (m: Metrics) => number) => list.reduce((a, m) => a + f(m), 0) / n;
  return { r2: s((m) => m.r2), mae: s((m) => m.mae), mse: s((m) => m.mse), rmse: s((m) => m.rmse), evs: s((m) => m.evs) };
}

function slope(pts: { x: number; y: number }[]) {
  const n = pts.length;
  if (n < 2) return 0;
  const mx = pts.reduce((a, p) => a + p.x, 0) / n;
  const my = pts.reduce((a, p) => a + p.y, 0) / n;
  let num = 0, den = 0;
  for (const p of pts) { num += (p.x - mx) * (p.y - my); den += (p.x - mx) ** 2; }
  return den ? num / den : 0;
}

export function runPipeline(cfg: PipelineConfig): PipelineResult {
  const { points } = cfg;
  const { points: pts, report } = analyzeQuality(points);

  const splitIdx = Math.max(2, Math.floor(pts.length * (cfg.trainPct / 100)));
  const train = pts.slice(0, splitIdx);
  const test = pts.slice(splitIdx);

  const sx = fitScaler(train.map((p) => p.x));
  const sy = fitScaler(train.map((p) => p.y));
  const Xtr = train.map((p) => [apply(sx, p.x)]);
  const ytr = train.map((p) => apply(sy, p.y));

  const extrapXs: number[] = [];
  if (cfg.predStart != null && cfg.predEnd != null && cfg.predEnd > cfg.predStart) {
    const steps = cfg.extrapSteps ?? 120;
    for (let i = 0; i <= steps; i++) extrapXs.push(cfg.predStart + ((cfg.predEnd - cfg.predStart) * i) / steps);
  }

  const nFolds = cfg.backtestFolds ?? 3;
  const observedAmp = Math.max(1e-12, report.yMax - report.yMin);
  const tailSlope = slope(pts.slice(-Math.max(5, Math.floor(pts.length * 0.1))));

  const runs: ModelRun[] = cfg.modelKeys.map((key) => {
    const t0 = performance.now();
    const est = createEstimator(key);
    est.fit(Xtr, ytr);

    const testPred = test.length
      ? est.predict(test.map((p) => [apply(sx, p.x)])).map((v, i) => ({
          x: test[i].x, actual: test[i].y, pred: invert(sy, v),
        }))
      : [];
    const validation = test.length
      ? computeMetrics(testPred.map((p) => p.actual), testPred.map((p) => p.pred))
      : { r2: 0, mae: 0, mse: 0, rmse: 0, evs: 0 };

    const fitCurve = est.predict(pts.map((p) => [apply(sx, p.x)])).map((v, i) => ({
      x: pts[i].x, pred: invert(sy, v),
    }));

    const extrapPred = extrapXs.length
      ? est.predict(extrapXs.map((x) => [apply(sx, x)])).map((v, i) => ({ x: extrapXs[i], pred: invert(sy, v) }))
      : [];

    // --- Task B: ordered extrapolation backtesting (expanding window, tail blocks)
    const folds: BacktestFold[] = [];
    const block = Math.floor(pts.length / (nFolds + 2));
    if (block >= 3) {
      for (let f = 1; f <= nFolds; f++) {
        const end = pts.length - (nFolds - f + 1) * block;
        if (end < 6) continue;
        const tr = pts.slice(0, end);
        const te = pts.slice(end, end + block);
        if (!te.length) continue;
        const bx = fitScaler(tr.map((p) => p.x));
        const by = fitScaler(tr.map((p) => p.y));
        const m = createEstimator(key);
        m.fit(tr.map((p) => [apply(bx, p.x)]), tr.map((p) => apply(by, p.y)));
        const pred = m.predict(te.map((p) => [apply(bx, p.x)])).map((v) => invert(by, v));
        folds.push({ fold: f, trainEnd: tr[tr.length - 1].x, metrics: computeMetrics(te.map((p) => p.y), pred) });
      }
    }
    const backtest = folds.length ? { folds, mean: meanMetrics(folds.map((f) => f.metrics)) } : null;

    // --- Extrapolation risk
    let extrapolationRisk: ModelRun["extrapolationRisk"] = null;
    if (extrapPred.length > 2) {
      const last = pts[pts.length - 1];
      const maxDev = Math.max(...extrapPred.map((p) => Math.abs(p.pred - last.y)));
      const driftRatio = maxDev / observedAmp;
      const exSlope = slope(extrapPred.map((p) => ({ x: p.x, y: p.pred })));
      const slopeChange = Math.abs(exSlope - tailSlope) / (Math.abs(tailSlope) || 1e-12);
      const flatTree = ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "xgboost", "adaboost", "knn"].includes(key);
      const level: "low" | "moderate" | "high" =
        driftRatio > 1.5 || slopeChange > 5 ? "high" : driftRatio > 0.5 || slopeChange > 2 ? "moderate" : "low";
      extrapolationRisk = {
        level,
        driftRatio,
        slopeChange,
        note: flatTree
          ? "Tree/instance-based learners cannot extrapolate: predictions saturate at the last learned leaf value beyond the training range."
          : "Smooth learner — extrapolation follows the fitted functional form and is unconstrained by data.",
      };
    }

    const meta = MODEL_CATALOG.find((m) => m.key === key)!;
    return {
      key, name: meta.name, family: meta.family, hyperparams: meta.hyperparams,
      validation, backtest, testPred, fitCurve, extrapPred, extrapolationRisk,
      trainSeconds: (performance.now() - t0) / 1000,
    };
  });

  const rankedByValidation = [...runs].sort((a, b) => b.validation.r2 - a.validation.r2).map((r) => r.key);
  const rankedByBacktest = [...runs]
    .filter((r) => r.backtest)
    .sort((a, b) => (b.backtest!.mean.r2) - (a.backtest!.mean.r2))
    .map((r) => r.key);

  const recKey = rankedByBacktest[0] ?? rankedByValidation[0];
  const rec = runs.find((r) => r.key === recKey) ?? null;

  return {
    quality: report,
    runs,
    trainRange: [train[0]?.x ?? 0, train[train.length - 1]?.x ?? 0],
    testRange: test.length ? [test[0].x, test[test.length - 1].x] : null,
    extrapRange: extrapXs.length ? [extrapXs[0], extrapXs[extrapXs.length - 1]] : null,
    nTrain: train.length,
    nTest: test.length,
    rankedByValidation,
    rankedByBacktest,
    recommendation: rec
      ? {
          key: rec.key,
          name: rec.name,
          reason: rec.backtest
            ? `Highest mean R² (${rec.backtest.mean.r2.toFixed(4)}) across ${rec.backtest.folds.length} ordered extrapolation backtest folds, validation R² ${rec.validation.r2.toFixed(4)}.`
            : `Highest validation R² (${rec.validation.r2.toFixed(4)}); dataset too small for extrapolation backtesting.`,
        }
      : null,
  };
}
