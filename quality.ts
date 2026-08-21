export interface Point { x: number; y: number }

export type Severity = "pass" | "warn" | "fail";

export interface Check {
  id: string;
  label: string;
  severity: Severity;
  detail: string;
}

export interface DataQualityReport {
  n: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  meanStep: number;
  stepCv: number;
  duplicates: number;
  droppedRows: number;
  nonFinite: number;
  outliers: number;
  noiseSigma: number;
  monotonic: boolean;
  checks: Check[];
  usable: boolean;
}

function median(a: number[]) {
  const s = [...a].sort((p, q) => p - q);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function analyzeQuality(raw: { x: unknown; y: unknown }[]): { points: Point[]; report: DataQualityReport } {
  let nonFinite = 0;
  const parsed: Point[] = [];
  for (const r of raw) {
    const x = Number(r.x), y = Number(r.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) { nonFinite++; continue; }
    parsed.push({ x, y });
  }
  parsed.sort((a, b) => a.x - b.x);

  // duplicate x → average y (never silently drop signal)
  const merged: Point[] = [];
  let duplicates = 0;
  for (const p of parsed) {
    const last = merged[merged.length - 1];
    if (last && Math.abs(last.x - p.x) < 1e-12) {
      duplicates++;
      last.y = (last.y + p.y) / 2;
    } else merged.push({ ...p });
  }

  const xs = merged.map((p) => p.x), ys = merged.map((p) => p.y);
  const steps: number[] = [];
  for (let i = 1; i < xs.length; i++) steps.push(xs[i] - xs[i - 1]);
  const meanStep = steps.length ? steps.reduce((a, b) => a + b, 0) / steps.length : 0;
  const stepSd = steps.length
    ? Math.sqrt(steps.reduce((a, b) => a + (b - meanStep) ** 2, 0) / steps.length)
    : 0;
  const stepCv = meanStep ? stepSd / Math.abs(meanStep) : 0;

  // robust outlier count vs local median (window 7)
  let outliers = 0;
  const resid: number[] = [];
  for (let i = 0; i < ys.length; i++) {
    const lo = Math.max(0, i - 3), hi = Math.min(ys.length, i + 4);
    resid.push(ys[i] - median(ys.slice(lo, hi)));
  }
  const mad = median(resid.map((r) => Math.abs(r))) || 1e-12;
  for (const r of resid) if (Math.abs(r) > 6 * 1.4826 * mad) outliers++;

  // noise estimate from lag-1 differences (Rice estimator)
  let s2 = 0;
  for (let i = 1; i < ys.length; i++) s2 += (ys[i] - ys[i - 1]) ** 2;
  const noiseSigma = ys.length > 1 ? Math.sqrt(s2 / (2 * (ys.length - 1))) : 0;

  const checks: Check[] = [];
  const push = (id: string, label: string, severity: Severity, detail: string) =>
    checks.push({ id, label, severity, detail });

  push("size", "Sample size", merged.length >= 30 ? "pass" : merged.length >= 12 ? "warn" : "fail",
    `${merged.length} usable points after cleaning`);
  push("finite", "Missing / non-numeric values", nonFinite === 0 ? "pass" : nonFinite / Math.max(1, raw.length) < 0.1 ? "warn" : "fail",
    `${nonFinite} row(s) removed as non-numeric or empty`);
  push("duplicates", "Duplicate X values", duplicates === 0 ? "pass" : "warn",
    duplicates ? `${duplicates} duplicate X value(s) averaged` : "No duplicate X values");
  push("spacing", "Sampling uniformity", stepCv < 0.05 ? "pass" : stepCv < 0.5 ? "warn" : "fail",
    `Mean step ${meanStep.toExponential(3)}, CV ${(stepCv * 100).toFixed(1)}%`);
  push("outliers", "Outliers (robust MAD, 6σ)", outliers === 0 ? "pass" : outliers / merged.length < 0.02 ? "warn" : "fail",
    `${outliers} point(s) flagged — retained, not removed`);
  push("noise", "Noise level (Rice estimator)", "pass",
    `σ ≈ ${noiseSigma.toExponential(3)} in target units`);

  const report: DataQualityReport = {
    n: merged.length,
    xMin: xs[0] ?? 0, xMax: xs[xs.length - 1] ?? 0,
    yMin: Math.min(...ys), yMax: Math.max(...ys),
    meanStep, stepCv, duplicates,
    droppedRows: nonFinite,
    nonFinite, outliers, noiseSigma,
    monotonic: true,
    checks,
    usable: merged.length >= 12 && checks.every((c) => c.severity !== "fail" || c.id === "spacing"),
  };
  return { points: merged, report };
}
