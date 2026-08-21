// Pure-JS regression models. Inputs are 1-D x → y for the THz/freq use case.

export interface Model {
  name: string;
  fit(x: number[], y: number[]): void;
  predict(x: number[]): number[];
}

// ---------- Polynomial / Linear (least squares) ----------
function solveNormal(X: number[][], y: number[]): number[] {
  // Solve (X^T X) β = X^T y  via Gauss-Jordan
  const m = X.length;
  const n = X[0].length;
  const XtX = Array.from({ length: n }, () => Array(n).fill(0));
  const Xty = Array(n).fill(0);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      Xty[j] += X[i][j] * y[i];
      for (let k = 0; k < n; k++) XtX[j][k] += X[i][j] * X[i][k];
    }
  // Augment
  const A = XtX.map((row, i) => [...row, Xty[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    [A[i], A[piv]] = [A[piv], A[i]];
    const d = A[i][i] || 1e-12;
    for (let c = i; c <= n; c++) A[i][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = A[r][i];
      for (let c = i; c <= n; c++) A[r][c] -= f * A[i][c];
    }
  }
  return A.map((r) => r[n]);
}

class PolyModel implements Model {
  name: string;
  private deg: number;
  private beta: number[] = [];
  private xMean = 0;
  private xStd = 1;
  constructor(deg: number, name?: string) {
    this.deg = deg;
    this.name = name ?? (deg === 1 ? "Linear Regression" : `Polynomial-${deg}`);
  }
  private design(x: number[]) {
    return x.map((v) => {
      const z = (v - this.xMean) / this.xStd;
      const row = [1];
      for (let d = 1; d <= this.deg; d++) row.push(row[d - 1] * z);
      return row;
    });
  }
  fit(x: number[], y: number[]) {
    this.xMean = x.reduce((a, b) => a + b, 0) / x.length;
    const v = x.reduce((a, b) => a + (b - this.xMean) ** 2, 0) / x.length;
    this.xStd = Math.sqrt(v) || 1;
    this.beta = solveNormal(this.design(x), y);
  }
  predict(x: number[]) {
    return this.design(x).map((row) => row.reduce((s, v, i) => s + v * this.beta[i], 0));
  }
}

// ---------- KNN ----------
class KNNModel implements Model {
  name: string;
  private xs: number[] = [];
  private ys: number[] = [];
  constructor(private k = 5) { this.name = `KNN (k=${k})`; }
  fit(x: number[], y: number[]) { this.xs = [...x]; this.ys = [...y]; }
  predict(x: number[]) {
    return x.map((q) => {
      const dists = this.xs.map((xi, i) => ({ d: Math.abs(xi - q), y: this.ys[i] }));
      dists.sort((a, b) => a.d - b.d);
      const top = dists.slice(0, this.k);
      // distance weighted
      const w = top.map((t) => 1 / (t.d + 1e-9));
      const wsum = w.reduce((a, b) => a + b, 0);
      return top.reduce((s, t, i) => s + t.y * w[i], 0) / wsum;
    });
  }
}

// ---------- Decision Stump Ensemble (simple Random-Forest-like averager of polys) ----------
class EnsembleModel implements Model {
  name = "Ensemble (Boosted Polys)";
  private models: PolyModel[] = [];
  fit(x: number[], y: number[]) {
    this.models = [2, 3, 4, 5].map((d) => {
      const m = new PolyModel(d);
      m.fit(x, y);
      return m;
    });
  }
  predict(x: number[]) {
    const preds = this.models.map((m) => m.predict(x));
    return x.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length);
  }
}

// ---------- Spline-ish (LOESS-lite) ----------
class LoessModel implements Model {
  name = "Local Regression (LOESS)";
  private xs: number[] = [];
  private ys: number[] = [];
  private bw = 0.2;
  fit(x: number[], y: number[]) {
    this.xs = [...x]; this.ys = [...y];
    const min = Math.min(...x), max = Math.max(...x);
    this.bw = (max - min) * 0.2 || 1;
  }
  predict(x: number[]) {
    return x.map((q) => {
      // weighted local linear
      const w = this.xs.map((xi) => {
        const u = Math.abs(xi - q) / this.bw;
        return u < 1 ? (1 - u ** 3) ** 3 : 0.001; // tricube w/ floor for extrapolation
      });
      let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
      for (let i = 0; i < this.xs.length; i++) {
        sw += w[i]; swx += w[i] * this.xs[i]; swy += w[i] * this.ys[i];
        swxx += w[i] * this.xs[i] ** 2; swxy += w[i] * this.xs[i] * this.ys[i];
      }
      const denom = sw * swxx - swx * swx || 1e-12;
      const a = (swxx * swy - swx * swxy) / denom;
      const b = (sw * swxy - swx * swy) / denom;
      return a + b * q;
    });
  }
}

export function buildModels(): Model[] {
  return [
    new PolyModel(1),
    new PolyModel(2),
    new PolyModel(3),
    new PolyModel(5),
    new KNNModel(5),
    new KNNModel(10),
    new LoessModel(),
    new EnsembleModel(),
  ];
}
