import { DecisionTree, mulberry32 } from "./tree";

export type Family = "required" | "baseline";

export interface Estimator {
  key: string;
  name: string;
  family: Family;
  hyperparams: Record<string, string | number>;
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
}

const SEED = 42;

// ---------------- Decision Tree ----------------
class DecisionTreeReg implements Estimator {
  key = "decision_tree"; name = "Decision Tree"; family: Family = "required";
  hyperparams = { criterion: "squared_error", max_depth: 8, min_samples_split: 4, min_samples_leaf: 2, random_state: SEED };
  private t = new DecisionTree({ maxDepth: 8, minSamplesSplit: 4, minSamplesLeaf: 2, rng: mulberry32(SEED) });
  fit(X: number[][], y: number[]) { this.t.fit(X, y); }
  predict(X: number[][]) { return this.t.predict(X); }
}

// ---------------- Forests ----------------
class ForestReg implements Estimator {
  key: string; name: string; family: Family = "required";
  hyperparams: Record<string, string | number>;
  private trees: DecisionTree[] = [];
  constructor(
    key: string, name: string,
    private nEstimators = 60,
    private maxDepth = 12,
    private extra = false,
  ) {
    this.key = key; this.name = name;
    this.hyperparams = {
      n_estimators: nEstimators, max_depth: maxDepth,
      bootstrap: extra ? "false" : "true",
      splitter: extra ? "random" : "best",
      min_samples_leaf: 1, random_state: SEED,
    };
  }
  fit(X: number[][], y: number[]) {
    const rng = mulberry32(SEED);
    this.trees = [];
    for (let t = 0; t < this.nEstimators; t++) {
      let Xs = X, ys = y;
      if (!this.extra) {
        const idx = X.map(() => Math.floor(rng() * X.length));
        Xs = idx.map((i) => X[i]); ys = idx.map((i) => y[i]);
      }
      const tree = new DecisionTree({
        maxDepth: this.maxDepth, minSamplesSplit: 2, minSamplesLeaf: 1,
        splitter: this.extra ? "random" : "best", rng,
      });
      tree.fit(Xs, ys);
      this.trees.push(tree);
    }
  }
  predict(X: number[][]) {
    return X.map((x) => this.trees.reduce((s, t) => s + t.predictOne(x), 0) / this.trees.length);
  }
}

// ---------------- Gradient Boosting (first order, MSE) ----------------
class GradientBoostingReg implements Estimator {
  key = "gradient_boosting"; name = "Gradient Boosting"; family: Family = "required";
  hyperparams = { n_estimators: 200, learning_rate: 0.1, max_depth: 3, loss: "squared_error", subsample: 1.0, random_state: SEED };
  private base = 0; private trees: DecisionTree[] = [];
  fit(X: number[][], y: number[]) {
    const rng = mulberry32(SEED);
    this.base = y.reduce((a, b) => a + b, 0) / y.length;
    let pred = y.map(() => this.base);
    this.trees = [];
    for (let m = 0; m < 200; m++) {
      const resid = y.map((v, i) => v - pred[i]);
      const t = new DecisionTree({ maxDepth: 3, minSamplesSplit: 2, minSamplesLeaf: 1, rng });
      t.fit(X, resid);
      const p = t.predict(X);
      pred = pred.map((v, i) => v + 0.1 * p[i]);
      this.trees.push(t);
    }
  }
  predict(X: number[][]) {
    return X.map((x) => this.base + this.trees.reduce((s, t) => s + 0.1 * t.predictOne(x), 0));
  }
}

// ---------------- XGBoost (second order, regularized) ----------------
class XGBoostReg implements Estimator {
  key = "xgboost"; name = "XGBoost"; family: Family = "required";
  hyperparams = { n_estimators: 250, learning_rate: 0.08, max_depth: 4, reg_lambda: 1.0, subsample: 0.9, objective: "reg:squarederror", random_state: SEED };
  private base = 0; private trees: DecisionTree[] = [];
  fit(X: number[][], y: number[]) {
    const rng = mulberry32(SEED);
    this.base = y.reduce((a, b) => a + b, 0) / y.length;
    let pred = y.map(() => this.base);
    this.trees = [];
    for (let m = 0; m < 250; m++) {
      const g = y.map((v, i) => pred[i] - v); // d/dF 0.5(F-y)^2
      const h = y.map(() => 1);
      const t = new DecisionTree({ maxDepth: 4, minSamplesSplit: 2, minSamplesLeaf: 1, lambda: 1.0, rng });
      t.fit(X, y, undefined, { g, h });
      const p = t.predict(X);
      pred = pred.map((v, i) => v + 0.08 * p[i]);
      this.trees.push(t);
    }
  }
  predict(X: number[][]) {
    return X.map((x) => this.base + this.trees.reduce((s, t) => s + 0.08 * t.predictOne(x), 0));
  }
}

// ---------------- AdaBoost.R2 ----------------
class AdaBoostReg implements Estimator {
  key = "adaboost"; name = "AdaBoost"; family: Family = "required";
  hyperparams = { n_estimators: 60, learning_rate: 1.0, loss: "linear", base_estimator: "DecisionTree(max_depth=4)", random_state: SEED };
  private trees: DecisionTree[] = []; private betas: number[] = [];
  fit(X: number[][], y: number[]) {
    const rng = mulberry32(SEED);
    const n = X.length;
    let w = X.map(() => 1 / n);
    this.trees = []; this.betas = [];
    for (let m = 0; m < 60; m++) {
      const t = new DecisionTree({ maxDepth: 4, minSamplesSplit: 2, minSamplesLeaf: 1, rng });
      t.fit(X, y, w.map((v) => v * n));
      const p = t.predict(X);
      const errs = y.map((v, i) => Math.abs(v - p[i]));
      const maxE = Math.max(...errs) || 1e-12;
      const L = errs.map((e) => e / maxE);
      const eps = L.reduce((s, l, i) => s + l * w[i], 0);
      if (eps >= 0.5 || eps <= 0) { if (!this.trees.length) { this.trees.push(t); this.betas.push(1); } break; }
      const beta = eps / (1 - eps);
      this.trees.push(t); this.betas.push(Math.log(1 / beta));
      w = w.map((wi, i) => wi * Math.pow(beta, 1 - L[i]));
      const sw = w.reduce((a, b) => a + b, 0);
      w = w.map((v) => v / sw);
    }
  }
  predict(X: number[][]) {
    const total = this.betas.reduce((a, b) => a + b, 0) || 1;
    return X.map((x) => this.trees.reduce((s, t, i) => s + this.betas[i] * t.predictOne(x), 0) / total);
  }
}

// ---------------- KNN ----------------
class KNNReg implements Estimator {
  key = "knn"; name = "KNN"; family: Family = "required";
  hyperparams = { n_neighbors: 5, weights: "distance", metric: "euclidean", algorithm: "brute" };
  private X: number[][] = []; private y: number[] = [];
  fit(X: number[][], y: number[]) { this.X = X; this.y = y; }
  predict(X: number[][]) {
    const k = Math.min(5, this.X.length);
    return X.map((q) => {
      const d = this.X.map((xi, i) => ({
        d: Math.sqrt(xi.reduce((s, v, j) => s + (v - q[j]) ** 2, 0)), y: this.y[i],
      })).sort((a, b) => a.d - b.d).slice(0, k);
      if (d[0].d < 1e-12) return d[0].y;
      const w = d.map((t) => 1 / t.d);
      const sw = w.reduce((a, b) => a + b, 0);
      return d.reduce((s, t, i) => s + t.y * w[i], 0) / sw;
    });
  }
}

// ---------------- SVR (epsilon-insensitive, RBF, dual coordinate descent) ----------------
class SVRReg implements Estimator {
  key = "svr"; name = "SVR"; family: Family = "required";
  hyperparams = { kernel: "rbf", C: 100, epsilon: 0.05, gamma: "scale", max_iter: 200 };
  private Xs: number[][] = []; private beta: number[] = []; private gamma = 1;
  private K(a: number[], b: number[]) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.exp(-this.gamma * s) + 1; // +1 absorbs the bias term
  }
  fit(X: number[][], y: number[]) {
    // subsample for tractability, deterministic stride
    const stride = Math.max(1, Math.ceil(X.length / 700));
    this.Xs = X.filter((_, i) => i % stride === 0);
    const ys = y.filter((_, i) => i % stride === 0);
    const nf = X[0].length;
    let varSum = 0;
    for (let j = 0; j < nf; j++) {
      const col = this.Xs.map((r) => r[j]);
      const m = col.reduce((a, b) => a + b, 0) / col.length;
      varSum += col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length;
    }
    this.gamma = 1 / (nf * (varSum / nf || 1)); // "scale"
    const n = this.Xs.length;
    const Km: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (j < i ? 0 : this.K(this.Xs[i], this.Xs[j])))
    );
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) Km[i][j] = Km[j][i];
    const C = 100, eps = 0.05;
    this.beta = Array(n).fill(0);
    const Qb = Array(n).fill(0);
    for (let it = 0; it < 200; it++) {
      let delta = 0;
      for (let i = 0; i < n; i++) {
        const Qii = Km[i][i];
        const c = Qb[i] - Qii * this.beta[i] - ys[i];
        let b: number;
        if (c + eps < 0) b = -(c + eps) / Qii;
        else if (c - eps > 0) b = -(c - eps) / Qii;
        else b = 0;
        b = Math.max(-C, Math.min(C, b));
        const d = b - this.beta[i];
        if (Math.abs(d) > 1e-12) {
          this.beta[i] = b;
          for (let j = 0; j < n; j++) Qb[j] += d * Km[j][i];
          delta += Math.abs(d);
        }
      }
      if (delta < 1e-6) break;
    }
  }
  predict(X: number[][]) {
    return X.map((q) => this.Xs.reduce((s, xi, i) => (this.beta[i] ? s + this.beta[i] * this.K(xi, q) : s), 0));
  }
}

// ---------------- MLP ----------------
class MLPReg implements Estimator {
  key = "mlp"; name = "MLP Regressor"; family: Family = "required";
  hyperparams = { hidden_layer_sizes: "(64, 32)", activation: "tanh", solver: "adam", learning_rate_init: 0.01, max_iter: 600, random_state: SEED };
  private W: number[][][] = []; private B: number[][] = [];
  private sizes: number[] = [];
  fit(X: number[][], y: number[]) {
    const rng = mulberry32(SEED);
    this.sizes = [X[0].length, 64, 32, 1];
    this.W = []; this.B = [];
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const fanIn = this.sizes[l], fanOut = this.sizes[l + 1];
      const lim = Math.sqrt(6 / (fanIn + fanOut));
      this.W.push(Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => (rng() * 2 - 1) * lim)));
      this.B.push(Array(fanOut).fill(0));
    }
    const mW = this.W.map((w) => w.map((r) => r.map(() => 0)));
    const vW = this.W.map((w) => w.map((r) => r.map(() => 0)));
    const mB = this.B.map((b) => b.map(() => 0));
    const vB = this.B.map((b) => b.map(() => 0));
    const lr = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-8;
    let t = 0;
    for (let epoch = 0; epoch < 600; epoch++) {
      t++;
      const gW = this.W.map((w) => w.map((r) => r.map(() => 0)));
      const gB = this.B.map((b) => b.map(() => 0));
      for (let s = 0; s < X.length; s++) {
        const a: number[][] = [X[s]];
        const z: number[][] = [];
        for (let l = 0; l < this.W.length; l++) {
          const zl = this.W[l].map((row, i) => row.reduce((acc, w, j) => acc + w * a[l][j], 0) + this.B[l][i]);
          z.push(zl);
          a.push(l === this.W.length - 1 ? zl : zl.map(Math.tanh));
        }
        const L = this.W.length;
        let d = [a[L][0] - y[s]];
        for (let l = L - 1; l >= 0; l--) {
          for (let i = 0; i < this.W[l].length; i++) {
            gB[l][i] += d[i];
            for (let j = 0; j < this.W[l][i].length; j++) gW[l][i][j] += d[i] * a[l][j];
          }
          if (l > 0) {
            const dPrev = Array(this.sizes[l]).fill(0);
            for (let j = 0; j < this.sizes[l]; j++) {
              let s2 = 0;
              for (let i = 0; i < this.W[l].length; i++) s2 += this.W[l][i][j] * d[i];
              dPrev[j] = s2 * (1 - Math.tanh(z[l - 1][j]) ** 2);
            }
            d = dPrev;
          }
        }
      }
      const n = X.length;
      for (let l = 0; l < this.W.length; l++) {
        for (let i = 0; i < this.W[l].length; i++) {
          const gb = gB[l][i] / n;
          mB[l][i] = b1 * mB[l][i] + (1 - b1) * gb;
          vB[l][i] = b2 * vB[l][i] + (1 - b2) * gb * gb;
          this.B[l][i] -= lr * (mB[l][i] / (1 - b1 ** t)) / (Math.sqrt(vB[l][i] / (1 - b2 ** t)) + e);
          for (let j = 0; j < this.W[l][i].length; j++) {
            const g = gW[l][i][j] / n;
            mW[l][i][j] = b1 * mW[l][i][j] + (1 - b1) * g;
            vW[l][i][j] = b2 * vW[l][i][j] + (1 - b2) * g * g;
            this.W[l][i][j] -= lr * (mW[l][i][j] / (1 - b1 ** t)) / (Math.sqrt(vW[l][i][j] / (1 - b2 ** t)) + e);
          }
        }
      }
    }
  }
  predict(X: number[][]) {
    return X.map((x) => {
      let a = x;
      for (let l = 0; l < this.W.length; l++) {
        const z = this.W[l].map((row, i) => row.reduce((acc, w, j) => acc + w * a[j], 0) + this.B[l][i]);
        a = l === this.W.length - 1 ? z : z.map(Math.tanh);
      }
      return a[0];
    });
  }
}

// ---------------- Optional baselines ----------------
function solveNormal(X: number[][], y: number[]): number[] {
  const m = X.length, n = X[0].length;
  const XtX = Array.from({ length: n }, () => Array(n).fill(0));
  const Xty = Array(n).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    Xty[j] += X[i][j] * y[i];
    for (let k = 0; k < n; k++) XtX[j][k] += X[i][j] * X[i][k];
  }
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

class PolyReg implements Estimator {
  key: string; name: string; family: Family = "baseline";
  hyperparams: Record<string, string | number>;
  private beta: number[] = [];
  constructor(private deg: number) {
    this.key = deg === 1 ? "linear" : `poly${deg}`;
    this.name = deg === 1 ? "Linear Regression" : `Polynomial Regression (deg ${deg})`;
    this.hyperparams = { degree: deg, fit_intercept: "true" };
  }
  private design(X: number[][]) {
    return X.map((x) => {
      const row = [1];
      for (const v of x) for (let d = 1; d <= this.deg; d++) row.push(v ** d);
      return row;
    });
  }
  fit(X: number[][], y: number[]) { this.beta = solveNormal(this.design(X), y); }
  predict(X: number[][]) { return this.design(X).map((r) => r.reduce((s, v, i) => s + v * this.beta[i], 0)); }
}

class LoessReg implements Estimator {
  key = "loess"; name = "LOESS"; family: Family = "baseline";
  hyperparams = { span: 0.2, degree: 1, kernel: "tricube" };
  private xs: number[] = []; private ys: number[] = []; private bw = 1;
  fit(X: number[][], y: number[]) {
    this.xs = X.map((r) => r[0]); this.ys = y;
    this.bw = (Math.max(...this.xs) - Math.min(...this.xs)) * 0.2 || 1;
  }
  predict(X: number[][]) {
    return X.map((row) => {
      const q = row[0];
      let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
      for (let i = 0; i < this.xs.length; i++) {
        const u = Math.abs(this.xs[i] - q) / this.bw;
        const w = u < 1 ? (1 - u ** 3) ** 3 : 1e-3;
        sw += w; swx += w * this.xs[i]; swy += w * this.ys[i];
        swxx += w * this.xs[i] ** 2; swxy += w * this.xs[i] * this.ys[i];
      }
      const den = sw * swxx - swx * swx || 1e-12;
      return (swxx * swy - swx * swxy) / den + ((sw * swxy - swx * swy) / den) * q;
    });
  }
}

export const REQUIRED_MODEL_KEYS = [
  "xgboost", "adaboost", "decision_tree", "random_forest",
  "extra_trees", "gradient_boosting", "knn", "svr", "mlp",
] as const;

export const BASELINE_MODEL_KEYS = ["linear", "poly3", "loess"] as const;

export function createEstimator(key: string): Estimator {
  switch (key) {
    case "xgboost": return new XGBoostReg();
    case "adaboost": return new AdaBoostReg();
    case "decision_tree": return new DecisionTreeReg();
    case "random_forest": return new ForestReg("random_forest", "Random Forest", 60, 12, false);
    case "extra_trees": return new ForestReg("extra_trees", "Extra Trees", 60, 12, true);
    case "gradient_boosting": return new GradientBoostingReg();
    case "knn": return new KNNReg();
    case "svr": return new SVRReg();
    case "mlp": return new MLPReg();
    case "linear": return new PolyReg(1);
    case "poly3": return new PolyReg(3);
    case "loess": return new LoessReg();
    default: throw new Error(`Unknown model key: ${key}`);
  }
}

export const MODEL_CATALOG = [...REQUIRED_MODEL_KEYS, ...BASELINE_MODEL_KEYS].map((k) => {
  const e = createEstimator(k);
  return { key: e.key, name: e.name, family: e.family, hyperparams: e.hyperparams };
});
