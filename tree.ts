// Decision tree regressor (CART, MSE criterion). Generic over n features.

export interface TreeOptions {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures?: number; // number of features considered per split
  splitter?: "best" | "random"; // random => Extra-Trees style
  rng?: () => number;
  lambda?: number; // L2 leaf regularization (XGBoost style)
}

interface Node {
  leaf: boolean;
  value?: number;
  feature?: number;
  threshold?: number;
  left?: Node;
  right?: Node;
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class DecisionTree {
  private root: Node | null = null;
  constructor(private opts: TreeOptions) {}

  fit(X: number[][], y: number[], sampleWeight?: number[], grad?: { g: number[]; h: number[] }) {
    const idx = X.map((_, i) => i);
    const w = sampleWeight ?? X.map(() => 1);
    this.root = this.build(X, y, idx, w, grad, 0);
  }

  private leafValue(y: number[], idx: number[], w: number[], grad?: { g: number[]; h: number[] }): number {
    if (grad) {
      let g = 0, h = 0;
      for (const i of idx) { g += grad.g[i]; h += grad.h[i]; }
      return -g / (h + (this.opts.lambda ?? 1));
    }
    let s = 0, sw = 0;
    for (const i of idx) { s += y[i] * w[i]; sw += w[i]; }
    return sw ? s / sw : 0;
  }

  private build(X: number[][], y: number[], idx: number[], w: number[], grad: { g: number[]; h: number[] } | undefined, depth: number): Node {
    const value = this.leafValue(y, idx, w, grad);
    if (depth >= this.opts.maxDepth || idx.length < this.opts.minSamplesSplit) {
      return { leaf: true, value };
    }
    const nFeat = X[0].length;
    const rng = this.opts.rng ?? Math.random;
    let features = Array.from({ length: nFeat }, (_, i) => i);
    const mf = this.opts.maxFeatures ?? nFeat;
    if (mf < nFeat) {
      features = features.sort(() => rng() - 0.5).slice(0, Math.max(1, mf));
    }

    let bestGain = 0, bestFeat = -1, bestThr = 0;
    const parentImpurity = this.impurity(y, idx, w, grad);

    for (const f of features) {
      const vals = idx.map((i) => X[i][f]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      if (hi - lo < 1e-12) continue;
      let thresholds: number[];
      if (this.opts.splitter === "random") {
        thresholds = [lo + rng() * (hi - lo)];
      } else {
        const sorted = [...new Set(vals)].sort((a, b) => a - b);
        const step = Math.max(1, Math.floor(sorted.length / 64));
        thresholds = [];
        for (let i = step; i < sorted.length; i += step) thresholds.push((sorted[i - 1] + sorted[i]) / 2);
        if (!thresholds.length) thresholds = [(lo + hi) / 2];
      }
      for (const thr of thresholds) {
        const L: number[] = [], R: number[] = [];
        for (const i of idx) (X[i][f] <= thr ? L : R).push(i);
        if (L.length < this.opts.minSamplesLeaf || R.length < this.opts.minSamplesLeaf) continue;
        const gain = parentImpurity - (this.impurity(y, L, w, grad) + this.impurity(y, R, w, grad));
        if (gain > bestGain) { bestGain = gain; bestFeat = f; bestThr = thr; }
      }
    }

    if (bestFeat < 0) return { leaf: true, value };
    const L: number[] = [], R: number[] = [];
    for (const i of idx) (X[i][bestFeat] <= bestThr ? L : R).push(i);
    return {
      leaf: false,
      feature: bestFeat,
      threshold: bestThr,
      left: this.build(X, y, L, w, grad, depth + 1),
      right: this.build(X, y, R, w, grad, depth + 1),
    };
  }

  // Weighted SSE (or negative gain proxy for gradient boosting)
  private impurity(y: number[], idx: number[], w: number[], grad?: { g: number[]; h: number[] }): number {
    if (!idx.length) return 0;
    if (grad) {
      let g = 0, h = 0;
      for (const i of idx) { g += grad.g[i]; h += grad.h[i]; }
      return -(g * g) / (h + (this.opts.lambda ?? 1));
    }
    let s = 0, sw = 0;
    for (const i of idx) { s += y[i] * w[i]; sw += w[i]; }
    const mean = sw ? s / sw : 0;
    let sse = 0;
    for (const i of idx) sse += w[i] * (y[i] - mean) ** 2;
    return sse;
  }

  predictOne(x: number[]): number {
    let n = this.root!;
    while (!n.leaf) n = x[n.feature!] <= n.threshold! ? n.left! : n.right!;
    return n.value!;
  }
  predict(X: number[][]): number[] { return X.map((x) => this.predictOne(x)); }
}
