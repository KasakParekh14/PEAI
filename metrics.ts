export interface Metrics {
  r2: number;
  mae: number;
  mse: number;
  rmse: number;
  evs: number;
}

export function computeMetrics(yTrue: number[], yPred: number[]): Metrics {
  const n = yTrue.length;
  const meanT = yTrue.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0, ssTot = 0, mae = 0;
  for (let i = 0; i < n; i++) {
    const e = yTrue[i] - yPred[i];
    ssRes += e * e;
    ssTot += (yTrue[i] - meanT) ** 2;
    mae += Math.abs(e);
  }
  const mse = ssRes / n;
  const errs = yTrue.map((t, i) => t - yPred[i]);
  const meanE = errs.reduce((a, b) => a + b, 0) / n;
  const varE = errs.reduce((a, b) => a + (b - meanE) ** 2, 0) / n;
  const varT = ssTot / n || 1e-12;
  return {
    r2: 1 - ssRes / (ssTot || 1e-12),
    mae: mae / n,
    mse,
    rmse: Math.sqrt(mse),
    evs: 1 - varE / varT,
  };
}
