// Distributions drawn by hand or measured from traces: a weight for each bin of a fixed ladder.
// A bin's weight is spread evenly on a log scale, over its whole numbers where the values are turns.
export const BINS = Object.freeze({
  token: Object.freeze({edges: Object.freeze([64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]), whole: false}),
  gap: Object.freeze({edges: Object.freeze([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 360, 720, 1440, 4320]), whole: false}),
  // Whole numbers: bin k holds edges[k] up to, not including, edges[k + 1].
  length: Object.freeze({edges: Object.freeze([1, 2, 3, 4, 5, 6, 8, 11, 15, 20, 26, 34, 45, 60, 80, 110, 161]), whole: true}),
  need: Object.freeze({edges: Object.freeze([1, 2, 3, 4, 5, 6, 8, 11, 16, 21, 31, 51, 161]), whole: true})
});
// The settings that hold a drawn distribution, and the ladder each one uses. Null means its law is used instead.
export const DRAWN = Object.freeze({tokenDrawn: 'token', gapDrawn: 'gap', lengthDrawn: 'length', inputNeedDrawn: 'need', retrievalNeedDrawn: 'need', outputNeedDrawn: 'need'});

export const binCount = kind => BINS[kind].edges.length - 1;
export function checkDrawn(name, weights) {
  if (weights === null) return;
  const n = binCount(DRAWN[name]);
  if (!Array.isArray(weights) || weights.length !== n || !weights.every(w => Number.isFinite(w) && w >= 0)) throw new RangeError(`${name} must be ${n} weights of at least 0`);
  if (!(weights.reduce((a, b) => a + b, 0) > 0)) throw new RangeError(`${name} must have some weight`);
}
// The chance of at most `x`, the inverse of that, and the mean.
export function histogram(kind, weights) {
  const {edges, whole} = BINS[kind], total = weights.reduce((a, b) => a + b, 0), share = weights.map(w => w / total), below = [0];
  for (const s of share) below.push(below.at(-1) + s);
  const last = share.findLastIndex(s => s > 0);
  const within = (k, x) => Math.log((whole ? Math.floor(x) + 1 : x) / edges[k]) / Math.log(edges[k + 1] / edges[k]);
  const cdf = x => {
    if (x < edges[0]) return 0;
    if (x >= edges.at(-1)) return 1;
    let k = 0; while (x >= edges[k + 1]) k++;
    return below[k] + share[k] * within(k, x);
  };
  const quantile = q => {
    const u = Math.max(0, Math.min(1, q)); let k = 0; while (k < last && below[k + 1] <= u) k++;
    while (share[k] === 0 && k < last) k++;
    const f = Math.max(0, Math.min(1, (u - below[k]) / share[k]));
    const value = edges[k] * (edges[k + 1] / edges[k]) ** f;
    return whole ? Math.min(edges[k + 1] - 1, Math.floor(value + 1e-9)) : value;
  };
  const mean = share.reduce((sum, s, k) => sum + s * ((edges[k + 1] - edges[k]) / Math.log(edges[k + 1] / edges[k]) - (whole ? .5 : 0)), 0);
  return Object.freeze({cdf, quantile, mean, share: Object.freeze(share)});
}
// The bin weights of any distribution, from its chance of at most `x`. What lies outside the ladder goes to its ends.
export function weightsOf(kind, cdf) {
  const {edges, whole} = BINS[kind], n = edges.length - 1, at = k => cdf(whole ? edges[k] - 1 : edges[k]);
  return Array.from({length: n}, (_, k) => Math.max(0, (k === n - 1 ? 1 : at(k + 1)) - (k === 0 ? 0 : at(k))));
}
// The bin weights of observed values.
export function weightsFrom(kind, values) {
  const {edges} = BINS[kind], n = edges.length - 1, counts = new Array(n).fill(0);
  for (const v of values) { let k = 0; while (k < n - 1 && v >= edges[k + 1]) k++; counts[k]++; }
  return counts;
}
