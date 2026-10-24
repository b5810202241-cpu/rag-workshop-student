// Browser port of rag/retrieve.py (method="hybrid"): char_wb TF-IDF (2-4) + BM25 on Thai word tokens + synonym expansion
function makeRetriever(chunks) {
  const SYNONYMS = { "อีวี": "รถยนต์ไฟฟ้า", "เสียบชาร์จ": "จุดชาร์จ" };
  const seg = new Intl.Segmenter("th", { granularity: "word" });
  const tokenize = (t) => [...seg.segment(t)].filter((s) => s.segment.trim()).map((s) => s.segment.toLowerCase());
  const charGrams = (text) => {
    const out = new Map();
    for (const word of text.toLowerCase().split(/\s+/).filter(Boolean)) {
      const w = " " + word + " ";
      for (let n = 2; n <= 4; n++) {
        let off = 0;
        out.set(w.slice(off, off + n), (out.get(w.slice(off, off + n)) || 0) + 1);
        while (off + n < w.length) {
          off++;
          const g = w.slice(off, off + n);
          out.set(g, (out.get(g) || 0) + 1);
        }
        if (off === 0) break;
      }
    }
    return out;
  };
  const texts = chunks.map((c) => `${c.title} ${c.section} ${c.text}`);
  const N = chunks.length;

  // TF-IDF, sklearn defaults: smooth idf, l2 norm
  const docGrams = texts.map(charGrams);
  const df = new Map();
  docGrams.forEach((g) => g.forEach((_, k) => df.set(k, (df.get(k) || 0) + 1)));
  const idf = new Map();
  df.forEach((d, k) => idf.set(k, Math.log((1 + N) / (1 + d)) + 1));
  const vec = (grams) => {
    const v = new Map();
    let norm = 0;
    grams.forEach((tf, k) => {
      if (!idf.has(k)) return;
      const x = tf * idf.get(k);
      v.set(k, x);
      norm += x * x;
    });
    norm = Math.sqrt(norm) || 1;
    v.forEach((x, k) => v.set(k, x / norm));
    return v;
  };
  const docVecs = docGrams.map(vec);

  // BM25Okapi, rank_bm25 defaults
  const k1 = 1.5, b = 0.75, eps = 0.25;
  const docToks = texts.map(tokenize);
  const tfs = docToks.map((ts) => ts.reduce((m, t) => m.set(t, (m.get(t) || 0) + 1), new Map()));
  const avgdl = docToks.reduce((s, t) => s + t.length, 0) / N;
  const tdf = new Map();
  tfs.forEach((m) => m.forEach((_, t) => tdf.set(t, (tdf.get(t) || 0) + 1)));
  const bidf = new Map();
  let idfSum = 0;
  const neg = [];
  tdf.forEach((d, t) => {
    const x = Math.log(N - d + 0.5) - Math.log(d + 0.5);
    bidf.set(t, x);
    idfSum += x;
    if (x < 0) neg.push(t);
  });
  const floor = eps * (idfSum / tdf.size);
  neg.forEach((t) => bidf.set(t, floor));

  const expand = (q) => {
    const extra = Object.keys(SYNONYMS).filter((k) => q.includes(k)).map((k) => SYNONYMS[k]);
    return extra.length ? q + " " + extra.join(" ") : q;
  };
  const normalize = (xs) => {
    const top = Math.max(...xs);
    return top > 0 ? xs.map((x) => x / top) : xs;
  };

  function search(query, k = 4, includeSuperseded = false) {
    const q = expand(query);
    const qv = vec(charGrams(q));
    const charScores = docVecs.map((dv) => {
      let s = 0;
      qv.forEach((x, g) => { if (dv.has(g)) s += x * dv.get(g); });
      return s;
    });
    const qt = tokenize(q);
    const bmScores = tfs.map((m, i) => {
      const dl = docToks[i].length;
      let s = 0;
      for (const t of qt) {
        const f = m.get(t) || 0;
        if (f) s += (bidf.get(t) || 0) * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * dl) / avgdl));
      }
      return s;
    });
    const nc = normalize(charScores), nb = normalize(bmScores);
    return chunks
      .map((c, i) => ({ ...c, score: 0.7 * nc[i] + 0.3 * nb[i] }))
      .filter((r) => includeSuperseded || r.status === "active")
      .sort((a, z) => z.score - a.score)
      .slice(0, k);
  }
  return { search };
}
if (typeof module !== "undefined") module.exports = { makeRetriever };
