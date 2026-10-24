"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import numpy as np
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunk import chunk_docs

# คำพ้องที่พบว่าทำให้ค้นพลาดในชุด dev (W4): คำถามปากเปล่าใช้ศัพท์ไม่ตรงกับเอกสาร
# "เสียบชาร์จรถอีวีที่ออฟฟิศได้นานแค่ไหน" (FA-01) ไม่เจอเพราะเอกสารใช้ "รถยนต์ไฟฟ้า"/"จุดชาร์จ"
SYNONYMS = {
    "อีวี": "รถยนต์ไฟฟ้า",
    "เสียบชาร์จ": "จุดชาร์จ",
}


def _index_text(c: dict) -> str:
    return f"{c['title']} {c['section']} {c['text']}"


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def _expand_query(query: str) -> str:
    extra = [formal for colloquial, formal in SYNONYMS.items() if colloquial in query]
    return f"{query} {' '.join(extra)}" if extra else query


def _normalize(scores: np.ndarray) -> np.ndarray:
    top = scores.max() if len(scores) else 0
    return scores / top if top > 0 else scores


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method not in ("tfidf", "hybrid"):
            raise ValueError(f"unknown method: {method}")
        self.chunks = chunks if chunks is not None else chunk_docs()
        self.method = method
        texts = [_index_text(c) for c in self.chunks]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(texts)
        if method == "hybrid":
            self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        q = _expand_query(query) if self.method == "hybrid" else query
        char_scores = cosine_similarity(self.vectorizer.transform([q]), self.matrix)[0]
        if self.method == "hybrid":
            bm25_scores = np.array(self.bm25.get_scores(_tokenize(q)))
            scores = 0.7 * _normalize(char_scores) + 0.3 * _normalize(bm25_scores)
        else:
            scores = char_scores
        results = [{**c, "score": float(s)} for c, s in zip(self.chunks, scores)]
        if not include_superseded:
            results = [r for r in results if r["status"] == "active"]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        for r in res:
            print(f"{r['score']:.3f}  {r['chunk_id']}  [{r['status']}]  {r['section']}")
            print(f"    {r['text'][:120]}")


if __name__ == "__main__":
    main()
