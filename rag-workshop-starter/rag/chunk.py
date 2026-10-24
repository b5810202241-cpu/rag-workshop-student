"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600


def _split_by_lines(text: str, max_chars: int) -> list[str]:
    """ตัดข้อความยาวตามบรรทัด ไม่ให้แต่ละก้อนเกิน max_chars ตัวอักษร"""
    parts, current = [], ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    if docs is None:
        docs = load_docs()

    chunks = []
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k != "body"}
        # เติม "\n" นำหน้าเพื่อให้หัวข้อ "## " ที่อยู่ต้นเอกสารถูกตัดเหมือนหัวข้ออื่น ๆ
        sections = ("\n" + doc["body"]).split("\n## ")[1:]
        seq = 0
        for raw in sections:
            section, _, rest = raw.partition("\n")
            text = rest.strip()
            if not text:
                continue
            for part in _split_by_lines(text, max_chars):
                chunks.append({
                    **meta,
                    "chunk_id": f"{doc['doc_id']}#{seq}",
                    "section": section.strip(),
                    "text": part,
                })
                seq += 1
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
