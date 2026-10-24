# ผู้ช่วยระเบียบบริษัท นาวา ดิจิทัล (หน้าเว็บ)

หน้าเว็บถามตอบที่ทำตามขั้นตอนเดียวกับ skill `/ask` เผยแพร่เป็น artifact ที่ https://claude.ai/artifact/Gd6WApCy8gALtjiLELFgKN

- `index.html` — หน้าเว็บไฟล์เดียว ฝัง chunk ทั้ง 77 ชิ้น (จาก `rag.chunk`) และโค้ดค้นคืนไว้ในตัว
- `retriever.js` — การค้นแบบ hybrid ที่แปลงจาก `rag/retrieve.py`: TF-IDF `char_wb` (2–4) + BM25 บนคำไทย (`Intl.Segmenter`) + คำพ้องชุดเดียวกับ `SYNONYMS` · กรองฉบับ superseded ได้

## ขั้นตอนตอบ
1. ค้นในเบราว์เซอร์ด้วยคำถามเดิม (คำค้นที่ 1)
2. Claude เขียนคำค้นแบบทางการแล้วเรียกเครื่องมือ `search` ของหน้าเว็บ (คำค้นที่ 2)
3. Claude ตอบไม่เกิน 3 ประโยคจาก chunk เท่านั้น อ้าง `[chunk_id]` ทุกประโยค แจ้งเมื่อมีฉบับเดิม เมื่อพบข้อความแทรกคำสั่ง หรือเมื่อไม่พบข้อมูล

ส่วนสร้างคำตอบใช้ capability `sample` ของ claude.ai (ใช้โควตา Claude ของผู้เปิดหน้า ไม่มี API key ในโค้ด) — ถ้าเปิดไฟล์นอก claude.ai หรือไม่อนุญาต หน้าเว็บจะทำงานแบบค้นอย่างเดียว

## ผลตรวจ
`retriever.js` บนชุด dev ให้ผลตรงกับ `python -m rag.evaluate --split dev --method hybrid -k 4`: hit@1 0.962 · hit@4 1.000 · MRR 0.981 · version top-1 active 1.000
