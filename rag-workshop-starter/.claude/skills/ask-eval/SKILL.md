---
name: ask-eval
description: รันชุดคำถาม data/eval/questions.csv ผ่านขั้นตอนของ /ask ทีละข้อแบบไม่ดูเฉลย แล้วตรวจผลด้วย rag.check_answers
---

อาร์กิวเมนต์: `$ARGUMENTS` คือ `<split> <จำนวนข้อ>` เช่น `dev 10`

ทำตามขั้นตอนนี้:

1. อ่าน `data/eval/questions.csv` ด้วย Python (เช่น `csv.DictReader`)
   กรองแถวที่ `split` ตรงกับที่ระบุ แล้วหยิบมาตามจำนวนข้อที่ระบุ
   **อ่านเฉพาะคอลัมน์ `qid` และ `question` เท่านั้น** ห้ามดูหรืออ้างอิงคอลัมน์ `gold_doc_ids`, `answer_keywords`, `must_not` เด็ดขาด
   (เป็นเฉลยที่ใช้ตรวจผลภายหลังเท่านั้น ไม่ใช่ข้อมูลที่ใช้ตอบ)

2. ทำทีละข้อตามลำดับ qid โดยทำตามขั้นตอนทั้งหมดของ skill `/ask` กับค่า `question` ของแถวนั้น
   (เขียนคำค้น 2 แบบ, รัน `python -m rag.retrieve ... --json`, ตอบไม่เกิน 3 ประโยคพร้อม `[chunk_id]`, ฯลฯ)

3. เก็บผลแต่ละข้อเป็น object `{"qid": "...", "answer": "...", "citations": ["<chunk_id>", ...]}`
   โดย `citations` คือ chunk_id ทั้งหมดที่อ้างถึงในคำตอบของข้อนั้น (ไม่ใช่ chunk_id ที่ค้นเจอทั้งหมด)

4. เมื่อทำครบทุกข้อแล้ว เขียนไฟล์ `out/answers_<split>.jsonl` บรรทัดละหนึ่ง object ตามข้อ 3
   (สร้างโฟลเดอร์ `out/` ถ้ายังไม่มี)

5. รัน `python -m rag.check_answers out/answers_<split>.jsonl`

6. อ่านผลลัพธ์แล้วสรุปเป็นภาษาไทย: จำนวนข้อที่ผ่าน/ไม่ผ่านรวม และรายชื่อ qid ที่ไม่ผ่านพร้อมเหตุผล (keywords/citation/safe)
