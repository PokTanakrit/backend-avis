# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # โหลดโมเดล + optimizer performance
# model = AutoModelForCausalLM.from_pretrained(
#     "D:\quanti-typhoon\llama3.2-typhoon2-3b",
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
#     trust_remote_code=True
# )
# model = torch.compile(model)  # ✅ ลด Latency

# tokenizer = AutoTokenizer.from_pretrained(
#     "D:\quanti-typhoon\llama3.2-typhoon2-3b",
#     use_fast=True  # ✅ Rust tokenizer
# )

# def textgenerate(question, relevant_data):
#     # สร้าง prompt ให้โมเดลตอบโดยอ้างอิงจาก relevant_data เท่านั้น
#     messages = [
#         {"role": "system", "content": (
#             "คุณเป็นผู้ช่วยอัจฉริยะที่สามารถตอบคำถามได้อย่างถูกต้อง กระชับ และมีอารมณ์ขันเล็กน้อย "
#             "คุณจะตอบโดยยึดเฉพาะข้อมูลที่ให้ไว้ในส่วนที่เกี่ยวข้องเท่านั้น "
#             "หากไม่มีข้อมูลที่เพียงพอหรือข้อมูลไม่ครอบคลุมคำถาม ให้ตอบว่า 'ไม่ทราบ' "
#             "โปรดใช้ภาษาไทยและใส่หางเสียง ครับ"
#         )},
#         {"role": "user", "content": f"ข้อมูลที่เกี่ยวข้อง: {relevant_data}"},
#         {"role": "user", "content": f"ช่วยตอบคำถามนี้: {question}"}
#     ]

#     model_inputs = tokenizer([messages], return_tensors="pt").to(device)

#     # Generate response with optimizations
#     with torch.inference_mode():  # ✅ ปิด Gradient Tracking
#         generated_ids = model.generate(
#             model_inputs.input_ids,
#             max_new_tokens=256,
#             temperature=0.5,
#             use_cache=True  # ✅ ใช้ KV Cache
#         )

#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response

# result = textgenerate()

from datetime import datetime

def convert_thai_date(thai_date_str):
    # แผนที่เดือนไทยเป็นตัวเลข
    months_thai = {
        "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", "เมษายน": "04",
        "พฤษภาคม": "05", "มิถุนายน": "06", "กรกฎาคม": "07", "สิงหาคม": "08",
        "กันยายน": "09", "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12"
    }
    
    # แยกวันที่ เดือน และปี
    day, month_thai, year_thai = thai_date_str.split()

    # แปลงปีพุทธศักราชเป็นคริสต์ศักราช
    year_ad = int(year_thai) - 543

    # ใช้แผนที่เดือนเพื่อแปลงชื่อเดือนเป็นตัวเลข
    month_ad = months_thai[month_thai]

    # สร้างวันที่ในรูปแบบสากล (YYYY-MM-DD)
    return f"{year_ad}-{month_ad}-{int(day):02d}"

# ตัวอย่างการใช้งาน
thai_date = "18 มีนาคม 2568"
converted_date = convert_thai_date(thai_date)
print(converted_date)  # ผลลัพธ์จะเป็น "2025-03-18"
current_date = datetime.now().strftime('%Y-%m-%d')
print(current_date)