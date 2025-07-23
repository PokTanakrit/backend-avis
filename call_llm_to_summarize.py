import json
from openai import OpenAI

client = OpenAI(
    api_key='',
    base_url='https://api.opentyphoon.ai/v1'
)

def summarize(text1, text2):
    # Create the chat completion request
    chat_completion = client.chat.completions.create(
        model="typhoon-v2-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"รวมข้อความใหม่ที่อัพเดตและข้อความเดิมเข้าด้วยกัน ข้อความเดิมคือ:\n{text1}\nและข้อความใหม่ที่อัพเดตคือ:\n{text2}\n ข้อความใหม่ที่ให้ไปเป็นข้อมูลที่ต้องอัพเดทของข้อความเก่า กรุณารวมทั้งสองข้อความนี้เข้าด้วยกันอย่างมีความหมายและไม่ทิ้งข้อมูลสำคัญจากทั้งสองข้อความ (ถ้าข้อความใหม่ที่อัพเดตไม่ได้เกี่ยวข้องกับข้อความเก่าให้ตอบกลับมาว่า haverelation  no)  รูปแบบการตอบเป็นjsonformat 'haverelation' : yes or no 'summarize': ข้อความที่อัพเดด"}
        ],
        max_tokens=1000,
        temperature=0.5,
    )
    
    # Return the generated text
    return chat_completion.choices[0].message.content

# text1 = ""

# text2 = ""
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # เพิ่ม CORS
# import json
# import re
# from runllm import *

# app = Flask(__name__)
# CORS(app)  # เปิดใช้งาน CORS สำหรับทุกโดเมน

# def clean_json_string(json_str):
#     """ลบอักขระควบคุมที่อาจทำให้ JSON โหลดไม่สำเร็จ"""
#     json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
#     return json_str.strip()

# def textgenerate_with_local_model(question, relevant_data, max_tokens=150, temperature=0.5):
#     if not relevant_data.strip():
#         return "ไม่ทราบครับ"

#     messages = [
#         {"role": "system", "content": "คุณคือผู้ช่วยอัจฉริยะที่มีความสามารถในการตอบคำถามได้อย่างถูกต้องและกระชับ โดยใช้ข้อมูลที่เกี่ยวข้องที่ให้มาเท่านั้น ไม่ใช้ความรู้ทั่วไปของโมเดล และใช้ภาษาไทยพร้อมใส่หางเสียงนะครับ "},
#         {"role": "system", "content": "หากไม่มีข้อมูลที่เพียงพอหรือข้อมูลไม่ครอบคลุมคำถาม ให้ตอบว่าสั้นๆว่า" + "ไม่ทราบ"},
#         {"role": "system", "content": f'ข้อมูลที่คุณจะใช้ในการตอบคำถามนี้คือ: {relevant_data} กรุณาตอบคำถามโดยยึดข้อมูลนี้ และตอบให้กระชับที่สุดเท่าที่จะทำได้  พูดให้เป็นะรรมชาติ  และหลีกเลี่ยงการตอบซ้ำ รวมถึงข้อมูลที่ไม่เกี่ยวข้องครับ คำตอบที่ต้องการมีความยาวไม่เกิน 50 คำ ถ้าคำตอบยาวเกินไปให้จบประโยค'},
#         {"role": "user", "content": f'โปรดให้ข้อมูลเกี่ยวกับ: {question}'}
#     ]

#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     eos_token_id = tokenizer.eos_token_id

#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=max_tokens,
#         eos_token_id=eos_token_id,
#         do_sample=False,  # เปิดให้สุ่มข้อความ
#         temperature=0.7,
#         top_p=0.9,  # ลดการสุ่มของคำตอบ
#     )

#     response = outputs[0][input_ids.shape[-1]:]
#     result = tokenizer.decode(response, skip_special_tokens=True).strip()
#     if result == "คุณคือผู้ช่วยอัจฉริยะที่มีความสามารถในการตอบคำถามได้อย่างถูกต้องและกระชับ":
#         result = "ไม่ทราบ"
#     return result if result else "ไม่ทราบครับ"

