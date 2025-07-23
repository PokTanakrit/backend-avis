import json
from openai import OpenAI

client = OpenAI(
    api_key='',
    base_url='https://api.opentyphoon.ai/v1'
)

def textgenerate(question, relevant_data):
    chat_completion = client.chat.completions.create(
        model="typhoon-v2-70b-instruct",
        messages=[
            {"role": "system", "content": "คุณคือผู้ช่วยอัจฉริยะที่มีความสามารถในการตอบคำถามได้อย่างแม่นยำ โดยใช้ข้อมูลที่ให้มาเท่านั้น อย่าใช้ผความรู้ทั่วไปของโมเดล และตอบเป็นภาษาไทย พร้อมใส่หางเสียงนะครับเท่านั้น"},
            {"role": "system", "content": "หากไม่มีข้อมูลที่เพียงพอหรือข้อมูลไม่ครอบคลุมคำถาม ให้ตอบว่า 'ไม่พบข้อมูล กรุณาถามอีกครั้ง'"},
            {"role": "system", "content": f'ข้อมูลที่คุณจะใช้ในการตอบคำถามนี้คือ: {json.dumps(relevant_data, ensure_ascii=False)} กรุณาตอบคำถามโดยยึดข้อมูลนี้ ตอบให้เข้าใจง่าย และหลีกเลี่ยงการตอบซ้ำ ความยาวของคำตอบไม่น้อยกว่า 30 คำ และจบประโยค'},
            {"role": "user", "content": f'โปรดให้ข้อมูลเกี่ยวกับ: {question}'}
        ],
        max_tokens=300,
        temperature=0.3,
    )
    
    return chat_completion.choices[0].message.content

