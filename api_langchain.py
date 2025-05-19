# from flask import Flask, request, jsonify
# import requests
# from connected_milvus import connect_to_milvus, get_embedding, search_cosine
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np
# from flask_cors import CORS



# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
# llm_url = "https://1438-202-44-40-186.ngrok-free.app/textgenerate"

# # Load the Hugging Face model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
# model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to("cuda:0")

# def rerank(query, results):
#     if not isinstance(query, str):
#         raise ValueError("Query should be a string.")
    
#     input_texts = [result['text'] for result in results]
#     if not all(isinstance(text, str) for text in input_texts):
#         raise ValueError("Each result text should be a string.")
    
#     # Tokenize input pairs and ensure tensors are on the GPU
#     inputs = tokenizer([query] * len(input_texts), input_texts, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
#     outputs = model(**inputs)

#     logits = outputs.logits
#     if logits.shape[1] == 1:
#         scores = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
#     else:
#         scores = torch.nn.functional.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

#     for i, result in enumerate(results):
#         result['score'] = float(scores[i])
#     reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
#     return reranked_results

# def calculate_similarity_between_context_and_generated(generated_texts):
#     """
#     Calculate cosine similarity between returned_context and generated_text.

#     Args:
#         generated_texts: A list of dictionaries with 'text' and 'generated_response' keys.

#     Returns:
#         A list of generated_texts with updated similarity scores between returned_context and generated_text.
#     """
#     for result in generated_texts:
#         context_vector = get_embedding(result['text'])
#         generated_text_vector = get_embedding(result['generated_response'])
#         similarity_score = cosine_similarity([context_vector], [generated_text_vector])[0][0]
#         result['similarity_score'] = similarity_score

#     # Sort the results based on similarity score in descending order
#     return sorted(generated_texts, key=lambda x: x['similarity_score'], reverse=True)

# @app.route('/searchindex3', methods=['POST'])
# def searchretruval():
#     try:
#         connect_to_milvus()
        
#         data = request.get_json()
#         if 'text' not in data:
#             return jsonify({"error": 'Key "text" not found in the data'}), 400

#         text = data['text']
#         print(f"Received text for search (Cosine): {text}")

#         query_vector = get_embedding(text)

#         results = search_cosine("course_information", query_vector)
        
#         if results:
#             reranked_results = rerank(text, results)
#             generate_text = []
            
#             for result in reranked_results[:3]:  # Take top 5 reranked results
#                 relevant_data = result['text']
#                 print(f"relevant_data: {relevant_data}")
#                 response = requests.post(llm_url, json={"input_text": text, "context": relevant_data})
#                 if response.status_code == 200:
#                     generated_text = response.json().get("generated_text", "")
#                     returned_context = response.json().get("context", "")
                    
#                     generate_text.append({
#                         "text": returned_context,
#                         "generated_response": generated_text
#                     })
#                 else:
#                     print("Error generating response from LLM")
            
#             # # Calculate similarity between returned_context and generated_text
#             # final_reranked_results = calculate_similarity_between_context_and_generated(generate_text)

#             return jsonify({"responses": generate_text[0]}), 200
#         else:
#             return jsonify({"message": "No results found."}), 404

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": "An error occurred while processing the request."}), 500

# if __name__ == '__main__':
#     app.run(port=3000)

import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

# โหลด stopwords ภาษาไทย
thai_stopwords = set(thai_stopwords())


import re
from pythainlp.tokenize import word_tokenize

def clean_text(text):
    """ ลบ \n, HTML Tags, ตัวเลขข้อ และอักขระพิเศษ แต่ยังคงรักษาคำไว้ """
    text = re.sub(r'\s+', ' ', text)  # ลบช่องว่างเกิน 1 ช่องและ \n
    text = re.sub(r'[\'"]', '', text)  # ลบเครื่องหมาย " และ '

    # text = re.sub(r'<.*?>', '', text)  # ลบ HTML tags เช่น <p>...</p>
    # text = re.sub(r'\d+\.\d+|\d+', '', text)  # ลบตัวเลขข้อ เช่น 5.1, 7.2, 2564
    # text = re.sub(r'[^\u0E00-\u0E7F\w\s]', '', text)  # ลบอักขระพิเศษ ยกเว้นตัวอักษรไทยและตัวอักษรทั่วไป

    return text.strip()  # ลบช่องว่างหัวท้าย

# 🔹 ทดสอบทำความสะอาดข้อความ
text = """
"2 \n \n5. วัตถุประสงค์ \n5.1  เพื่อผลิตบัณฑิตที่มีความรู้ ความสามารถในด้านวิทยาการคอมพิวเตอร์ทั้งในทฤษฎีและปฏิบัติ โดย\nเพิ่มขีดความสามารถทางด้านการสื่อสารด้วยภาษาอังกฤษอย่างมีประสิทธิภาพ  \n5.2  เพื่อผลิตบัณฑิตที่สามารถแข่งขันในตลาดแรงงานระดับประเทศ ระดับภูมิภาคและระดับนานาชาติ  \n5.3  เพื่อขยายโอกาสในการศึกษาต่อต่างประเทศให้กับบัณฑิต \n5.4  เพื่อผลิตบัณฑิตที่มีคุณธรรม จรรยาบรรณ และเจตคติที่ดีต่อวิชาชีพ  \n \n6. หลักสูตร \n \nใช้หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาวิทยาการคอมพิวเตอร์ (หลักสูตรปรับปรุง พ.ศ. 2564) โดย\nจัดการเรียนการสอนเป็นแบบสองภาษา \n \n7. คุณสมบัติของผู้เข้าศึกษาต่อ \n7.1 รับผู้ส าเร็จการศึกษาในระดับมัธยมศึกษาตอนปลาย (ม.6) หรือ  \n7.2 ส าเร็จการศึกษาระดับประกาศนียบัตรวิชาชีพ (ปวช.) จากสถาบันการศึกษาซึ่งกระทรวงศึกษาธิการ\nรับรอง \n7.3 มีคุณสมบัติอื่นๆตามระเบียบมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ ว่าด้วยการศึกษา\nระดับปริญญาบัณฑิต \n \n8. การจัดการเรียนการสอน \n \nการจัดการเรียนการสอนจะใช้ทั้งภาษาไทยและภาษาอังกฤษ  โดยก าหนดให้สัดส่วนของการเรียนเป็น\nภาษาอังกฤษตลอดหลักสูตรรวมแล้วไม่น้อยกว่าร้อยละ 50 (ไม่น้อยกว่า 64 หน่วยกิต) และมีสัดส่วนของการ\nเรียนเป็นภาษาอังกฤษในหมวดวิชาเฉพาะรวมแล้วไม่น้อยกว่าร้อยละ 40 (ไม่น้อยกว่า 36.4 หน่วยกิต) \n \nนอกจากนี้เพื่อเป็นการประกันคุณภาพนักศึกษาด้านการใช้ภาษาอังกฤษ นักศึกษาที่จะจบการศึกษา\nได้จะต้องมีผลสอบภาษาอังกฤษ TOEIC (หรือเทียบเท่า) ไม่ต่ ากว่า 550 คะแนน  \n \n9. วิธีการสอบคัดเลือก \n \nเป็นไปตามระเบียบการคัดเลือกบุคคลเข้าศึกษาในสถาบันอุดมศึกษา (Admission) และระเบียบการ\nคัดเลือกบุคคลด้วยระบบโควต้า
"""
print(clean_text(text))
