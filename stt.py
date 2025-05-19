# import torch
# import librosa
# from transformers import pipeline
# import numpy as np

# def transcribe_audio(audio_data):
#     try:
#         MODEL_NAME = "./whisper-th-small"
#         device = "cpu"

#         pipe = pipeline(
#             task="automatic-speech-recognition",
#             model=MODEL_NAME,
#             chunk_length_s=30,
#             device=device,
#         )

#         # Ensure the use of input_features instead of deprecated inAputs
#         input_features = np.array(audio_data)

#         # Perform transcription
#         transcriptions = pipe(
#             input_features,
#             batch_size=8,
#             return_timestamps=False,
#             generate_kwargs={"language": "thai", "task": "transcribe"}
#         )["text"]

#         return transcriptions
#     except Exception as e:
#         return str(e)
    
# audio_file = "./audio.mp3"  # Adjust this to your actual audio file path
# audio_array, sampling_rate = librosa.load(audio_file, sr=16000, mono=True)

# transcription_result = transcribe_audio(audio_array)
# print(transcription_result)

from flask import Flask, request, jsonify
import requests
from connected_milvus import *
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from flask_cors import CORS
from langchain.document_loaders import WebBaseLoader 
from langchain_community.document_loaders import PyMuPDFLoader
from werkzeug.utils import secure_filename
from connect_mariadb import *
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes



def load_web_content(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    # แปลง documents ให้เป็น JSON-serializable
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]




def rerank(query, results):
    if not isinstance(query, str):
        raise ValueError("Query should be a string.")
    
    input_texts = [result['text'] for result in results]
    if not all(isinstance(text, str) for text in input_texts):
        raise ValueError("Each result text should be a string.")
    
    # Tokenize input pairs and ensure tensors are on the GPU
    inputs = tokenizer([query] * len(input_texts), input_texts, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
    outputs = model(**inputs)

    logits = outputs.logits
    # print(logits)
    scores = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    # if logits.shape[1] == 1:
    #     scores = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    # else:
    #     scores = torch.nn.functional.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    for i, result in enumerate(results):
        result['score'] = float(scores[i])
    reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return reranked_results

def calculate_similarity_between_context_and_generated(generated_texts):
    """
    Calculate cosine similarity between returned_context and generated_text.

    Args:
        generated_texts: A list of dictionaries with 'text' and 'generated_response' keys.

    Returns:
        A list of generated_texts with updated similarity scores between returned_context and generated_text.
    """
    for result in generated_texts:
        context_vector = get_embedding(result['text'])
        generated_text_vector = get_embedding(result['generated_response'])
        similarity_score = cosine_similarity([context_vector], [generated_text_vector])[0][0]
        result['similarity_score'] = similarity_score

    # Sort the results based on similarity score in descending order
    return sorted(generated_texts, key=lambda x: x['similarity_score'], reverse=True)

@app.route('/searchkeyword', methods=['POST'])
def searchkeyword():
    try:
        # รับข้อมูล JSON จากคำขอ
        data = request.get_json()
        print(data)
        if not data or 'text' not in data:
            return jsonify({"error": 'Key "text" not found in the request data'}), 400

        query_text = str(data['text'])
        print(query_text)
        # เชื่อมต่อกับ Milvus
        connect_to_milvus()
        # ดึงข้อมูล collections จาก Milvus
        collection_name = "knowledge_base"  # ตั้งค่าให้ตรงกับ collection ที่คุณใช้งานใน Milvus
        collections = Collection(name=collection_name)
        collections.load()  # โหลดข้อมูลใหม่
        if not collections:
            return jsonify({"error": "No collections found in the database"}), 404

        # คำนวณเวกเตอร์ของข้อความ query
        query_vector = get_embedding(query_text)
        
        results = search_cosine(collection_name, query_vector)
        # results = rerank(query_text, results)
        # ส่งผลลัพธ์กลับไปยังผู้ใช้
        return jsonify({
            "results": results,
        }), 200

    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้น
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500



@app.route('/insert-pdf', methods=['POST'])
def insertpdf():
    try:
        # Check if the request contains files
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        files = request.files.getlist('file')  # Get list of files

        # Validate required form fields
        cluster = request.form.get('cluster')
        if not cluster:
            return jsonify({"error": "Cluster is required"}), 400
        
        subject = request.form.get('subject')
        if not subject:
            return jsonify({"error": "Subject is required"}), 400
        
        sub_subject = request.form.get('sub_subject')

        collection_name = request.form.get('collection_name')
        if not collection_name:
            return jsonify({"error": "Collection name is required"}), 400

        # Ensure at least one file is selected
        if len(files) == 0 or all(file.filename == '' for file in files):
            return jsonify({"error": "No selected file"}), 400

        print(f"Number of files uploaded: {len(files)}")
        
        # Connect to Milvus
        connect_to_milvus()

        # Create the collection if it doesn't exist
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist. Creating new collection.")
            create_collection(collection_name)

        collection = Collection(name=collection_name)

        # Create index for Cosine Similarity
        create_index_for_collection(collection, metric_type="COSINE")

        # Load the collection into memory
        collection.load()

        total_inserted = 0  # Counter for total documents inserted

        # Process each file in the uploaded files
        for file in files:
            if file and file.filename != '':
                # Save the file to a temporary location
                filename = secure_filename(file.filename)
                file_path = os.path.join("./uploadfile", filename)  # Set a suitable file path
                file.save(file_path)

                # Extract text from the uploaded PDF file
                Docs = read_pdf(file_path)

                # Update metadata for each document
                for doc in Docs:
                    doc["metadata"]['subject'] = subject
                    doc["metadata"]['sub_subject'] = sub_subject
                    doc["metadata"]['cluster'] = cluster

                # Generate embeddings for the texts
                texts = [doc["page_content"] + " " + str(doc["metadata"]) for doc in Docs]
                vectors = [get_embedding(text) for text in texts]  # Generate embeddings for each text
                vectors = [np.array(vector, dtype=np.float32) for vector in vectors]

                # Insert data into the collection
                insert_result = insert_data(collection, texts, vectors, subject)
                total_inserted += len(texts)
                print(f"Inserted {len(texts)} documents from '{filename}' into '{collection_name}' collection.")

        return jsonify({"message": f"Inserted {total_inserted} documents into '{collection_name}' collection."}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500


@app.route('/insert-text', methods=['POST'])
def inserttext():
    try:
        # Check if a source is provided
        source = request.form.get('source')
        if not source:
            return jsonify({"error": "Source is required"}), 400
        
        cluster = request.form.get('cluster')
        if not cluster:
            return jsonify({"error": "cluster is required"}), 400
        
        # Check if a subject is provided
        subject = request.form.get('subject')
        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        # Get the 'sub_subjects' field (optional)
        sub_subjects = request.form.get('sub_subjects')

        
        # Check if content is provided
        content = request.form.get('content')
        if not content:
            return jsonify({"error": "Content is required"}), 400
        
        # Check if a collection name is provided
        collection_name = request.form.get('collection_name')
        if not collection_name:
            return jsonify({"error": "Collection name is required"}), 400

        # Connect to Milvus
        connect_to_milvus()

        # Create the collection if it doesn't exist
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist. Creating new collection.")
            create_collection(collection_name)

        collection = Collection(name=collection_name)

        # Create index for Cosine Similarity
        create_index_for_collection(collection, metric_type="COSINE")

        # Load the collection into memory
        collection.load()

        # Generate the metadata
        metadata = {
            "source": source,
            "subject": subject,
            "cluster": cluster,
            "sub_subjects": sub_subjects,
        }

        # Combine content and metadata for embedding
        combined_text = content + " " + str(metadata)
        # print(combined_text)
        # Generate embeddings for the content
        
        vector = get_embedding(combined_text)  # Generate embedding for the content
        vector = np.array(vector, dtype=np.float32)
        

        # Insert data into the collection
        insert_result = insert_data(collection, [content], [vector], subject)
        
        print(f"Inserted 1 document into '{collection_name}' collection with source '{source}' and subject '{subject}'.")
        
        return jsonify({"message": f"Inserted 1 document into '{collection_name}' collection."}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500
    
@app.route('/insert-url', methods=['POST'])
def inserturl():
    try:
        # Check if a source is provided
        source = request.form.get('source')
        if not source:
            return jsonify({"error": "Source is required"}), 400

        cluster = request.form.get('cluster')
        if not cluster:
            return jsonify({"error": "Cluster is required"}), 400

        # Check if a subject is provided
        subject = request.form.get('subject')
        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        # Get the 'sub_subjects' field (optional)
        sub_subject = request.form.get('sub_subject')

        # Check if content is provided
        doc = load_web_content(source)
        if isinstance(doc, list) and len(doc) > 0:
            # Access the first element
            doc_item = doc[0]
            doc_item["metadata"]['subject'] = subject
            doc_item["metadata"]['sub_subject'] = sub_subject
            doc_item["metadata"]['cluster'] = cluster

            # Combine content and metadata
            content = doc_item["content"]  # Extract content
            metadata = doc_item["metadata"]  # Extract metadata

            if not isinstance(content, str) or not content.strip():
                return jsonify({"error": "Content must be a non-empty string."}), 400

            # Convert metadata to string and combine with content
            combined_text = content + " " + str(metadata)
        else:
            return jsonify({"error": "Invalid content format"}), 400

        # Check if a collection name is provided
        collection_name = request.form.get('collection_name')
        if not collection_name:
            return jsonify({"error": "Collection name is required"}), 400

        # Validate subject
        if not isinstance(subject, str):
            raise ValueError("Invalid subject: must be a string.")

        # Connect to Milvus
        connect_to_milvus()

        # Create the collection if it doesn't exist
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist. Creating new collection.")
            create_collection(collection_name)

        collection = Collection(name=collection_name)

        # Create index for Cosine Similarity
        create_index_for_collection(collection, metric_type="COSINE")

        # Load the collection into memory
        collection.load()

        # Generate embeddings for the combined content
        vector = get_embedding(combined_text)  # Generate embedding for the combined text
        vector = np.array(vector, dtype=np.float32)
        # Insert data into the collection
        insert_result = insert_data(collection, [combined_text], [vector] , subject)

        print(f"Inserted 1 document into '{collection_name}' collection with source '{source}' and subject '{subject}'.")
        return jsonify({"message": f"Inserted 1 document into '{collection_name}' collection."}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500




@app.route('/searchkeyword2', methods=['POST'])
def searchkeyword2():
    try:
        # รับข้อมูล JSON จากคำขอ
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": 'Key "question" not found in the request data'}), 400
        
        # เชื่อมต่อกับ MariaDB
        conn, cur = connecting_mariadb()
        if not conn:
            return jsonify({"error": "Could not connect to MariaDB"}), 500
        
        # ดึงข้อมูล collections จาก MariaDB
        subject = fetch_subject(cur)
        print(subject)
        
        # ดึงข้อความจากคีย์ 'question'
        query_text = data['question']

        # ใช้ URL ที่ได้จาก ngrok
        ngrok_url = "https://56cc-34-125-150-157.ngrok-free.app/api"  

        # เรียก API ด้วยคำสั่ง POST
        response = requests.post(ngrok_url, json=data)  # ส่ง data ที่มีคีย์ 'question'

        # ตรวจสอบการตอบกลับจาก API
        api_response = response.json()
        if 'response' not in api_response:
            return jsonify({"error": 'Key "response" not found in the API response'}), 400

        # ดึงตัวเลขจาก response (เช่น '01') และจับคู่กับข้อความในหมวดหมู่
        response_code = api_response['response'].strip()  # เอาตัวเลขจากการตอบกลับ

        # ดึงข้อมูล collections จาก Milvus
        collections = "knowledge_base"  # ตั้งค่าให้ตรงกับ collection ที่คุณใช้งานใน Milvus
        if not collections:
            return jsonify({"error": "No collections found in the database"}), 404

        # โหลด tokenizer และ model สำหรับการคำนวณเวกเตอร์
        tokenizer, model = load_embedding_model("BAAI/bge-m3")

        # คำนวณเวกเตอร์ของข้อความ query
        query_vector = get_embedding(query_text)

        # เชื่อมต่อกับ Milvus
        connect_to_milvus()

        # ค้นหาผลลัพธ์จาก subject
        results = search_cosine_with_subject(collections, query_vector, subject)

        # ทำการ rerank ผลลัพธ์ที่ได้
        final_results = rerank(query_text, results)

        # ส่งผลลัพธ์กลับไปยังผู้ใช้
        return jsonify({
            "results": final_results,
            "subject": subject  # ส่ง subject ที่ใช้ในการค้นหากลับไป
        }), 200

    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้น
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/update', methods=['POST'])
def update_log():
    data = request.json
    logs = data.get("logData", [])
    print(data)

    try:
        connect_to_milvus()  # Connect to Milvus once
        collection_name = "knowledge_base"

        for log in logs:
            method = log.get("method")
            doc_id = log.get("id")
            new_content = log.get("new_content")


            if method == "edit":
                update_data(collection_name, doc_id, new_content)
                print(f"อัปเดตข้อมูล ID {doc_id} ใน collection '{collection_name}' สำเร็จ")

            elif method == "delete":
                delete_data(collection_name, doc_id)
                print(f"ลบข้อมูล ID {doc_id} ใน collection '{collection_name}' สำเร็จ")

            else:
                print(f"Method '{method}' ไม่ถูกต้อง")

        return jsonify({"message": "Logs updated successfully!"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/insert', methods=['POST'])
def insert_content():
    try:
        data = request.json
        print(data)

        content_data = data.get("contentData", [])
        if not content_data:
            return jsonify({"error": "No contentData provided."}), 400

        # รวมข้อมูลทั้งหมดจาก contentData ลงใน texts
        texts = [item.get("page_content", "") for item in content_data]  # ✅ ดึงแค่เนื้อหา
        texts = [text for text in texts if text.strip()]  # ✅ กรองข้อความที่ไม่ว่าง

        if not texts:
            return jsonify({"error": "No valid page_content found."}), 400

        # ✅ สร้าง embedding สำหรับแต่ละข้อความ
        vectors = [get_embedding(text) for text in texts]
        vectors = [np.array(vector, dtype=np.float32) for vector in vectors]

        # ✅ ดึง subject จาก metadata ถ้ามี
        subject = content_data[0].get("metadata", {}).get("subject", "Unknown")

        collection_name = "knowledge_base"

        # Connect to Milvus
        connect_to_milvus()

        # Create the collection if it doesn't exist
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist. Creating new collection.")
            create_collection(collection_name)

        collection = Collection(name=collection_name)

        # Create index for Cosine Similarity
        create_index_for_collection(collection, metric_type="COSINE")

        # Load the collection into memory
        collection.load()

        # ✅ Insert data into the collection
        insert_result = insert_data(collection, texts, vectors, subject)

        return jsonify({
            "message": f"Inserted {insert_result.insert_count} documents into '{collection_name}' successfully!",
            "status": "success"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    
if __name__ == '__main__':
    app.run(port=5000)

