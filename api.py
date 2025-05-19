from flask import Flask, request, jsonify
import requests
from connected_milvus import *
import numpy as np
from flask_cors import CORS
from connect_mariadb import *
from call_llm import *
import os
import mariadb
import sys
import hashlib
import datetime
import pytz
import re
import logging
import re
from datetime import datetime, timedelta



app = Flask(__name__)
CORS(app)

def connecting_mariadb():
    conn_params = {
        'user': "root",
        'password': "root",
        'host': "localhost",
        'port': 3306,
        'database': "avismariadb"
    }
    try:
        connection = mariadb.connect(**conn_params)
        cursor = connection.cursor(dictionary=True)  # คืนค่าเป็น dict
        print("Connected to MariaDB successfully.")
        return connection, cursor
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        sys.exit(1)



def log_action(method, detail, admin_username):
    conn, cursor = connecting_mariadb()
    try:
        # กำหนด timezone เป็น UTC+7
        bangkok_tz = pytz.timezone('Asia/Bangkok')
        log_date = datetime.now(bangkok_tz).strftime('%Y-%m-%d %H:%M:%S ')

        print(log_date)

        cursor.execute("SELECT admin_id FROM admin WHERE admin_username = %s", (admin_username,))
        user = cursor.fetchone()
        admin_id = user["admin_id"] if user else None

        cursor.execute(
            "INSERT INTO log (log_method, log_date, log_detail, admin_id) VALUES (%s, %s, %s, %s)",
            (method, log_date, detail, admin_id)
        )
        conn.commit()
    except mariadb.Error as e:
        print(f"Error logging action: {e}")
    finally:
        cursor.close()  # ปิด cursor ก่อน
        conn.close()

def verify_password(username, password):
    if not username or not password:
        return jsonify({"success": False, "message": "กรุณากรอกข้อมูลให้ครบ"}), 400

    conn, cursor = connecting_mariadb()
    if not conn or not cursor:
        return jsonify({"success": False, "message": "เชื่อมต่อฐานข้อมูลล้มเหลว"}), 500

    try:
        cursor.execute("SELECT * FROM admin WHERE admin_username = %s", (username,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"success": False, "message": "ไม่พบผู้ใช้"}), 401

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if hashed_password != user["admin_password"]:
            return jsonify({"success": False, "message": "รหัสผ่านไม่ถูกต้อง"}), 401

        return jsonify({"success": True, "message": "รหัสผ่านถูกต้อง"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()
        
def clean_text(text):
    """ ลบ \n, HTML Tags, ตัวเลขข้อ และอักขระพิเศษ แต่ยังคงรักษาคำไว้ """
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[\'"()#/,→o•]', '', text)  

    return text.strip()  # ลบช่องว่างหัวท้าย

@app.route('/searchid', methods=['GET'])
def searchid():
    try:
        data = request.get_json()
        doc_id = data["id"]
        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()
        # ใช้การค้นหาที่เหมาะสมจาก Milvus
        result = search_by_id(collection_name, doc_id)
        return jsonify({"results": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/searchkeyword', methods=['POST'])
def searchkeyword():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": 'Key "text" not found in the request data'}), 400
        query_text = str(data['text']).strip()
        if not query_text:
            return jsonify({"error": "Query text is empty"}), 400


        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()
        

        dense_vector = get_embedding(query_text)
        sparse_vector = get_bm25_vector([query_text])
        sparse_vector = dense_to_sparse(sparse_vector)
        # print("Sparse Vector:", sparse_vector)
        sparse_vector = list(sparse_vector.values())[0]  # ดึง list ออกมา
        # print("Sparse Vector ดึง list ออกมา:", sparse_vector)

        results_hybrid = search_hybrid(collection_name, dense_vector, sparse_vector, alpha=0.5)
        

        if not results_hybrid:
            return jsonify({"message": "No relevant documents found.", "results": []}), 200

        return jsonify({"results": results_hybrid}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/update', methods=['POST'])
def update_log():
    try:
        data = request.get_json(silent=True) or {}  #  ป้องกัน NoneType
        logs = data.get("logData", [])
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({"success": False, "message": "กรุณากรอกข้อมูลให้ครบ"}), 400

        response = verify_password(username, password)  #  ตรวจสอบรหัสผ่าน
        if response.status_code != 200:  #  ถ้าล็อกอินไม่สำเร็จ
            return response

        if not logs:
            return jsonify({"error": "No logData provided."}), 400

        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()

        for log in logs:
            method = log.get("method")
            doc_id = log.get("id")
            new_content = clean_text(log.get("new_content", ""))
            
            print("new_content:",new_content)

            if method == "edit":
                try:
                    update_data(collection_name, doc_id, new_content)
                    log_action("EDIT", f"แก้ไขเอกสาร ID: {doc_id}", username)
                except Exception as e:
                    return jsonify({"error": f"Failed to edit document {doc_id}: {str(e)}"}), 500

            elif method == "delete":
                try:
                    delete_data(collection_name, doc_id)
                    log_action("DELETE", f"ลบเอกสาร ID: {doc_id}", username)
                except Exception as e:
                    return jsonify({"error": f"Failed to delete document {doc_id}: {str(e)}"}), 500

        return jsonify({"success": True, "message": "Logs updated successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/insert', methods=['POST'])
def insert_content():
    try:
        data = request.get_json(silent=True) or {}
        # print(data)
        content_data = data.get("contentData", [])
        admin_username = data.get("username")
        admin_password = data.get("password")

        if not admin_username or not admin_password:
            return jsonify({"error": "กรุณากรอก username และ password"}), 400

        verify = verify_password(admin_username, admin_password)
        if not verify:
            return jsonify({"error": "Invalid username or password"}), 401
        
        warningIndexes = data.get("warningIndexes")
        replacementMapping = data.get("replacementMapping")
        print(replacementMapping)
        if not content_data:
            return jsonify({"error": "No contentData provided."}), 400


        texts = [clean_text(item.get("page_content", "").strip()) for item in content_data if item.get("page_content", "").strip()]
        if not texts:
            print("Error: No valid page content found")
            return jsonify({"error": "No valid page_content found."}), 400


        subject = content_data[0].get("metadata", {}).get("subject", "Unknown")

        #  สร้าง Dense Vector
        dense_vectors = [get_embedding(text).tolist() for text in texts]  # แปลงเป็น list

        #  สร้าง Sparse Vector (BM25)
        sparse_vectors = get_bm25_vector(texts) 
        sparse_vector_dict = dense_to_sparse(sparse_vectors)
        
        #  เชื่อมต่อกับ Milvus
        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()

        #  ลบข้อมูลที่มี similarid จาก replacementMapping
        if replacementMapping:
            for idx in replacementMapping:
                similarid = content_data[idx].get("similarid")
                if similarid:
                    # ลบข้อมูลเก่าที่มี similarid
                    delete_data(collection_name, similarid)
                    log_action("DELETE", f"ลบข้อมูลที่มี ID {similarid}", admin_username)

        #  Insert ข้อมูลใหม่ลง Milvus
        insert_ids = insert_data(collection, texts, dense_vectors, list(sparse_vector_dict.values()), subject)
        log_action("ADD", f"เพิ่มข้อมูล {len(texts)} รายการ", admin_username)

        return jsonify({
            "message": f"Inserted {len(texts)} documents into '{collection_name}' successfully!",
            "status": "success",
            "inserted_ids": list(insert_ids)  #  แปลงเป็น list ปกติ
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500





replacement_dict = {
    "csb": "หลักสูตรปริญญาตรีสาขาวิชาวิทยาการคอมพิวเตอร์โครงการพิเศษ(สองภาษา)",
    "ซีเอสบี": "หลักสูตรปริญญาตรีสาขาวิชาวิทยาการคอมพิวเตอร์โครงการพิเศษ(สองภาษา)",
    "cs": "ภาควิชาวิทยาการคอมพิวเตอร์",
    "ซีเอส": "ภาควิชาวิทยาการคอมพิวเตอร์",
    "มจพ": "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "คณะวิท": "คณะวิทยาศาสตร์ประยุกต์"
}

# เรียงลำดับคีย์จากยาว -> สั้น ป้องกัน cs แทนที่ก่อน csb
sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)

def replace_words(text, replacement_dict):
    for word in sorted_keys:
        text = text.replace(word, replacement_dict[word])  # ใช้ replace() แทน regex
    return text

@app.route('/call_llm', methods=['POST'])
def call_llm():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": 'Key "text" not found in the request data'}), 400
        query_text = str(data['text'])
        query_text = replace_words(query_text, replacement_dict)
        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()

        dense_vector = get_embedding(query_text)
        sparse_vector = get_bm25_vector([query_text])
        sparse_vector = dense_to_sparse(sparse_vector)
        sparse_vector = list(sparse_vector.values())[0] 

        results_hybrid = search_hybrid(collection_name, dense_vector, sparse_vector, alpha=0.5)

        if not results_hybrid:
            return jsonify({"message": "No results found."}), 404

        results_text = "\n".join([res["text"] for res in results_hybrid[:5]]) if results_hybrid else "No relevant results found."
        generated_response = textgenerate(query_text, results_text)

        conn, cursor = connecting_mariadb()

        bangkok_tz = pytz.timezone('Asia/Bangkok')
        log_date = datetime.now(bangkok_tz).strftime('%Y-%m-%d %H:%M:%S ')

        try:
            cursor.execute(
                "INSERT INTO question (q_text, q_date, k_id) VALUES (%s, %s, %s)", 
                (query_text,log_date ,1)
            )
            q_id = cursor.lastrowid  

            search_result_ids = []
            print(results_hybrid)
            for res in results_hybrid:
                kb_id = res.get('id', 0)
                score = res.get('score', 0.0)
                subject = res.get('subject', None)  
                if subject is None or subject == "":  
                    subject = None
                else:
                    subject = int(subject)  

                cursor.execute(
                    "INSERT INTO searchresults (q_id, kb_id, subject_id ,re_score, seach_datetime) VALUES (%s, %s, %s,%s, %s)",
                    (q_id, kb_id,subject ,score,log_date)
                )
                search_result_ids.append(cursor.lastrowid) 

            cursor.execute(
                "INSERT INTO answer (ans_text, ans_date) VALUES (%s, %s)",
                (generated_response,log_date)
            )
            ans_id = cursor.lastrowid  

            for sr_id in search_result_ids:
                cursor.execute(
                    "INSERT INTO generatingdata (sr_id, ans_id, gen_date) VALUES (%s, %s, %s)",
                    (sr_id, ans_id,log_date)
                )

            conn.commit()  #  บันทึกข้อมูลลงฐานข้อมูล

        except mariadb.Error as e:
            conn.rollback()  # ❌ Rollback ถ้าเกิดปัญหา
            print(f"Database error: {e}")
            return jsonify({"error": "Failed to insert data into database"}), 500

        finally:
            cursor.close()  #  ปิด Cursor
            conn.close()  #  ปิด Connection

        return jsonify({"responses": generated_response}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    conn, cursor = connecting_mariadb()
    try:
        cursor.execute("SELECT subject_id, subject_name FROM subject;")
        subjects = cursor.fetchall()
        return jsonify(subjects)
    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        
@app.route('/api/cluster', methods=['GET'])
def get_cluster():
    conn, cursor = connecting_mariadb()
    try:
        cursor.execute("SELECT cluster_id, cluster_name FROM cluster;")
        cluster = cursor.fetchall()
        return jsonify(cluster)
    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/api/sub_subjects/<int:subject_id>', methods=['GET'])
def get_sub_subjects(subject_id):
    conn, cursor = connecting_mariadb()
    try:
        cursor.execute("SELECT sub_subject_id, sub_subject_name FROM sub_subject WHERE subject_id = %s", (subject_id,))
        sub_subjects = cursor.fetchall()
        return jsonify(sub_subjects)
    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/history', methods=['GET'])
def get_history():
    conn, cursor = connecting_mariadb()
    
    try:
        #  ดึงข้อมูล Cluster และ Subject
        cursor.execute("SELECT cluster_id, cluster_name FROM cluster;")
        clusters_data = cursor.fetchall()
        clusters = {c["cluster_id"]: c["cluster_name"] for c in clusters_data}

        cursor.execute("SELECT subject_id, subject_name FROM subject;")
        subjects_data = cursor.fetchall()
        subjects = {s["subject_id"]: s["subject_name"] for s in subjects_data}

        #  ดึงข้อมูลประวัติการสร้างคำตอบ
        cursor.execute("""
            SELECT g.gen_date, s.cluster_id, sr.subject_id, q.q_text, a.ans_text, g.ans_id
            FROM generatingdata g
            JOIN searchresults sr ON g.sr_id = sr.sr_id
            JOIN question q ON sr.q_id = q.q_id
            JOIN answer a ON g.ans_id = a.ans_id
            JOIN subject s ON sr.subject_id = s.subject_id
            ORDER BY g.gen_date DESC
        """)
        history_data = cursor.fetchall()

        #  รวมข้อมูลที่มี ans_id เดียวกัน
        history_map = defaultdict(lambda: {"time": "", "category": set(), "subject": set(), "question": "", "answer": ""})

        for row in history_data:
            ans_id = row["ans_id"]

            clean_time = row["gen_date"].strftime("%Y-%m-%d %H:%M:%S")

            history_map[ans_id]["time"] = clean_time
            history_map[ans_id]["category"].add(clusters.get(row["cluster_id"], "-"))
            history_map[ans_id]["subject"].add(subjects.get(row["subject_id"], "-"))
            history_map[ans_id]["question"] = row["q_text"]
            history_map[ans_id]["answer"] = row["ans_text"]

        #  จัดรูปแบบ JSON Response
        result = [
            {
                "time": data["time"],
                "category": ", ".join(data["category"]),
                "subject": ", ".join(data["subject"]),
                "question": data["question"],
                "answer": data["answer"],
            }
            for data in history_map.values()
        ]

        return jsonify(result)

    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/admin_login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}  
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "กรุณากรอกข้อมูลให้ครบ"}), 400

    response = verify_password(username, password)  
    if response.json["success"]:
        log_action("LOGIN", "เข้าสู่ระบบสำเร็จ", username)  
    return response



@app.route('/logout', methods=['POST'])
def logout():
    data = request.json
    admin_username = data.get("username")

    if admin_username:
        log_action("LOGOUT", "ออกจากระบบสำเร็จ", admin_username)

    return jsonify({"success": True, "message": "ออกจากระบบแล้ว"})


# ดึงข้อมูล Log
@app.route('/api/logs', methods=['GET'])
def get_logs():
    conn, cursor = connecting_mariadb()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # ดึงข้อมูลจากฐานข้อมูล
        cursor.execute("SELECT log_id, log_method, log_date, log_detail, admin_id FROM log")
        logs = cursor.fetchall()

        return jsonify(logs)
    
    except mariadb.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        conn.close()
        
@app.route('/api/weekly-data', methods=['GET'])
def get_weekly_data():
    conn, cursor = connecting_mariadb()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # กำหนด timezone เป็น Bangkok
        bangkok_tz = pytz.timezone('Asia/Bangkok')

        # รับค่า offset สำหรับการเปลี่ยนสัปดาห์
        offset = request.args.get("offset", 0, type=int)

        # คำนวณ start_date และ end_date ตาม offset
        today = datetime.now(bangkok_tz)
        start_date = today - timedelta(days=today.weekday(), weeks=-offset)  # เลื่อนตาม offset
        start_date = start_date.replace(hour=0, minute=0, second=0, tzinfo=None)  # ลบ tzinfo
        start_date = bangkok_tz.localize(start_date)  # ทำการ localize ใหม่

        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)

        cursor.execute("""
            SELECT DATE(ans_date) AS date, COUNT(*) AS count 
            FROM answer 
            WHERE ans_date BETWEEN %s AND %s 
            GROUP BY DATE(ans_date)
        """, (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S")))

        data = cursor.fetchall()

        # จัดรูปแบบให้อยู่ในลำดับวันจันทร์ - อาทิตย์
        weekly_counts = { (start_date + timedelta(days=i)).date(): 0 for i in range(7) }

        for row in data:
            date_obj = row["date"]
            if date_obj in weekly_counts:
                weekly_counts[date_obj] = row["count"]

        return jsonify({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "weekly_data": [weekly_counts[date] for date in sorted(weekly_counts.keys())]
        })
    
    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/linkimage', methods=['GET'])
def get_latest_images():
    conn, cursor = connecting_mariadb()
    try:
        cursor.execute("SELECT imageurl FROM linkimage ORDER BY date DESC LIMIT 8")
        image_urls = [row["imageurl"] for row in cursor.fetchall()]

        response = jsonify({"image_urls": image_urls})
        response.headers["Content-Type"] = "application/json"  #  บังคับให้ส่ง JSON
        return response

    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

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


@app.route('/data_scapingpage', methods=['POST'])
def data_scapingpage():
    try:
        data = request.get_json(silent=True) or {}  
        

        current_date = datetime.now().strftime('%Y-%m-%d')
        date = convert_thai_date(data["date"])

        if date < current_date:
            return jsonify({
                "error": f"Date {date} is in the past. Data insertion aborted."
            }), 200
        imageurl = data["image"]
        texts = [data["content"]]  

        subject = data["subject"]
        print(subject)

        dense_vectors = [get_embedding(text).tolist() for text in texts]  

        sparse_vectors = get_bm25_vector(texts)  
        sparse_vector_dict = dense_to_sparse(sparse_vectors)

        connect_to_milvus()
        collection_name = "knowledge_base"
        collection = Collection(name=collection_name)
        collection.load()

        insert_ids = insert_data(collection, texts, dense_vectors, list(sparse_vector_dict.values()), subject)
        result = insert_imageurl(imageurl,current_date,date)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify({
            "message": f"Inserted {len(texts)} documents into '{collection_name}' successfully!",
            "status": "success",
            "inserted_ids": list(insert_ids)  
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def insert_imageurl(imageurl, datepost, date):
    conn, cursor = connecting_mariadb()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor()
        sql = "INSERT INTO linkimage (imageurl, datepost, date) VALUES (%s, %s, %s)"
        cursor.execute(sql, (imageurl, datepost, date))
        conn.commit()

        return {"message": "Image URL inserted successfully", "inserted_id": cursor.lastrowid}

    except mariadb.Error as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    finally:
        cursor.close()
        conn.close()

# API สำหรับรับ POST request
@app.route("/postlinkimage", methods=["POST"])
def post_link_image():
    try:
        data = request.json
        imageurl = data["image"]
        date = data["date"]
        

        datepost = datetime.now().strftime('%Y-%m-%d')
        
        date = convert_thai_date(data["date"])
        
        if date < datepost:
            return jsonify({
                "error": f"Date {date} is in the past. Data insertion aborted."
            }), 200

        if not imageurl or not datepost or not date:
            return jsonify({"error": "Missing required fields"}), 400

        result = insert_imageurl(imageurl, datepost, date)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/monthly-data', methods=['GET'])
def get_monthly_data():
    conn, cursor = connecting_mariadb()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        offset = int(request.args.get("offset", 0))

        today = datetime.now()
        target_month = today.month + offset
        target_year = today.year

        if target_month < 1:
            target_month += 12
            target_year -= 1
        elif target_month > 12:
            target_month -= 12
            target_year += 1

        first_day = datetime(target_year, target_month, 1)
        last_day = (first_day + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        first_day_str = first_day.strftime('%Y-%m-%d 00:00:00')
        last_day_str = last_day.strftime('%Y-%m-%d 23:59:59')


        cursor.execute("SELECT subject_id, subject_name FROM subject")
        subjects = cursor.fetchall()
        subject_mapping = {str(row["subject_id"]): row["subject_name"] for row in subjects}

        label_to_count = {subject_id: 0 for subject_id in subject_mapping.keys()}


        cursor.execute("""
            SELECT subject_id, COUNT(*) as count 
            FROM searchresults
            WHERE seach_datetime BETWEEN %s AND %s
            GROUP BY subject_id
        """, (first_day_str, last_day_str))

        search_counts = cursor.fetchall()
        for row in search_counts:
            subject_id = str(row["subject_id"])
            if subject_id in label_to_count:
                label_to_count[subject_id] = row["count"]

        labels = [subject_mapping[subject_id] for subject_id in subject_mapping]
        data = [label_to_count[subject_id] for subject_id in subject_mapping]

        monthly_data = {
            "labels": labels,
            "datasets": [
                {
                    "data": data,
                }
            ]
        }

        return jsonify(monthly_data)

    except mariadb.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

        
if __name__ == '__main__':
    app.run(port=5000)
