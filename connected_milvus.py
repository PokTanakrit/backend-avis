from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
from datetime import datetime
from rank_bm25 import BM25Okapi
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from pythainlp.tokenize import word_tokenize



bm25_corpus = []  # เก็บเอกสารทั้งหมดในรูปแบบ tokenized
bm25_model = None  # ใช้เก็บโมเดล BM25
doc_id_map = {}  # ใช้ map doc_id -> index ใน corpus


# ======= Load Model Once =======
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_embedding = AutoTokenizer.from_pretrained(MODEL_EMBEDDING_NAME)
model_embedding = AutoModel.from_pretrained(MODEL_EMBEDDING_NAME).to(device).eval()
print(f"Model and tokenizer '{MODEL_EMBEDDING_NAME}' loaded successfully.")

# Load the Hugging Face model and tokenizer for reranking
MODEL_RERANKING_NAME = "BAAI/bge-reranker-v2-m3"
tokenizer_rerank = AutoTokenizer.from_pretrained(MODEL_RERANKING_NAME)
model_rerank = AutoModelForSequenceClassification.from_pretrained(MODEL_RERANKING_NAME).to(device).eval()
print(f"Model and tokenizer '{MODEL_RERANKING_NAME}' loaded successfully.")



def connect_to_milvus(host="localhost", port="19530"):
    connections.connect("default", host=host, port=port)
    print(f"Connected to Milvus server at {host}:{port}")
    
import logging
logging.basicConfig(level=logging.DEBUG)

def get_embedding(text):

    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    try:
        inputs = tokenizer_embedding(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        #  แก้ไข: ย้าย inputs ไปที่ device เดียวกัน
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model_embedding(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # ส่งกลับไปที่ CPU ก่อนใช้ numpy
        print("embedding complete")
        return embedding / np.linalg.norm(embedding)
    
    except Exception as e:
        raise RuntimeError(f"Failed to compute embedding: {str(e)}")

from rank_bm25 import BM25Okapi

from scipy.sparse import coo_matrix, csr_matrix
from rank_bm25 import BM25Okapi

def get_bm25_vector(texts, use_existing=True):
    collection_name = "knowledge_base"
    connect_to_milvus()
    if isinstance(texts, str):
        texts = [texts]

    texts_old = []
    collection = Collection(collection_name)  

    if collection and use_existing:
        if utility.has_collection(collection.name):
            collection.load()
            if collection.num_entities > 0:
                results = collection.query(expr="id >= 0", output_fields=["text"])
                texts_old = [res["text"] for res in results if isinstance(res.get("text"), str)]

    texts_all = texts_old + texts if texts_old else texts
    tokenized_texts = [word_tokenize(text) for text in texts_all]
    bm25 = BM25Okapi(tokenized_texts)
    vocab = list(bm25.idf.keys())  
    vocab_size = len(vocab)
    sparse_vectors = []  

    for text in texts:
        tokens = word_tokenize(text)  
        sparse_vector = {}  

        default_min_score = 0.0001 
        default_idf = 0.5  

        for token in set(tokens):
            if token in vocab:
                token_idx = vocab.index(token)
                idf_score = bm25.idf.get(token, default_idf) 
                score = max(default_min_score, idf_score)
                sparse_vector[token_idx] = score

        sparse_vectors.append(sparse_vector if sparse_vector else {0: default_min_score})  

    print(" BM25 Vector สร้างเสร็จแล้ว!")
    return sparse_vectors

# ======= Rerank Results =======
def rerank(query, results):
    """
    Re-rank search results based on relevance using a fine-tuned model.
    """
    if not isinstance(query, str):
        raise ValueError("Query should be a string.")
    
    if not results or not isinstance(results, list):
        raise ValueError("Results should be a non-empty list.")
    
    input_texts = [result['text'] for result in results if isinstance(result.get('text'), str)]
    
    if not input_texts:
        raise ValueError("Each result must contain a valid 'text' field as a string.")
    
    inputs = tokenizer_rerank(
        [query] * len(input_texts), input_texts,
        padding=True, truncation=True, return_tensors="pt"
    )
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model_rerank(**inputs).logits  

    if outputs.shape[1] == 2:  
        scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
    else:
        scores = outputs.squeeze(dim=-1).detach().cpu().numpy()  

    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])  
    
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

def create_collection(collection_name):

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR), 
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128)
    ]

    schema = CollectionSchema(fields, description="Hybrid Dense & Sparse Search Collection")
    collection = Collection(name=collection_name, schema=schema)
    index_dense = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",  
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="dense_vector", index_params=index_dense)
    index_sparse = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",  
        "params": {"inverted_index_algo": "DAAT_MAXSCORE"}
    }
    collection.create_index(field_name="sparse_vector", index_params=index_sparse)

    collection.load()
    print(f"Collection '{collection_name}' created with Dense & Sparse Search support.")

def insert_data(collection, texts, dense_vectors, sparse_vectors, subject):
    if not utility.has_collection(collection.name):
        raise ValueError(f"Collection '{collection.name}' does not exist.")

    if len(texts) != len(dense_vectors) or len(texts) != len(sparse_vectors):
        raise ValueError("Number of texts, dense_vectors, and sparse_vectors must be the same.")

    try:
        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dates = [current_date] * len(texts)  
        data = [
            dates,           
            texts,        
            dense_vectors,    
            sparse_vectors,  
            [subject] * len(texts)  
        ]

        insert_result = collection.insert(data)
        collection.flush()

        inserted_ids = insert_result.primary_keys if insert_result else []
        if not inserted_ids:
            raise RuntimeError("Insert operation failed, no IDs returned.")

        print(f" Inserted {len(inserted_ids)} documents with subject '{subject}' at '{current_date}'.")
        return inserted_ids

    except Exception as e:
        raise RuntimeError(f"❌ Failed to insert data: {str(e)}")

def search_cosine(collection_name, query_vector, top_k=10, threshold=0.0):
    """
    Perform cosine similarity search in Milvus with adjustable threshold.
    """
    try:
        collection = Collection(name=collection_name)
        collection.load()
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 5}
        }

        results = collection.search(
            [query_vector.tolist()],
            "dense_vector",
            search_params,
            limit=top_k,
            expr=None,
            output_fields=["id", "text", "subject", "date"]
        )
        
        search_results = [
            {
                "id": hit.id,
                "score": float(hit.distance),
                "text": hit.entity.get("text"),
                "subject": hit.entity.get("subject"),
                "date": hit.entity.get("date")
            }
            for result in results for hit in result if float(hit.distance) >= threshold
        ]

        return sorted(search_results, key=lambda x: x['score'], reverse=True)

    except Exception as e:
        raise RuntimeError(f"Search failed for collection '{collection_name}': {str(e)}")


def search_sparse(collection_name, query_vector, top_k=10, threshold=0.0, expr=None):
    """
    Perform sparse vector search (BM25, TF-IDF) in Milvus with adjustable parameters.
    """
    try:
        collection = Collection(name=collection_name)
        collection.load()
        
        search_params = {
            "index_type": "SPARSE_INVERTED_INDEX"
        }

        results = collection.search(
            [query_vector],
            "sparse_vector",
            search_params,
            limit=top_k,
            expr=expr, 
            output_fields=["id", "text", "subject", "date"]
        )
        
        search_results = [
            {
                "id": hit.id,
                "score": float(hit.distance),
                "text": hit.entity.get("text"),
                "subject": hit.entity.get("subject"),
                "date": hit.entity.get("date")
            }
            for result in results for hit in result if float(hit.distance) >= threshold
        ]

        return sorted(search_results, key=lambda x: x['score'], reverse=True)

    except Exception as e:
        raise RuntimeError(f"Sparse search failed for collection '{collection_name}': {str(e)}")

def search_hybrid(collection_name, dense_vector, sparse_vector, alpha=0.5):
    try:
        collection = Collection(name=collection_name)
        collection.load()
        dense_search_params = {"metric_type": "COSINE", "params": {"nprobe": 5}}
        dense_results = collection.search(
            [dense_vector.tolist()],  
            "dense_vector",
            dense_search_params,
            limit=10,
            expr=None,
            output_fields=["id", "text", "subject"]
        )

        sparse_search_params = {"index_type": "SPARSE_INVERTED_INDEX"}
        sparse_results = collection.search(
            [sparse_vector], 
            "sparse_vector",
            sparse_search_params,
            limit=10,
            expr=None,
            output_fields=["id", "text", "subject"]
        )

        if not dense_results and not sparse_results:
            return []

        score_map = {}

        for result in dense_results:
            for hit in result:
                doc_id = hit.id
                score = float(hit.distance)
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "dense": score,
                        "sparse": 0.0,
                        "text": hit.get("text"),
                        "subject": hit.get("subject"),
                    }
                else:
                    score_map[doc_id]["dense"] = score

        for result in sparse_results:
            for hit in result:
                doc_id = hit.id
                score = float(hit.distance)
                score = score 
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "dense": 0.0,
                        "sparse": score,
                        "text": hit.get("text"),
                        "subject": hit.get("subject"),
                    }
                else:
                    score_map[doc_id]["sparse"] = score

        hybrid_results = [
            {
                "id": str(doc_id),
                "hybrid_score": alpha * data["dense"] + (1 - alpha) * data["sparse"],
                "dense_score": data["dense"],
                "sparse_score": data["sparse"],
                "text": data["text"],
                "subject": data["subject"]
            }
            for doc_id, data in score_map.items()
        ]

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return hybrid_results

    except Exception as e:
        raise RuntimeError(f"Hybrid search failed for collection '{collection_name}': {str(e)}")

def delete_collection(collection_name):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")

def update_data(collection_name, doc_id, new_content, admin_username=None):
    """
    อัปเดตเอกสารใน Milvus โดยลบข้อมูลเดิม (ตาม doc_id) และแทรกข้อมูลใหม่จาก new_content (list of texts)
    """
    try:
        texts = new_content if isinstance(new_content, list) else [new_content.strip()]
        if not texts or all(text.strip() == "" for text in texts):
            print("⚠️ Error: No valid page content found")
            return False

        connect_to_milvus()
        collection = Collection(name=collection_name)
        collection.load()

        result = collection.query(expr=f"id == {doc_id}", output_fields=["subject", "date"])
        if not result:
            print(f"⚠️ ID {doc_id} not found in '{collection_name}'.")
            return False

        old_subject = result[0]["subject"]

        dense_vectors = [get_embedding(text).tolist() for text in texts]

        sparse_vectors = get_bm25_vector(texts)
        sparse_vector_dict = dense_to_sparse(sparse_vectors)

        collection.delete(expr=f"id == {doc_id}")
        collection.flush()

        insert_ids = insert_data(collection, texts, dense_vectors, list(sparse_vector_dict.values()), old_subject)
        collection.flush()

        print(f" Updated ID {doc_id} with new content successfully.")
        return True

    except Exception as e:
        logging.error(f"❌ update_data error: {str(e)}")
        return False








# ======= Delete Data by ID =======
def delete_data(collection_name, doc_id):
    """
    Delete a document from Milvus collection by ID.
    
    Args:
    - collection_name (str): The name of the Milvus collection.
    - doc_id (int): The ID of the document to delete.
    
    Returns:
    - bool: True if deletion was successful, False if ID not found.
    """
    collection = Collection(name=collection_name)
    collection.load()

    # Check if the document exists
    result = collection.query(expr=f"id == {doc_id}", output_fields=["id"])
    if not result:
        print(f"❌ ID {doc_id} not found in '{collection_name}'.")
        return False

    # Perform deletion
    collection.delete(expr=f"id == {doc_id}")
    collection.flush()  # Ensure deletion is committed
    print(f" Deleted ID {doc_id} from '{collection_name}'.")
    return True

# ======= Search Data by ID =======
def search_by_id(collection_name, doc_id):
    """
    Search for a document in Milvus by its ID.
    
    Args:
    - collection_name (str): The name of the Milvus collection.
    - doc_id (int): The ID of the document to search.
    
    Returns:
    - dict | None: The document if found, otherwise None.
    """
    try:
        collection = Collection(name=collection_name)
        collection.load()

        # ค้นหาข้อมูลโดยใช้ id
        results = collection.query(
            expr=f"id == {doc_id}",
            output_fields=["id", "text", "subject", "date"]
        )

        if not results:
            print(f"❌ ID {doc_id} not found in '{collection_name}'.")
            return None

        print(f"🔍 Found document: {results[0]}")
        return results[0]  # คืนค่าผลลัพธ์ที่พบ
    except Exception as e:
        print(f"❌ Search by ID failed: {str(e)}")
        return None

def delete_collection(collection_name):
    """
    Deletes a collection from Milvus.
    
    Args:
    - collection_name (str): The name of the Milvus collection to be deleted.
    """
    try:
        # ตรวจสอบว่า collection ที่ต้องการลบมีอยู่จริงใน Milvus หรือไม่
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)  # ลบ collection
            print(f"Collection '{collection_name}' has been deleted successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting collection '{collection_name}': {str(e)}")
        
def dense_to_sparse(vector):
    return {i: v for i, v in enumerate(vector) if v != 0}

# if __name__ == "__main__":
#     connect_to_milvus()
#     collection_name = "knowledge_base"
#     # delete_collection(collection_name)
#     # สร้าง collection (หากยังไม่มี)
#     create_collection(collection_name)

#     # จากนั้นจึงค่อยดึง collection และ load
#     collection = Collection(name=collection_name)
#     collection.load()

#     print(collection.schema)


# if __name__ == "__main__":
#     vector = get_embedding("สวัสดี")
    
#     print(vector.shape)