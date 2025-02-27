# 匯入套件
from bs4 import BeautifulSoup as bs
import requests as req
from pprint import pprint
import pymysql
from sentence_transformers import SentenceTransformer
import os
import time
import faiss
import numpy as np

# 資料庫連線
connection = pymysql.connect(
    host='localhost',
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 從資料庫讀取所有QA資料
def fetch_data_from_db():
    try:
        with connection.cursor() as cursor:
            # 選取所有QA資料
            sql = "SELECT id, question FROM library_qa"
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    except Exception as e:
        print("從資料庫讀取資料時發生錯誤：")
        print(e)
        return []

# 模型名稱
# model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
# model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
# model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# 索引存放路徑
index_path = 'Vector_index/vector_model4.index'

# 初始化 SentenceTransformer 模型
model = SentenceTransformer(model_name)

# 用於存儲所有的嵌入向量和對應的 ID
all_embeddings = []
all_ids = []

# 從資料庫獲取資料並生成向量
print("正在從資料庫讀取資料...")
db_data = fetch_data_from_db()

if db_data:
    print(f"總共讀取到 {len(db_data)} 筆資料")
    
    # 將所有問題轉換成向量並計算時間
    start_time = time.time()
    questions = [item['question'] for item in db_data]
    question_embeddings = model.encode(questions, normalize_embeddings=True)
    end_time = time.time()
    print(f"向量轉換時間: {end_time - start_time} 秒")

    # 將嵌入向量添加到列表中
    all_embeddings.extend(question_embeddings)
    all_ids.extend([item['id'] for item in db_data])

# 將列表轉換為 numpy 數組
embeddings = np.array(all_embeddings).astype('float32')
ids = np.array(all_ids).astype('int64')

# 確保 embeddings 和 ids 的長度相同
assert len(embeddings) == len(ids), f"embeddings 長度 ({len(embeddings)}) 不等於 ids 長度 ({len(ids)})"

# 讀取索引，不存在就初始化
if not os.path.exists(index_path):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatIP(dims)  # 初始化索引的維度
    index = faiss.IndexIDMap(index)  # 讓 index 有記錄對應 doc id 的能力
else:
    # 索引存在，直接讀取
    index = faiss.read_index(index_path)

    # 檢查嵌入向量的維度是否與索引的維度一致
    if embeddings.shape[1] != index.d:
        print(f"嵌入向量的維度: {embeddings.shape[1]}")
        print(f"索引的維度: {index.d}")
        raise ValueError("嵌入向量的維度與索引的維度不一致")

# 加入 doc id 到 對應的 vector
index.add_with_ids(embeddings, ids)  # 加入 向量 與 文件ID

# 儲存索引
faiss.write_index(index, index_path)

# 釋放記憶體
del index, embeddings, all_embeddings, all_ids



