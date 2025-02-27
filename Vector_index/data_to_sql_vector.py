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
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    charset=os.getenv('DB_CHARSET'),
    cursorclass=pymysql.cursors.DictCursor
)

# 自行輸入頁數.取得所有問答內容與網址
def fetch_library_data(page):
    # 建立 list 來放置列表資訊
    list_posts = []

    # 透過url 找到該頁面所有問答內容
    url = f"https://tpml.gov.taipei/News.aspx?n=E5F579B94C9D2941&sms=87415A8B9CE81B16&page={page}&PageSize=20"
    res = req.get(url)
    soup = bs(res.text, "lxml")

    # 用迴圈取得列表的文字與超連結
    for td in soup.select('td.CCMS_jGridView_td_Class_1'):
        # 使用 select 找到 <a> 標籤
        a_tags = td.select('a')
        for a in a_tags:
            # 加入列表資訊
            list_posts.append({
                'question': a.get_text(),
                'link': 'https://tpml.gov.taipei/' + a['href']
            })

    # 走訪每一個 a link，整合網頁內文
    for index, obj in enumerate(list_posts):
        res_ = req.get(obj['link'])
        soup_ = bs(res_.text, "lxml")

        # 取得實際需要的內容 (類似 JavaScript 的 innerHTML)
        html = soup_.select_one('p').decode_contents()
        clean_soup = bs(html, "lxml")

        # 找到所有的 <p> 標籤，並提取其文字
        paragraphs = clean_soup.select('p')
        content = "".join([p.get_text() for p in paragraphs])

        # 整合到列表資訊的變數當中
        list_posts[index]['answer'] = content

        # 加入延遲，避免對網站造成過大負擔
        time.sleep(3)  # 新增

    return list_posts

# 模型名稱
model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
# model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# 索引存放路徑
index_path = '.Vector_index/vectorv2.index'

# 初始化 SentenceTransformer 模型
model = SentenceTransformer(model_name)  # 新增

# 起始頁數
init_page = 1

# 最新頁數
latest_page = 5

# 用於存儲所有的嵌入向量和對應的 ID
all_embeddings = []
all_ids = []

# 在已經知道分頁數的情況下
for page in range(init_page, latest_page + 1):
    print(f"正在處理第 {page} 頁...")
    posts = fetch_library_data(page)  # 獲取當前頁的所有問答資料

    # 將所有問答轉換成向量並計算時間
    start_time = time.time()  # 新增
    questions = [post['question'] for post in posts]
    question_embeddings = model.encode(questions, normalize_embeddings=True)
    end_time = time.time()  # 新增
    print(f"轉換時間: {end_time - start_time} 秒")  # 新增

    # 將嵌入向量添加到列表中
    all_embeddings.extend(question_embeddings)

    try:
        with connection.cursor() as cursor:  # 使用 with 語法自動處理游標的打開和關閉
            # 直接寫入資料 (新舊資料都在)
            sql = "INSERT INTO `library_qa` (`question`, `answer`, `link`) VALUES (%s, %s, %s)"

            # 準備要插入的資料
            insert_data = [(post['question'], post['answer'], post['link']) for post in posts]

            # 使用 executemany() 插入多筆資料
            cursor.executemany(sql, insert_data)

            # 提交 SQL 執行結果
            connection.commit()

            # 獲取插入的最後一個 ID
            last_id = cursor.lastrowid

            # 計算插入的 ID 範圍
            inserted_ids = range(last_id - len(insert_data) + 1, last_id + 1)
            all_ids.extend(inserted_ids)

    except Exception as e:
        # 回滾
        connection.rollback()
        print("SQL 執行失敗")
        print(e)

# 關閉資料庫連線
connection.close()

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



