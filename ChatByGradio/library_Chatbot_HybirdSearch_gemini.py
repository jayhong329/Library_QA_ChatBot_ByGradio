## 設定 GOOGLE_API_KEY 變數
from dotenv import load_dotenv
import os
import gradio as gr
import pymysql
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import jieba
from fuzzywuzzy import process
import numpy as np
from google import genai
from google.genai import types
from google.genai.types import Part, Content

# 取得目前檔案的目錄和專案根目錄
current_dir = Path(__file__).parent
root_dir = current_dir.parent

# 載入 .env 檔案（使用完整路徑）
env_path = root_dir / '.env'
load_dotenv(env_path)

# 設定 新版 GOOGLE_API_KEY
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError(f"⚠️ 找不到 GOOGLE_API_KEY .env 檔案存在於：{env_path}")

client = genai.Client(api_key=google_api_key)

# 定義全域變數儲存歷史紀錄
history = []

# 系統提示詞
SYSTEM_PROMPT = """你是台北市立圖書館的 AI 客服，專門回答關於「圖書館」的問題。
擅長根據資料庫內容提供正確資訊並進行友善的互動回應。
如果用戶的問題超出這些範圍，請回覆：
『抱歉，我僅限於「圖書館」相關話題喔！您可以嘗試在網路上搜尋相關資訊。』

你的任務是：
* 直接回答用戶問題，內容必須來自資料庫
* 僅提供資料庫中檢索結果的資訊
* 不需分析資料庫內容，也不需提出改進建議
* 如資料庫有多筆資訊，請直接整理成簡明答案
* 若資料庫無相關資訊，回覆：「抱歉，沒有找到相關資訊，建議參考圖書館官網查詢。」

### 回答格式：
- **答案：** 提供資料庫中最相關的答案
- 若資訊不足，可以補充說明
- 不要重述用戶問題
- 不要分析資料結構或內容品質
- 使用正體中文，語氣專業且親切

範例回答：
**答案：** 台北市立圖書館週二至週六開放時間為 8 時 30 分至 21 時，週日與週一為 9 時至 17 時。
如需查詢特定分館資訊，建議參考官網公告。"""

desc = "直接輸入您的問題或關鍵詞 " \
       "AI 助手會根據圖書館問答資料庫回復相關服務與政策。" 

article = "<h1>圖書館客服問答系統 </h1>"\
          "<h3>使用說明:</h3> " \
          "<ul><li>直接輸入您的問題或關鍵詞</li>" \
          "<li>AI 助手會根據圖書館問答資料庫回復相關服務與政策</li></ul>"

class QASystem:
    index_cache = None  # 類別靜態屬性 (儲存索引)

    def __init__(self):
        # 初始化模型
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = SentenceTransformer(self.model_name)
        self.index_path = root_dir / 'Vector_index' / 'vector_model4.index'
        
        if QASystem.index_cache is None:
            print(f"首次載入索引：{self.index_path}")
            try:
                QASystem.index_cache = faiss.read_index(str(self.index_path))
                print("索引載入成功！")
            except Exception as e:
                print(f"讀取索引文件時發生錯誤：{e}")
                raise

        # 讓實例共用靜態索引
        self.index = QASystem.index_cache
        
        # 資料庫連線
        try:
            self.connection = connect_db()
        except Exception as e:
            print(f"資料庫連線錯誤：{e}")
            raise

        # 從資料庫讀取問答資料
        db_results = fetch_data_from_db(self.connection)
        if not db_results:
            raise ValueError("資料庫中沒有問答資料！")
        
        # 處理問答資料  以問題為鍵，答案為值
        self.questions = [item['question'] for item in db_results]
        self.answers = {item['question']: item['answer'] for item in db_results}

    def search_similar_questions(self, query, top_k=3):
        # try:

        # 1. Jieba 斷詞
        query_tokens = " ".join(jieba.cut(query))
        print(f"斷詞: {query_tokens}") 
        
        # 模糊搜尋 (FuzzyWuzzy)
        fuzzy_results = process.extract(query_tokens, self.questions, limit=10, scorer=process.fuzz.partial_ratio)
        fuzzy_candidates = {res[0]: res[1] / 100 for res in fuzzy_results if res[1] > 60}

        # 語義檢索 (FAISS)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        _, top_indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        faiss_candidates = {self.questions[idx]: 1.0 for idx in top_indices[0] if idx < len(self.questions)}

        # 合併結果，並加權
        merged_scores = {}
        for q in set(fuzzy_candidates.keys()).union(faiss_candidates.keys()):
            fuzzy_score = fuzzy_candidates.get(q, 0)
            faiss_score = faiss_candidates.get(q, 0)
            merged_scores[q] = 0.6 * fuzzy_score + 0.4 * faiss_score

        # 排序並回傳
        sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        final = [(q, self.answers[q]) for q, _ in sorted_results if q in self.answers]
        return final[:top_k]
        
        # finally:
        #     # 關閉資料庫連線
        #     if self.connection and self.connection.open:
        #         self.connection.close()
        #         print("🔌 資料庫連線已關閉")
    
    def __del__(self):
        if self.connection and self.connection.open:  # 確保連線存在且未關閉
            self.connection.close()
            print("🔌 資料庫連線已關閉")
        
def handle_user_query(message):
    try:
        qa_system = QASystem()
        db_results = qa_system.search_similar_questions(message)  # 確保這裡能夠獲取到結果
        print(f"查詢結果：{db_results}")  # 添加日誌輸出
        return db_results
    except Exception as e:
        print(f"處理查詢時發生錯誤：{e}")
        return []  # 返回空列表而不是拋出錯誤

# 連接到 MySQL 資料庫
def connect_db():
    try:
        conn = pymysql.connect(
            host='localhost',
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("資料庫連接成功！")  # 添加日誌輸出
        return conn
    except Exception as e:
        print(f"❌ 無法連線到資料庫: {e}")
        return None

# 2. 從資料庫讀取 FAQ
def fetch_data_from_db(conn):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id, question, answer FROM library_qa"
            cursor.execute(sql)
            results = cursor.fetchall()
            # print(f"讀取到的資料：{results}")  # 添加日誌輸出
            return results
    except Exception as e:
        print(f"資料庫讀取失敗: {e}")
        return []
    
def get_ai_response(message, db_results):
    global history  # 使用全域變數
    try:
        if db_results is None or len(db_results) == 0:
            return "資料庫中沒有找到相關的答案。請提供一個合適的回應。"

        context_parts = []
        # context_parts.append(types.Part(text=f"用戶問題：{message}\n\n"))
        context_parts.append(types.Part(text="僅回答資料庫內容，請勿分析資料或提供建議。"))
        context_parts.append(types.Part(text="相關的資料庫結果（按相似度排序）：\n"))

        for question, answer in db_results:
            context_parts.append(types.Part(text=f"* 問：{question}\n* 答：{answer}\n\n"))

        history.append({"role": "user", "content": message})
        
        # 創建 Content 物件
        content = types.Content(role="user", parts=context_parts)

        # 使用 Google API 生成回應
        print("發送請求到 Google API，請求內容：", content)  # 添加日誌輸出

        response = client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=[content], # 直接傳遞 Content 物件
            config=types.GenerateContentConfig(    
                temperature=0.7,
                top_k=5,
            )
        )

        # 處理回應
        if response.text:
            # 將回應加入歷史紀錄
            history.append({"role": "assistant", "content": response.text})
            history = history[-10:]  # 保留最後 10 筆資料

            print(f"目前的歷史紀錄：{history}")
            print(f"目前總共儲存的對話數：{len(history)}")

            return response.text
        else:
            print("AI 回應生成失敗，沒有收到任何文字。")
            return "AI 回應生成失敗，請再試一次。"

    except Exception as e:
        print(f"AI 處理時發生錯誤：{str(e)}")
        import traceback
        print(f"詳細錯誤：\n{traceback.format_exc()}")
        return "很抱歉，我現在無法提供適當的回應。請稍後再試。"


def chat_response(message, history=None):
    history = history or []

    try:
        # 1. 查詢資料庫
        db_results = handle_user_query(message)

        # 2. 使用 AI 生成回應
        response = get_ai_response(message, db_results)
        print(f"使用AI生成回應：{response}")

        return response
    
    except Exception as e:
        print(f"聊天回應發生錯誤：{e}")
        return "抱歉，系統暫時無法處理您的請求，請稍後再試。"

# 修改 ChatInterface 的設置
gr.ChatInterface(
    fn=chat_response,  # 改用 chat_response 而不是 get_ai_response
    theme="Soft",
    title=article,
    examples=[
        "請問圖書館的開放時間是什麼時候？",
        "我想了解如何辦理借書證",
        "兒童區有什麼特別的服務？",
        "介紹一下圖書館有哪些資源？"
    ],
    chatbot=gr.Chatbot(show_label=False, show_copy_button=True)
).queue().launch(
    debug=True
)