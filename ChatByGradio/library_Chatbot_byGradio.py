## 設定 OpenAI API Key 變數
from dotenv import load_dotenv
import os
import openai
import gradio as gr
import pymysql
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# 取得目前檔案的目錄和專案根目錄
current_dir = Path(__file__).parent
root_dir = current_dir.parent

# 載入 .env 檔案（使用完整路徑）
env_path = root_dir / '.env'
load_dotenv(env_path)

# 驗證環境變數
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(f"⚠️ 找不到 OPENAI_API_KEY，請確認 .env 檔案存在於：{env_path}")

# 設定 新版 OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# 系統提示詞
SYSTEM_PROMPT = """你是台北市立圖書館的 AI 客服，專門回答關於「圖書館」的問題。
擅長根據資料庫內容提供正確資訊並進行友善的互動回應。
如果用戶的問題超出這些範圍，請回覆：
『抱歉，我僅限於「圖書館」相關話題喔！您可以嘗試在網路上搜尋相關資訊。』

你的任務是：
* 根據用戶問題，從資料庫中檢索出最相關的內容
* 優先參考資料庫檢索結果進行回答，並在回答中體現檢索內容
* 如果資料庫沒有直接相關答案，根據語義相似度結果提供延伸建議或可能的資訊
* 結合多筆資料庫結果，整理出完整、簡明且友善的答案

回答時請遵循：
- 使用正體中文，語氣專業且親切
- 先提供直接答案，再補充說明（如有延伸建議）
- 確保信息準確性，避免猜測
- 在資料庫檢索結果中，標註資料來源或關鍵資訊
- 若無相關結果，說明資料庫查詢無結果並提供其他建議

範例回答格式：
1. **答案：** 提供資料庫中最相關的答案
2. **補充說明：** 根據資料庫內容延伸的背景資訊
3. **建議：** 若無資料庫結果，提出可能的查詢方向或其他可參考資源"""

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

    def search_similar_questions(self, query, top_k=3):
        # 將查詢轉換為向量
        query_vector = self.model.encode(
            [query], 
            normalize_embeddings=True
        )
        
        # 使用 FAISS 搜尋最相似的問題
        distances, indices = self.index.search(query_vector, k=top_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and distance > 0.5:  # 相似度閾值
                try:
                    with self.connection.cursor() as cursor:
                        sql = "SELECT question, answer FROM library_qa WHERE id = %s"
                        cursor.execute(sql, (int(idx),))
                        result = cursor.fetchone()
                        if result:
                            results.append({
                                'similarity': float(distance),
                                'question': result['question'],
                                'answer': result['answer']
                            })
                except Exception as e:
                    print(f"資料庫查詢錯誤：{e}")
        
        return results

def handle_user_query(user_query):
    try:
        qa_system = QASystem()
        try:
            return qa_system.search_similar_questions(user_query)
        finally:
            if hasattr(qa_system, 'connection') and qa_system.connection:
                qa_system.connection.close()
            if hasattr(qa_system, 'index'):
                del qa_system.index
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
        return conn
    except Exception as e:
        print(f"❌ 無法連線到資料庫: {e}")
        return None

def get_ai_response(user_query, db_results):
    try:
        if db_results is None:
            db_results = []

        context = f"用戶問題：{user_query}\n\n"
        
        if not db_results:
            context += "資料庫中沒有找到相關的答案。請提供一個合適的回應。\n"
        else:
            context += "相關的資料庫結果（按相似度排序）：\n"
            # 確保 db_results 是字典列表
            for result in db_results:
                context += f"[相似度: {result.get('similarity', 0):.4f}]\n"
                context += f"問：{result.get('question', '')}\n"
                context += f"答：{result.get('answer', '')}\n\n"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        # 使用新版 API 調用方式
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.7,
            stream=True
        )

        full_response = ""
        for chunk in response:
            content = getattr(chunk.choices[0].delta, 'content', '')
            if content:
                full_response += content

        # 清理並返回回應
        return full_response.strip() if full_response else "AI 回應生成失敗，請再試一次。"

    except Exception as e:
        print(f"AI 處理時發生錯誤：{str(e)}")
        import traceback
        print(f"詳細錯誤：\n{traceback.format_exc()}")
        return f"很抱歉，我現在無法提供適當的回應。請稍後再試。"

def chat_response(user_query, history):
    try:
        # 先查詢資料庫
        db_results = handle_user_query(user_query)
        
        # 再使用 AI 生成回應
        response = get_ai_response(user_query, db_results)
        return response
    except Exception as e:
        print(f"聊天回應發生錯誤：{e}")
        return "抱歉，系統暫時無法處理您的請求，請稍後再試。"

# 修改 ChatInterface 的設置
gr.close_all()
gr.ChatInterface(
    fn=chat_response,  # 改用 chat_response 而不是 get_ai_response
    theme="Soft",
    title=article,
    examples=[
        "請問圖書館的開放時間是什麼時候？",
        "我想了解如何辦理借書證",
        "兒童區有什麼特別的服務？",
        "幫我介紹一下圖書館的資源嗎？"
    ]
).queue().launch(
    debug=True
)