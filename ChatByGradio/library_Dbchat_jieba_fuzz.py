from dotenv import load_dotenv
import os
import gradio as gr
import pymysql
import jieba
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load the environment variables from .env file
load_dotenv()

# 連接到 MySQL 資料庫
def connect_db():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        charset=os.getenv('DB_CHARSET'),
        cursorclass=pymysql.cursors.DictCursor
    )

# 查詢資料庫
def query_db(message):
    try:
        db = connect_db()
        cursor = db.cursor()
        
        if not message.strip():
            return "請輸入關鍵字"

        # 使用結巴進行中文斷詞
        keywords = list(jieba.cut(message))
        if not keywords:
            return "請輸入有效關鍵字"

        # 先取出所有資料
        cursor.execute("SELECT question, answer FROM library_qa")
        all_data = cursor.fetchall()

        cursor.close()
        db.close()

        if not all_data:
            return "資料庫沒有資料"

        results = []
        
        # 模糊比對
        for item in all_data:
            question = item['question']
            answer = item['answer']
            score = 0

            # 計算關鍵字與問題、答案的模糊比對分數
            for keyword in keywords:
                question_score = fuzz.partial_ratio(keyword, question)
                answer_score = fuzz.partial_ratio(keyword, answer)

                # 加權分數 (問題比答案重要)
                score += (question_score * 0.7 + answer_score * 0.3) / len(keywords)

            if score > 50:  # 篩選最低相似度閥值 (可自行調整)
                item['relevance_score'] = score
                results.append(item)

        # 依相似度排序，取前 3 筆
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:3]

        if not results:
            return f"抱歉，找不到與「{message}」相關的資訊"

        # 組織回應
        response = f"找到以下與「{message}」最相關的資訊：\n\n"
        for i, item in enumerate(results, 1):
            response += f"【相關結果 {i}】\n"
            response += f"問：{item['question']}\n"
            response += f"答：{item['answer']}\n"
            response += f"相關性分數：{round(item['relevance_score'], 2)}\n"
            response += "-" * 40 + "\n"
        return response

    except Exception as e:
        print(f"資料庫錯誤：{e}")
        return "查詢時發生錯誤"

def chat_response(message, history):
    """處理用戶輸入的函數，適用於 ChatInterface"""
    if not message:
        return "請輸入問題"
    return query_db(message)

# 使用 ChatInterface 創建聊天界面
demo = gr.ChatInterface(
    fn=chat_response,
    title="圖書館問答系統",
    description="""
    使用說明：
    - 可以輸入多個關鍵字，用空格分隔
    - 系統會返回最相關的3個結果
    - 關鍵字越多，搜索結果越精確
    """,
    examples=[
        "圖書館 開放時間",
        "借書 規則 逾期",
        "超商借書 範圍",
        "借書證 辦理 資格"
    ],
    theme="soft"
)

if __name__ == "__main__":
    try:
        print("啟動聊天界面...")
        demo.launch(
            debug=True
        )
        print("聊天界面已啟動")
    except Exception as e:
        print(f"啟動錯誤：{e}")