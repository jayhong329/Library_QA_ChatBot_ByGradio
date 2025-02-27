from dotenv import load_dotenv
import os
import gradio as gr
import pymysql

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
        
        # 處理多個關鍵字
        keywords = message.strip().split()
        if not keywords:
            return "請輸入關鍵字"
        
        # 構建查詢條件
        conditions = []
        params = []
        relevance_cases = []
        
        for i, keyword in enumerate(keywords):
            # 為每個關鍵字添加條件
            conditions.append(f"(question LIKE %s OR answer LIKE %s)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            # 計算相關性分數
            relevance_cases.append(f"""
                WHEN question LIKE %s THEN {5 * (len(keywords) - i)}
                WHEN answer LIKE %s THEN {3 * (len(keywords) - i)}
            """)
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        # 構建完整的 SQL 查詢
        query = f"""
        SELECT 
            question, 
            answer,
            (CASE 
                {' '.join(relevance_cases)}
                ELSE 0 
            END) as relevance_score
        FROM library_qa 
        WHERE {' OR '.join(conditions)}
        ORDER BY relevance_score DESC
        LIMIT 3
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        cursor.close()
        db.close()
        
        if not results:
            return f"抱歉，找不到與「{message}」相關的資訊"
            
        # 組織回應
        response = f"找到以下與「{message}」最相關的資訊：\n\n"
        for i, item in enumerate(results, 1):
            response += f"【相關結果 {i}】\n"
            response += f"問：{item['question']}\n"
            response += f"答：{item['answer']}\n"
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
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
        print("聊天界面已啟動")
    except Exception as e:
        print(f"啟動錯誤：{e}")