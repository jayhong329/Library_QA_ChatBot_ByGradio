'''
匯入套件
'''
from sentence_transformers import SentenceTransformer
import faiss
import pymysql

class QASystem:
    def __init__(self):
        # 初始化模型 依序為 1 2 3 4
        # self.model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        # self.model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
        # self.model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

        self.model = SentenceTransformer(self.model_name)
        
        # 資料庫連線
        self.connection = pymysql.connect(
            host='localhost',
            user='root',
            password='P@ssw0rd',
            database='taipei_library',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        # 讀取 FAISS 索引 - model2  model4 比較OK
        self.index_path = 'Vector_index/vector_model4.index'
        self.index = faiss.read_index(self.index_path)
        
    def get_answer(self, query, top_k=3, similarity_threshold=0.5):
        # 將查詢轉換為向量
        query_vector = self.model.encode(
            [query], 
            normalize_embeddings=True  # 使用 cosine similarity 需要正規化
        )
        
        # 使用 FAISS 搜尋最相似的問題
        distances, indices = self.index.search(query_vector, k=top_k)
        
        # 過濾結果
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and distance > similarity_threshold:
                try:
                    with self.connection.cursor() as cursor:
                        sql = "SELECT question, answer FROM library_qa WHERE id = %s"
                        cursor.execute(sql, (int(idx),))
                        result = cursor.fetchone()
                        if result:
                            results.append({
                                'similarity': distance,
                                'question': result['question'],
                                'answer': result['answer']
                            })
                except Exception as e:
                    print(f"查詢資料庫時發生錯誤: {e}")
        
        return results

    def close(self):
        self.connection.close()
        del self.index

def main():
    qa_system = QASystem()
    
    try:
        while True:
            query = input("\n請輸入您的問題 (輸入 'q' 結束)：")
            if query.lower() == 'q':
                break
                
            results = qa_system.get_answer(query)
            
            if not results:
                print("\n抱歉，沒有找到相關的答案。")
                continue
                
            print("\n找到以下相關答案：")
            for i, result in enumerate(results, 1):
                print(f"\n--- 答案 {i} (相似度: {result['similarity']:.4f}) ---")
                print(f"相關問題：{result['question']}")
                print(f"答案：{result['answer']}")
                
    finally:
        qa_system.close()

if __name__ == "__main__":
    main()