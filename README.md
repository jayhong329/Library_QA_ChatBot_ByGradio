#  Library QA ChatBot - with OpenAI_API

## 訓練資料來源
臺北市立圖書館 - 常見問題 (https://tpml.gov.taipei/News.aspx?n=E5F579B94C9D2941&sms=87415A8B9CE81B16)

### 基礎模型
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- sentence-transformers/multi-qa-MiniLM-L6-cos-v1
- sentence-transformers/distiluse-base-multilingual-cased-v1
- sentence-transformers/paraphrase-mpnet-base-v2

## 安裝套件
* gradio 5.6.0
* gradio_client 1.4.3
* jieba 0.42.1
* fuzzywuzzy 0.18.0
* faiss 1.8.0
* openai 1.64.0
* sentence-transformers 3.1.1

## 說明
1. 嘗試用 SQL 直接查找關鍵字，再透過 結巴斷詞 + 模糊搜尋
2. 使用四個不同的模型，創建向量索引。透過 cosine similarity ，排序出 Top-K，測試生成回復
   
總共分以下三個檔案夾:
一. Web_scraping - 透過 web_scraping - BeautifulSoup 抓取資料，存入 SQL 內。
二. Vector_index - 連線 SQL，存入 Database ，加入 SentenceTransformer 模型，生成向量索引，查找最相似的問題。
三. ChatByGradio - 結合以上兩份檔案，透過不同的檢索方式，練習聊天機器人 ( FuzzyWuzzy + FAISS) ，構建 Gradio_ChatBot，並重新設計 SYSTEM_PROMPT

成果
- 執行過程的擷圖 (測試從 SQl Database + Jieba + Fuzzywuzzy + Faiss + renew SYSTEM_PROMPT) 
  ![image](https://github.com/user-attachments/assets/42a2ef09-f580-4c8a-8acf-57ce0ead1a3b)
  ![image](https://github.com/user-attachments/assets/6a86cff5-466a-4e8f-aea5-b6f05802b7ae)
  ![image](https://github.com/user-attachments/assets/87a7009d-4f03-459c-8748-809b89ceb445)
  ![image](https://github.com/user-attachments/assets/73767b56-fc45-4765-afc4-b053648ab0a5)



