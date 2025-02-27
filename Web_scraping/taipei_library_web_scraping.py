# 匯入套件
from bs4 import BeautifulSoup as bs
import requests as req
from pprint import pprint
import pymysql

# 資料庫連線
connection = pymysql.connect(
    host = 'localhost',
    user = 'root',
    password = 'P@ssw0rd',
    database = 'taipei_library',
    charset = 'utf8mb4',
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
        a_tags  = td.select('a')
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

    return list_posts  

        # 預覽所有結果
        # print(f"問: {obj['question']} \n 答: {content} \n 網址:{obj['link']} ")

# 起始頁數
init_page = 1

# 最新頁數
latest_page = 5

# 在已經知道分頁數的情況下
for page in range(init_page, latest_page + 1):
    print(f"正在處理第 {page} 頁...")
    posts = fetch_library_data(page)  # 獲取當前頁的所有問答資料

    try:
        with connection.cursor() as cursor:  # 使用 with 語法自動處理游標的打開和關閉
            # 直接寫入資料 (新舊資料都在)
            sql = "INSERT INTO `library_qa` (`question`, `answer`, `link`) VALUES (%s, %s, %s)"

            # 準備要插入的資料
            insert_data = [(post['question'], post['answer'],  post['link']) for post in posts]
            
            # 使用 executemany() 插入多筆資料
            cursor.executemany(sql, insert_data)

            # 提交 SQL 執行結果
            connection.commit()

    except Exception as e:
        # 回滾
        connection.rollback()
        print("SQL 執行失敗")
        print(e)


# 關閉資料庫連線
connection.close()
