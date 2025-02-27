# 匯入套件
from bs4 import BeautifulSoup as bs
import requests as req
from pprint import pprint


# 自行輸入頁數.取得所有問答內容與網址
def fetch_library_data(page):
    # 建立 list 來放置列表資訊
    list_posts = []
    
    # 取得新聞列表
    url = f"https://tpml.gov.taipei/News.aspx?n=E5F579B94C9D2941&sms=87415A8B9CE81B16&page={page}&PageSize=20" 

    # 用 requests 的 get 方法把網頁抓下來
    res = req.get(url) 

    # 指定 lxml 作為解析器
    soup = bs(res.text, "lxml") 

    # 取得列表的文字與超連結
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

            # 使用 BeautifulSoup 解析並清理內容
            clean_soup = bs(html, "lxml")

            # 找到所有的 <p> 標籤，並提取其文字
            paragraphs = clean_soup.select('p')
            content = "".join([p.get_text() for p in paragraphs])
            
            # 整合到列表資訊的變數當中
            list_posts[index]['html'] = content

            # 預覽所有結果
            print(f"問: {obj['question']} \n 答: {content} \n 網址:{obj['link']} ")
            # pprint(list_posts)

        return list_posts  # 返回所有的資料
            # 預覽所有結果
            # print(f"問: {obj['question']} \n 答: {content} \n 網址:{obj['link']} ")
    
# 自行輸入頁數 顯示該頁資料
page_number = int(input("請輸入頁數: "))
fetch_library_data(page_number)