# -*- coding: utf-8 -*-
"""
Data Collection from Sciencedirect
keywords: online game, video game, mobile game
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time
#options.add_argument('window-size=1920x1080')
#options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
mobile_emulation = {
    #"deviceMetrics": { "width": 360, "height": 380, "pixelRatio": 3.0 },
    "userAgent": "Mozilla/5.0 (Linux; Android 6.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"
}

chrome_options = Options()

# chrome_options.add_argument('headless') # 브라우저가 켜질 지 말지 / 버그 방지(코드 완벽 시)
chrome_options.add_argument("---window-size=380,520")
chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--blink-settings=imagesEnabled=false") # 이미지 로딩 설정

driver = webdriver.Chrome('./chromedriver.exe', chrome_options=chrome_options) # 설정 옵션으로 크롬드라이버 실행
driver.implicitly_wait(3) # 3초 wait 후

import pandas as pd
contents = pd.DataFrame(columns = ["id","paper"])

# totally 7 pages
for j in range(6):
    url1="""https://www.sciencedirect.com/search?qs=online%20game%20OR%20video%20game%20OR%20mobile%20game%20AND%20
    NOT%20game%20theory&show=100&sortBy=relevance&articleTypes=FLA&offset={j}00""".format(j=j)
    driver.get(url1) #Sciencedirect(version free for KHU students)
    result = driver.find_element_by_css_selector("div.ResultList.col-xs-24")
    time.sleep(3)
    for i in range(1,102):
        child = driver.find_element_by_xpath("//*[@id='main_content']/main/div[1]/div[2]/div[2]/ol/li[{i}]".format(i=i))
        #Click abstract icon and get each abstract
        if child.get_attribute('class') == 'ResultItem col-xs-24 push-m':
            submit_btn = child.find_element_by_css_selector("[aria-label='Abstract']")
            submit_btn.click()
            time.sleep(1)
            paper = child.text
            time.sleep(1)
            submit_btn.click()
            time.sleep(1)
            Dict = dict(  zip(contents.columns, [i, paper])  )
            contents = contents.append(Dict, ignore_index=True)
        else:
            pass

df1 = contents
df1['paper']=df1['paper'].apply(lambda x: str(x).replace('"',"'"))
df2= pd.DataFrame(df1['paper'].str.split("\n").values.tolist())
a=df2.iloc[:,5:24]
df2['abstract']=df2[a.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis = 1)
df2.drop(df2.iloc[:,5:23], inplace = True, axis = 1)
df2.columns=['type','title','journal','author','dae','abstract']
df2['abstract']=df2['abstract'].apply(lambda x: str(x).replace('\n'," "))

df2 = df2.drop(['dae'], 1)
df2.to_csv("sd_game.csv", index = False)
