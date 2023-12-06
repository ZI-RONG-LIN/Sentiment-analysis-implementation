# 針對PTT汽車版 Nissan 相關主題文章進行輿情分析
在學習網頁爬蟲相關資源時，發現到大多數的文章僅著重在網頁爬蟲技術分享。輿情分析的部分，免費的中文學習資源較少，大多數以收費課程之形式存在。

因此希望透過此專案可以將自己的實作過程記錄下來，並將學習的結果分享給大家。
這邊僅針對語法概念及目的進行說明，完整之語法執行結果大家可藉由此專案資料夾內之.ipynb檔做更進一步的了解。

在網頁爬蟲專案中已經透過網頁爬蟲技巧收集 PTT Car 版中有關 Nissan 的討論文章，接下來將在本專案中探討社會大眾對於 Nissan 相關主題的意見與情感傾向。

以下為本專案之大綱，後續將於每段步驟進行介紹並提供示範語法

**大綱**
1. 環境安裝
2. 語法
   * 套件匯入
  
## 輿情分析
### 環境安裝
本專案使用 Python3 並且會使用 pip 來安裝所需的套件。以下是需要安裝的套件：

* pandas：進行資料處理和資料分析的工具。

* jieba：用於中文文本分詞。它廣泛用於中文文本處理，提供了詞分詞、詞性標註等功能。
  
* nltk（Natural Language Toolkit）：Python 中用於自然語言處理。它包括文本處理、分詞、詞幹提取、標記、解析等工具。通常用於英文文本處理

* snownlp：是一個用於處理中文文本。它包括情感分析、詞性標註、文本分類等模塊。
  
* scikit-learn：是一個機器學習庫，提供了簡單而有效的數據挖掘和分析工具。它包括各種用於分類、回歸、聚類等的算法。

* matplotlib：用於 Python 的 2D 繪圖庫。它提供各種靜態、動畫和交互式繪圖。通常用於數據可視化。

* wordcloud：用於創建文字雲。文字雲以可視化的方式呈現了給定文本中單詞的頻率，頻率更高的單詞以較大的字體顯示。通常用於展示文集中最常見的單詞。

使用以下指令來安裝這些套件：

```python
pip install pandas jieba nltk snownlp scikit-learn matplotlib wordcloud
```

### 語法
#### 套件匯入
首先，開發過程有時會忘記到底先前有沒有匯入過想使用的套件，導致重複在不同地方 import 一樣的套件進來，因此如果要使用之套件數量不多時，可以先將會使用到的套件一次匯入，避免讓 import 語法分散在不同段落中。

使用以下指令來匯入稍後將使用到的套件：

```python
import pandas as pd
import re
import jieba
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba.analyse as analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```

#### 讀取PTT文章資料
在網頁爬蟲專案中，最後將爬取回來的文章資訊儲存在csv檔案中，因此第一步我們先將csv檔案匯入。

使用以下指令來匯入csv檔：
```python
# 資料讀取
df = pd.read_csv("20231129_nissan_web_crawler.csv")
```

#### 原文提取與清理
輿情分析的內容主要會著重在content(文章內容)跟comment(留言)欄位，在開始分析之前，首先會需要針對文本數據進行前處理，以便進行後續的文本分析。清理的目的是移除所有非文字之資訊，清理步驟包括刪除網址、HTML 標籤、非字母數字漢字空白字符以及換行符，可視資料格式或分析需求新增或刪減清理邏輯。以下會透過迴圈分別針對每篇文章的內容以及留言進行格式清理，並將清理好的文字分別存進空的陣列中，以利後續步驟使用。

使用以下指令來清理原文：
```python
#建立一個空的陣列，用於存放清理好的文章內容
resub_content=[]

#透過迴圈將文章內容進行清理後，依序存進陣列中
for i in df['content']:
    http = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*',re.S)
    clean_text = re.sub(http, '', i)
    clean_text = re.sub(r'<.*?>', '', clean_text)
    clean_text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\s]', '', clean_text)
    clean_text = re.sub(r'\n', '', clean_text)
    clean_text = re.sub(r' ', '', clean_text)
    resub_content.append(clean_text)

#建立一個空的陣列，用於存放清理好的留言
resub_comment=[]

#透過迴圈將留言進行清理後，依序存進陣列中
for i in df['comment']:
    http = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*',re.S)
    clean_text = re.sub(http, '', i)
    clean_text = re.sub(r'<.*?>', '', clean_text)
    clean_text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\s]', '', clean_text)
    clean_text = re.sub(r'\n', '', clean_text)
    clean_text = re.sub(r' ', '', clean_text)
    resub_comment.append(clean_text)
```
