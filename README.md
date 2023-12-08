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

#### 資料預處理
自然語言的預處理又分成好幾個階段，基本的前處理步驟包括斷詞(Tokenization)、移除停用詞、詞幹提取(Stemming)、詞形還原(Lemmatization)等等，一樣可視資料格式或分析需求新增或刪減清理邏輯。接下來將針對基本的前處理步驟依序介紹。
##### 斷詞(Tokenization)
在自然語言處理中，斷詞是一個重要且關鍵的步驟，透過這個步驟我們能夠識別語言中的基本單位，即單詞(Token)，有助於機器深入理解語言的結構。其次，斷詞使得文本更容易被處理和分析，為後續的語法分析、語意分析等任務打下基礎。這一步驟還有助於特徵提取，將文本轉換成機器可處理的數字向量表示。在信息檢索和檢索模型方面，斷詞確保在搜索引擎和相關系統中能夠找到相符的文件，提升系統效能。最後，斷詞也是訓練語言模型的重要步驟，協助模型理解文本的結構和語境。總而言之，斷詞的運用促進了自然語言處理的各個面向，有助於機器理解和處理人類語言。

而中文的語句表達結構不像英文是透過空白或標點符號區分，相同的句子用不同的切分方式傳達出的涵義會大不相同，因此處理中文斷詞會相較英文來的複雜許多。

###### Jieba套件介紹
Jieba是一個開源的Python套件，經常被用來進行中文斷詞的套件，它支援多種斷詞模式（精確模式、全模式與搜尋引擎）。由於此套件最初是由中國百度的開發者開發，因此其核心為簡體中文，但由於其開源的特性，陸陸續續也有台灣的開發者補上繁體中文的部分，因此目前Jieba已經可以支援簡體和繁體中文的斷詞。另外，若對Jieba的斷詞結果不滿意時，也可以自行加入特定單詞或者是自行定義字典，調整分詞的結果。

前面有提到Jieba支援不同的斷詞模式，接下來會用以下這段文字來呈現各模式的分詞結果：
```python
text='日產門市人員專業度佳 試乘體驗好\n我買智駕版和雙色精裝配備原廠電子後視鏡 這台內裝有氣氛燈用久了很值得'
```
1. 全模式：jieba 會將句子中所有可能的詞語都列出來，不考慮詞語的連貫性。
* 優點：能夠涵蓋句子中的所有可能詞語。
* 缺點：產生的詞語可能過多，且包含了一些不必要的詞。
```python
  print("全模式：")
  seg_list = jieba.cut(text, cut_all=True) # 全模式
  print("/ ".join(seg_list))  
  ```
斷詞結果如下：
<p align="center">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E5%85%A8%E6%A8%A1%E5%BC%8F.png">
</p>

精確模式：是最基本的分詞模式，它通常用於對文本進行細粒度的分詞。在這種模式下，jieba 會儘量將句子切分成最小的詞語，也是jieba斷詞的預設模式。
* 優點：適用於對文本進行精確的分詞，產生的詞語相對較短。
* 缺點：可能無法處理一些特殊的詞彙或慣用語。
```python
print("精確模式：")
seg_list = jieba.cut(text, cut_all=False)  # 精確模式
print("/ ".join(seg_list)) 
```
斷詞結果如下：
<p align="center">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E7%B2%BE%E7%A2%BA%E6%A8%A1%E5%BC%8F.png">
</p>

3. 搜尋引擎模式：搜尋引擎模式在精確模式的基礎上，對長詞進行再次切分，以提高召回率，適合用於搜尋引擎。
* 優點：針對長詞進行再次切分，提高召回率。
* 缺點：可能會出現一些不必要的短詞。
```python
print("搜索引擎模式：")
seg_list = jieba.cut_for_search(text)  # 搜索引擎模式
print("/ ".join(seg_list)) 
```
斷詞結果如下：
<p align="center">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E6%90%9C%E5%B0%8B%E5%BC%95%E6%93%8E%E6%A8%A1%E5%BC%8F.png">
</p>

