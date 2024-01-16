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
df = pd.read_csv("/nissan_web_crawler.csv")
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

可以看到由於語句中含有特定品牌或是汽車領域用語，斷詞上還是會有些許誤差，因此可以透過自行加入一些跟汽車相關的用語來定義斷詞字典，調整分詞的結果，接著再使用斷詞語法即可看到事先定義的單詞已經成功被正確切割開來。
4. 自定義斷詞字典
```python
#使用切換詞庫的功能來改善斷詞結果。
jieba.set_dictionary('/dict.txt.big')
print("精確模式：")
seg_list = jieba.cut(text, cut_all=False)  # 精確模式
print("/ ".join(seg_list)) 
```
斷詞結果如下：
<p align="center">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E8%87%AA%E5%AE%9A%E7%BE%A9%E6%96%B7%E8%A9%9E%E5%AD%97%E5%85%B8.png">
</p>

##### 移除停用詞
Stop Words(停用詞)是指那些在處理文本時被視為無意義或無助於理解文本內容的常見詞語。這些詞通常是高頻率的、普遍存在的詞，但它們並未提供太多的上下文資訊，因此在進行文本分析時，可以將它們排除，以提高模型的效能並降低處理的複雜性。常見的停用詞像是：的/了/是/在/可以/但是等等的單詞，一樣可視分析目的排除停用詞，另外也可以自透過自定義停用詞字典來快速移除大量不必要的單詞。
```python
# 開啟停用詞字典 
f = open('/停用詞-繁體中文.txt', encoding='utf8')
# 讀取停用詞字典 
data = f.read() 
# 使用換行符號分割
stop_words = data.split("\n") 
f.close()

# 建立一個空的陣列，用於存放預處理後的文章內容 
token_content = []
# 透過迴圈將先前已清理過的內容進行斷詞及排除停用詞
for i in resub_content:
    tokens = jieba.cut(i)
    text = [word for word in tokens if word not in stop_words]
    text = ' '.join(text)
    token_content.append(text)

 # 建立一個空的陣列，用於存放預處理後的留言   
token_comment = []
# 透過迴圈將先前已清理過的留言進行斷詞及排除停用詞
for i in resub_comment:
    tokens = jieba.cut(i)
    text = [word for word in tokens if word not in stop_words]
    text = ' '.join(text)
    token_comment.append(text)
```
#### 情感分析
經過前面的資料清理後，終於來到這個專案的重點部分，也就是輿情分析。這邊我們使用SnowNLP套件，如同先前的介紹，SnowNLP主要被應用在分析中文文本，裡頭包括了情感分析、詞性標註、文本分類等模組，用於情感分析的模組可透過SnowNLP.sentiments進行引用。SnowNLP.sentiments主要透過返回一個介於 0 到 1 之間的浮點數，來表達輸入文本的情感分數。分數越接近 1，表示正面情感越強；分數越接近 0，表示負面情感越強；分數為 0.5，表示為中立情緒。

不知道大家有沒有常常逛評論區的習慣，由於文章本身的內容與留言區的評論可能會因為每個人的想法不同，有時候文章內容與底下留言討論的風向會有極大的差異，因此我會將文章的本身內容的輿情以及留言討論區的輿情區分開來觀察。透過以下語法可以快速了解文章整體的輿情狀況以及情感分數，並統計正、負面文章分別有幾篇，且將情感分數呈現出來。若想更進一步將每篇文章一一列出觀察也可以，但由於爬回來的文章數量也不少，因此這個部分就不逐一列出，這部分有附上相關語法，有興趣的人可以嘗試印出觀察看看。

使用以下指令來進行情感分析：
```python
# 建立空的陣列存放針對文章內容分析後的情感標籤及分數
sentiment_score_content=[]
sentiment_label_content=[]

# 透過迴圈依序針對"文章內容"進行情感分析
for i in token_content:
    s = SnowNLP(i)
    sentiment_score = s.sentiments
    sentiment_label = 'Positive' if sentiment_score > 0.5 else 'Negative' if sentiment_score < 0.5 else 'Neutral'
    sentiment_score_content.append(sentiment_score)
    sentiment_label_content.append(sentiment_label)
    # 若想知道每篇文章個別的情感分數與結果可使用此語法印出:print(f"文章內容情感分析結果：{sentiment_label} (情感分數: {sentiment_score})")

# 統計正負面文章數量
positive_count_content = sum(1 for label in sentiment_label_content if label == 'Positive')
negative_count_content = sum(1 for label in sentiment_label_content if label == 'Negative')

# 統計正負面文章情感分數，數字越接近0或1，代表情緒越強烈
positive_avgscore_content = mean(x for x in sentiment_score_content if x >0.5)
negative_avgscore_content = mean(x for x in sentiment_score_content if x <0.5)
total_label = 'Positive' if mean(sentiment_score_content) > 0.5 else 'Negative' if mean(sentiment_score_content) < 0.5 else 'Neutral'

print("文章內容分析結果如下：")
print(f"整體文章情緒：{total_label} (整體平均情感分數:{mean(sentiment_score_content):.2f})")
print(f"正面文章數量：{positive_count_content} (平均情感分數: {positive_avgscore_content:.2f})")
print(f"負面文章數量：{negative_count_content} (平均情感分數: {negative_avgscore_content:.2f})")


# 建立空的陣列存放針對"留言內容"分析後的情感標籤及分數
sentiment_score_comment=[]
sentiment_label_comment=[]
# 透過迴圈依序針對"留言內容"進行情感分析
for i in token_comment:
    s = SnowNLP(i)
    sentiment_score = s.sentiments
    sentiment_label = 'Positive' if sentiment_score > 0.5 else 'Negative' if sentiment_score < 0.5 else 'Neutral'
    sentiment_score_comment.append(sentiment_score)
    sentiment_label_comment.append(sentiment_label)
    #print(f"文章留言情感分析結果：{sentiment_label} (情感分數: {sentiment_score})")

# 統計數量
positive_count_comment = sum(1 for label in sentiment_label_comment if label == 'Positive')
negative_count_comment = sum(1 for label in sentiment_label_comment if label == 'Negative')

# 統計情感分數，數字越接近0或1，代表情緒越強烈
positive_avgscore_comment = mean(x for x in sentiment_score_comment if x >0.5)
negative_avgscore_comment = mean(x for x in sentiment_score_comment if x <0.5)
total_label = 'Positive' if mean(sentiment_score_comment) > 0.5 else 'Negative' if mean(sentiment_score_comment) < 0.5 else 'Neutral'

print()
print("留言內容分析結果如下：")
print(f"整體留言情緒：{total_label} (整體平均情感分數:{mean(sentiment_score_comment):.2f})")
print(f"正面留言數量：{positive_count_comment} (平均情感分數: {positive_avgscore_comment:.2f})")
print(f"負面留言數量：{negative_count_comment} (平均情感分數: {negative_avgscore_comment:.2f})")

```
分析結果如下：
<p align="left">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E6%AD%A3%E8%B2%A0%E6%96%87%E7%AB%A0%E6%95%B8%E9%87%8F.png" width="350" height="150">
</p>

#### 文本主題及關鍵字分析
接著，我們將進一步分析文章內容及留言中的主題以及關鍵字，這樣的分析可以幫助我們更快的理解內容談論的重點資訊為何，也可以更快速的幫助我們篩選出那些我們關注的議題。

使用以下指令來進行文本主題及關鍵字分析：
```python
#創建了一個 TF-IDF（Term Frequency-Inverse Document Frequency） 向量化器。
tfidf_vectorizer = TfidfVectorizer()
#將文檔轉換為 TF-IDF 矩陣
tfidf_matrix = tfidf_vectorizer.fit_transform([token_comment[10]])
#創建了一個 Latent Dirichlet Allocation (LDA) 主題模型，設置主題數量為 1。
lda = LatentDirichletAllocation(n_components=1, random_state=42)
#使用 LDA 模型擬合 TF-IDF 矩陣，發現文檔中的主題
lda.fit(tfidf_matrix)

# 顯示主題詞彙
# 獲取 TF-IDF 向量化器中的特徵名稱，即詞彙。
feature_names = tfidf_vectorizer.get_feature_names_out()
# 找到每個主題中權重最高的前 10 個詞彙的索引。
top_keywords_idx = lda.components_[0].argsort()[:-10 - 1:-1]
# 根據索引獲取詞彙名稱，得到每個主題的前 10 個關鍵詞。
top_keywords = [feature_names[i] for i in top_keywords_idx]
# 印出主題分析結果，即每個主題的前 10 個詞彙。
print(f"主題分析結果：{', '.join(top_keywords)}")

print()
# 使用 jieba 提取文本中的關鍵詞，限定詞性為名詞。
keywords = jieba.analyse.extract_tags(token_comment[10], topK=10, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# 印出關鍵詞提取結果，即文本中的前 10 個關鍵詞及其權重。
print(f"關鍵詞提取結果：{keywords}")
```
分析結果如下：
<p align="left">
<img src="https://github.com/ZI-RONG-LIN/Sentiment-analysis-implementation/blob/main/img/%E4%B8%BB%E9%A1%8C%E5%8F%8A%E9%97%9C%E9%8D%B5%E5%AD%97%E5%88%86%E6%9E%90.png" width="1000" height="150">
</p>
