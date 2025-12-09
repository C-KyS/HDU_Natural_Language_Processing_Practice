import jieba.analyse
import warnings
warnings.filterwarnings("ignore")
sentence = input()

# 任务：基于jieba中的TF-IDF算法完成对sentence的关键词提取，提取前三个关键词并以一行输出
# ********** Begin *********#

keywords=jieba.analyse.extract_tags(sentence, topK=3, withWeight=False, allowPOS=("ns", "n","nr"))
#keywords = jieba.analyse.tfidf(sentence, topK=3, withWeight=False, allowPOS=("ns", "n", "vn", "v"))
print(" ".join(keywords))
# ********** End **********#