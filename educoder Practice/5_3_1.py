from gensim import corpora, models
import jieba.posseg as jp, jieba
from basic import get_stopword_list

texts=[]
# 构建语料库
for i in range(5):
    s=input()
    texts.append(s)

flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = get_stopword_list()
words_ls = []
for text in texts:
    words = [word.word for word in jp.cut(text) if word.flag in flags and word.word not in stopwords]
    words_ls.append(words)

# 去重，存到字典
dictionary = corpora.Dictionary(words_ls)
corpus = [dictionary.doc2bow(words) for words in words_ls]

# 任务:基于 gensim 的models构建一个lda模型，主题数为1个
# ********** Begin *********#
 
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=1)


# ********** End **********#
for topic in lda.print_topics(num_words=1):
    print(topic[1].split('*')[1],end="")
