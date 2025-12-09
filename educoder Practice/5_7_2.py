from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import pandas as pd
import gensim
def D2V():
    article = pd.read_excel('data.xlsx') #data为训练集，繁体
    sentences = article['内容'].tolist()
    split_sentences = []
    
    for i in sentences:
        split_sentences.append(i.split(' '))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(split_sentences)]

# 任务：基于 gensim 构建 doc2vec 模型并命名为doc2vec_stock进行保存
# ********** Begin *********#


    model = Doc2Vec(documents, size=100, window=5, min_count=5, workers=4, epoch=5000)  
    model.save("doc2vec_stock.model") 
# ********** End **********#