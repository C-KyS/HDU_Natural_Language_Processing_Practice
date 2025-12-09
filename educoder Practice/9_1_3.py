import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DbscanClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None): # 加载停用词 
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path): # 文本预处理
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def get_text_tfidf_matrix(self, corpus): # 获取tf-idf矩阵
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        weights = tfidf.toarray() # 获取tfidf矩阵中权重
        return weights

    def pca(self, weights, n_components=2): # PCA对数据进行降维
        pca = PCA(n_components=n_components)
        return pca.fit_transform(weights)

    def dbscan(self, corpus_path, eps=0.1, min_samples=3, fig=True): # 基于密度的文本聚类算法
     
        # 任务：完成 DBSCAN 聚类算法
        # ********** Begin *********#
        corpus = self.preprocess_data(corpus_path) # 加载语料
        weights = self.get_text_tfidf_matrix(corpus) # 词向量转换
        pca_weights = self.pca(weights) # 减低维度
        clf = DBSCAN(eps=eps, min_samples=min_samples) # 构建聚类算法
        
        # ********** End **********#
        y = clf.fit_predict(pca_weights)

       
        result = {}  # 每个样本所属的簇
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result