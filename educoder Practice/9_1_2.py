import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans


class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        # 加载停用词

        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path):
        # 文本预处理，每行一个文本
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def get_text_tfidf_matrix(self, corpus):

        # 获取tfidf矩阵

        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights

    def kmeans(self, corpus_path, n_clusters=2):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        corpus = self.preprocess_data(corpus_path)
        weights = self.get_text_tfidf_matrix(corpus)
        result = {}
        #任务：完成基于K-Means算法的文本聚类，并将结果保存到result变量中。
        # ********** Begin *********#
        myKmeans = KMeans(n_clusters=n_clusters)
        result = myKmeans.fit_predict(weights)  
        '''
        myKmeans.fit(weights)
        labels = myKmeans.labels_

        for i,label in enumerate(labels):
            if label not in result:
                result[label]=[]
            result[label].append(i)
            '''
        # ********** End **********#
        return  result


