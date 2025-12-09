import xlwt
import pickle
import itertools
import nltk
import os
import sklearn
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pos_f = 'src/step3/pkl_data/1000/pos_review.pkl'
neg_f = 'src/step3/pkl_data/1000/neg_review.pkl'

def load_data():  # 加载训练集数据
    global pos_review, neg_review
    pos_review = pickle.load(open(pos_f, 'rb'))
    neg_review = pickle.load(open(neg_f, 'rb'))

def create_word_bigram_scores(): # 计算整个语料里面每个词和双词搭配的信息量
    posdata = pickle.load(open(pos_f, 'rb'))
    negdata = pickle.load(open(neg_f, 'rb'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def find_best_words(word_scores, number): # 根据信息量进行倒序排序，选择排名靠前的信息量的词
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words

def pos_features(feature_extraction_method): # 赋予积极的文本类标签
    posFeatures = []
    for i in pos_review:
        posWords = [feature_extraction_method(i), 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method): # 赋予消极的文本类标签

    negFeatures = []
    for j in neg_review:
        negWords = [feature_extraction_method(j), 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures

def best_word_features(words): # 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
    global best_words
    return dict([(word, True) for word in words if word in best_words])

def score(classifier):
   # 任务：构建分类器模型并进行训练
   # ********** Begin *********#

    classifier = nltk.SklearnClassifier(classifier)  # 在nltk 中使用scikit-learn的接口
    classifier.train(train)  #训练分类器
   # ********** End **********#

    pred = classifier.classify_many(dev)  # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_dev, pred)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

# 使用测试集测试分类器的最终效果
def use_the_best():
    word_scores = create_word_bigram_scores()  # 使用词和双词搭配作为特征
    best_words = find_best_words(word_scores, 4000)  # 特征维度1500
    load_data()
    posFeatures = pos_features(best_word_features, best_words)
    negFeatures = neg_features(best_word_features, best_words)
    cut_data(posFeatures, negFeatures)
    trainSet = posFeatures[1500:] + negFeatures[1500:]  # 使用了更多数据
    testSet = posFeatures[:500] + negFeatures[:500]
    test, tag_test = zip(*testSet)

    # 存储分类器
    def final_score(classifier):
        classifier = SklearnClassifier(classifier)
        classifier.train(trainSet)
        pred = classifier.classify_many(test)
        return accuracy_score(tag_test, pred)

    print(final_score(MultinomialNB()))  # 使用开发集中得出的最佳分类器


# 把分类器存储下来（存储分类器和前面没有区别，只是使用了更多的训练数据以便分类器更为准确）
def store_classifier():
    load_data()
    word_scores = create_word_bigram_scores()
    global best_words
    best_words = find_best_words(word_scores, 7500)

    posFeatures = pos_features(best_word_features)
    negFeatures = neg_features(best_word_features)

    trainSet = posFeatures + negFeatures

    MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
    MultinomialNB_classifier.train(trainSet)
    pickle.dump(MultinomialNB_classifier, open('src/step3/out/classifier.pkl', 'wb'))
    