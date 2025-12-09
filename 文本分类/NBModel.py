import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题
from wordcloud import WordCloud

##数据导入及数据预处理
# 加载自定义词典和停用词
jieba.load_userdict('newdic1.txt')
with open('stopword.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# 读取数据（这里假设数据量很大，我们只读取前10万条）
data = pd.read_csv('message80W.csv', nrows=100000)
data.columns = ['id', 'label', 'text']  # 假设列名是这样

# 数据去重
data = data.drop_duplicates(subset=['text'])

# 查看数据分布
print(data['label'].value_counts())



##文本预处理与分词
# 分词函数
def chinese_word_cut(text):
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords and len(word) > 1])

# 应用分词
data['cut_text'] = data['text'].apply(chinese_word_cut)

# 分离垃圾短信和常规短信
spam = data[data['label'] == 1]['cut_text']
ham = data[data['label'] == 0]['cut_text']

# 生成词云
def generate_wordcloud(text, title):
    # 设置支持中文的字体（确保系统中存在该字体）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    wordcloud = WordCloud(font_path='simhei.ttf', 
                         background_color='white',
                         max_words=200).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# 垃圾短信词云
generate_wordcloud(' '.join(spam.tolist()), '垃圾短信高频词')
# 常规短信词云
generate_wordcloud(' '.join(ham.tolist()), '常规短信高频词')


##特征提取与模型训练
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['cut_text'], 
                                                    data['label'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# 特征提取 - 词袋模型
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 训练朴素贝叶斯模型
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

##模型评估
# 预测
y_pred = nb.predict(X_test_vec)

# 评估
print("准确率:", accuracy_score(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 可视化重要特征
def plot_important_features(vectorizer, model, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.feature_log_prob_
    top_spam = np.argsort(coefs[1])[-n:]
    top_ham = np.argsort(coefs[0])[-n:]
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.barh(range(n), coefs[1][top_spam])
    plt.yticks(range(n), [feature_names[i] for i in top_spam])
    plt.title("垃圾短信最具代表性特征词")
    
    plt.subplot(1, 2, 2)
    plt.barh(range(n), coefs[0][top_ham])
    plt.yticks(range(n), [feature_names[i] for i in top_ham])
    plt.title("常规短信最具代表性特征词")
    plt.tight_layout()
    plt.show()

plot_important_features(vectorizer, nb)

