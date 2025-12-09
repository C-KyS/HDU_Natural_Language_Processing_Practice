import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pickle
import nltk
def Text_categorization():
    file_name=input()
    dataset = pd.read_csv('src/step2/data/'+file_name)
    # nltk.download('stopwords')
    stemmer = PorterStemmer()
    #数据预处理
    words = stopwords.words("english")
    dataset['cleaned'] = dataset['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
   
    vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    final_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
    
    from sklearn.linear_model import LogisticRegression
    X = dataset['cleaned']
    Y = dataset['category']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    # 任务：完成对逻辑回归的建模
    # ********** Begin *********#
    model = Pipeline([('vect', vectorizer),('chi', SelectKBest(chi2, k=1200)),('clf', LogisticRegression(random_state=0))]) # 构建逻辑回归分类器完成建模

    model.fit(X_train, y_train)
    # ********** End **********#

    ytest = np.array(y_test)
    return X_test,ytest,model