import jieba
import pandas as pd

# 1. 修正词典加载方式
def load_dict(filepath):
    """加载词典文件为简单字典格式"""
    try:
        df = pd.read_csv(filepath, sep='\s+', header=None, quoting=3, on_bad_lines='skip')
        return dict(zip(df[0], df[1])) if len(df.columns) > 1 else set(df[0])
    except:
        with open(filepath, 'r', encoding='utf-8') as f:
            if '\t' in f.read(100):  # 检查是否是制表符分隔
                f.seek(0)
                return dict(line.strip().split('\t') for line in f)
            else:
                f.seek(0)
                return set(line.strip() for line in f)

# 加载词典
sentiment_dict = load_dict('./数据集/BosonNLP_sentiment_score.txt')
degree_dict = load_dict('./数据集/程度副词（中文）.txt')  # 作为集合使用
negation_dict = load_dict('./数据集/neg.txt')  # 作为集合使用
stopwords = load_dict('./数据集/stoplist.txt')  # 作为集合使用

# 2. 修正情感分数计算（处理浮点数）
sentiment_dict = {k: float(v) for k, v in sentiment_dict.items()}

# 分词函数（保持不变）
def segment(sentence):
    words = jieba.cut(sentence)
    return [word for word in words if word not in stopwords and len(word) > 1]

# 3. 改进得分计算逻辑
def calculate_word_score(word, prev_word):
    score = sentiment_dict.get(word, 0)
    
    # 处理程度副词（增强/减弱）
    if prev_word in degree_dict:
        if "非常" in prev_word or "极其" in prev_word:
            score *= 1.5
        elif "稍微" in prev_word or "有点" in prev_word:
            score *= 0.7
    
    # 处理否定词
    if prev_word in negation_dict:
        score *= -1
    
    return score

def calculate_sentence_score(sentence):
    words = segment(sentence)
    total_score = 0
    prev_word = None
    
    for word in words:
        total_score += calculate_word_score(word, prev_word)
        prev_word = word
    
    # 归一化处理（根据词语数量调整）
    word_count = len([w for w in words if w in sentiment_dict])
    return total_score / (word_count or 1)  # 避免除以零

# 测试
test_sentences = [
    "电影比预期要更恢宏磅礴",  # 应得正分
    "煽情显得太尴尬",       # 应得负分 
]

print("=== 情感分析结果 ===")
for sent in test_sentences:
    score = calculate_sentence_score(sent)
    print(f"句子: {sent}\n得分: {score:.2f} | 分词: {segment(sent)}\n")