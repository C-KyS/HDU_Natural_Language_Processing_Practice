import jieba

jieba.setLogLevel(jieba.logging.INFO)  # 设置日志级别为 INFO 或更高
text= input()
words = jieba.lcut(text)  
data={} # 词典

# 任务：完成基于 Jieba 模块的词频统计
# ********** Begin *********#

#分词
words = jieba.cut(text , cut_all=False)

for char in words:
    if len(char)<2:
        continue
    if char in data:
        data[char]+=1
    else:
        data[char]=1 

# ********** End **********#
data = sorted(data.items(), key=lambda x: x[1], reverse=True)  # 排序
print(data[:3],end="")