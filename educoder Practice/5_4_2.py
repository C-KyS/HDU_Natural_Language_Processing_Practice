from jieba import analyse
text = input() # 原始文本
# 任务：使用jieba模块中有关TextRank算法的模块完成对text中前三个关键字的提取并输出
# ********** Begin *********#
textrank = analyse.textrank  
keywords = textrank(text)  

s=''
count = 0
for keyword in keywords:
    s+=keyword+' '
    count+=1
    if count==3:
        break

print(s)

# ********** End **********#