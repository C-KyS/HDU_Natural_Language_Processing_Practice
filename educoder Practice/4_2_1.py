import jieba.posseg as psg
text=input()
#任务：使用jieba模块的函数对text完成词性标注并将结果存储到result变量中
# ********** Begin *********#

result = psg.cut(text)#进行分词
#for word, flag in words:
   # print("%s//%s" %word,flag)
for word, flag in result:
    print(f"{word}/{flag}", end=" ")


# ********** End **********#
#print(result)