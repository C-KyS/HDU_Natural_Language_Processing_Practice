import re
text=input()
list_ret=list()
#任务：完成对text文本的分句并输出结果
# ********** Begin *********#

split = re.split("[?!]\s",text)
for i in split:
    res = re.sub("\.$","",i)
    if re.search("\.\s[A-Z]",res):
        res = re.split("\.\s",res)
        for r in res:
            list_ret.append(r)
    else:
        list_ret.append(res)
print(list_ret)

 # ********** End **********#
