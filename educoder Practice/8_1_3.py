from snownlp import SnowNLP

def Analysis():
  text = input()
  result=0
# 任务：使用 SnowNLP 模块，对 text文本进行情感分析，将分析结果保存到result变量中
# ********** Begin *********#

  s = SnowNLP(text); # 对文本进行情感分析
  result = s.sentiments
#print("[sentiments]",s.sentiments); 


# ********** End **********#
  return result