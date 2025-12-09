import re

# 任务1：抽取years_string中所有的年份并输出
years_string = input()
# ********** Begin *********#

years = re.findall(r'\b\d{4}\b', years_string)  # 匹配4位数字的年份

# ********** End **********#
print(years)

# 任务2：匹配text_string中包含“文本”的句子，并使用print输出，以句号作为分隔
text_string = input()
regex = '文本'
# ********** Begin *********#
sentences = re.split(r'[。！？]', text_string)  # 按照句号、叹号、问号分割句子
for sentence in sentences:
    if re.search(regex, sentence):  # 检查句子是否包含“文本”
        print(sentence.strip())  # 输出句子并去除首尾空格

# ********** End **********#