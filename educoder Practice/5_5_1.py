# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list

if __name__ == '__main__':
    text=input()
    result=""
    # 任务：使用停用词表去掉text文本中的停用词，并将结果保存至result变量
    # ********** Begin *********#
    stoplist = get_stopword_list()
    words = list(text)
    for word in words:
        if word not in stoplist:
            result+=word
        
    
    # ********** End **********#

    print(result,end="")