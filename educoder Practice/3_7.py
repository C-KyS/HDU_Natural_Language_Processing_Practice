def cutA(sentence, dictA):
    # sentence：要分词的句子
    result = []
    sentenceLen = len(sentence)
    n = 0
    maxDictA = max([len(word) for word in dictA])
    # 任务：完成正向匹配算法的代码描述，并将结果保存到result变量中
    # result变量为分词结果
    # ********** Begin *********#
  
    while n < sentenceLen:
        matched = False
        for i in range(maxDictA, 0, -1):  # 从最长词开始匹配
            if n + i > sentenceLen:  # 如果超出句子长度，跳过
                continue
            word = sentence[n:n + i]
            if word in dictA:  # 如果匹配成功
                result.append(word)
                n += i
                matched = True
                break
        if not matched:  # 如果没有匹配成功，按单字切分
            result.append(sentence[n])
            n += 1
    
  
    # ********** End **********#
    print(result)  # 输出分词结果