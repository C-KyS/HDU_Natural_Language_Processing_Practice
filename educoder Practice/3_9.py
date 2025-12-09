class BiMM():
    def __init__(self):
        self.window_size = 3  # 字典中最长词数

    def MMseg(self, text, dict): # 正向最大匹配算法
        result = []
        index = 0
        text_length = len(text)
        while text_length > index:
            for size in range(self.window_size + index, index, -1):
                piece = text[index:size]
                if piece in dict:
                    index = size - 1
                    break
            index += 1
            result.append(piece)
        return result

    def RMMseg(self, text, dict): # 逆向最大匹配算法
        result = []
        index = len(text)
        while index > 0:
            for size in range(index - self.window_size, index):
                piece = text[size:index]
                if piece in dict:
                    index = size + 1
                    break
            index = index - 1
            result.append(piece)
        result.reverse()
        return result

    def main(self, text, r1, r2):
    # 任务：完成双向最大匹配算法的代码描述
    # ********** Begin *********#
    
        #正向分词，所有分词结果
        allWordsMM = len(r1)
        #正向分词，单字分词结果
        singleWordsMM = sum(1 for word in r1 if(len(word)==1))

         #逆向分词，所有分词结果
        allWordsRMM = len(r2)
        #逆向分词，单字分词结果
        singleWordsRMM = sum(1 for word in r2 if len(word)==1)
        
        #结果比较
        if allWordsMM < allWordsRMM:
            print(r1)
        elif allWordsMM > allWordsRMM:
            print(r2)
        else:
            if singleWordsMM < singleWordsRMM:
                print(r1)
            else:
                print(r2)
    # ********** End **********#