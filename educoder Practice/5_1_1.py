# coding=utf-8


import re
import math


def indexOfSortedSuffix(doc, max_word_len):
    """
    Treat a suffix as an index where the suffix begins.
    Then sort these indexes by the suffixes.
    """
    indexes = []
    length = len(doc)
    for i in range(0, length):
        for j in range(i + 1, min(i + 1 + max_word_len, length + 1)):
            indexes.append((i, j))
    return sorted(indexes, key=lambda i_j: doc[i_j[0]:i_j[1]])


def genSubparts(string):
    """
    Partition a string into all possible two parts, e.g.
    given "abcd", generate [("a", "bcd"), ("ab", "cd"), ("abc", "d")]
    For string of length 1, return empty list
    """
    length = len(string)
    res = []
    for i in range(1, length):
        res.append((string[0:i], string[i:]))
    return res


def entropyOfList(ls):
    """
    Given a list of some items, compute entropy of the list
    The entropy is sum of -p[i]*log(p[i]) for every unique element i in the list, and p[i] is its frequency
    """
    elements = {}
    for e in ls:
        elements[e] = elements.get(e, 0) + 1
    # elements 字典以字为key，其value为这个字出现的频次
    length = float(len(ls))
    # 如果length是0，意味着词的某一边没有接其他字，就没有信息熵，返回0
    # 如果length不为零，按照公式计算信息熵
    if length:
        # 任务：按照说明实现信息熵的计算
        #********** Begin **********#
         
        entropy = 0.0
        
        for count in elements.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        '''
        for word in elements:
            p=elements[word]/length
            entropy -= p * math.log2(p)    
        '''
        #**********  End  **********#
        return entropy
    return length


class WordInfo(object):
    """
    Store information of each word, including its freqency, left neighbors and right neighbors
    """

    def __init__(self, text):
        super(WordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []
        self.right = []
        self.aggregation = 0

    def update(self, left, right):
        """
        Increase frequency of this word, then append left/right neighbors
        @param left a single character on the left side of this word
        @param right as left is, but on the right side
        """
        self.freq += 1
        if left: self.left.append(left)
        if right: self.right.append(right)

    def compute(self, length):
        """
        Compute frequency and entropy of this word
        @param length length of the document for training to get words
        """
        self.freq /= length     # 概率p(x,y)
        self.left = entropyOfList(self.left)    # 左熵
        self.right = entropyOfList(self.right)  # 右熵

    def computeAggregation(self, words_dict):
        """
        Compute aggregation of this word
        @param words_dict frequency dict of all candidate words
        words_dict是字典，其key为字符串，value为该字符串WordInfo类型数据，该数据类型记录了自己的出现频率
        """
        parts = genSubparts(self.text)
        if len(parts) > 0:
            # 任务：按照互信息公式计算此text的互信息，公式中的x，y指parts中text被分割成的两个部分。
            # ********** Begin **********#
            
            self.aggregation = 0.0
            for x, y in parts:
                px = words_dict[x].freq if x in words_dict else 0.0
                py = words_dict[y].freq if y in words_dict else 0.0
                pxy = self.freq
                if px > 0 and py > 0:
                    self.aggregation += pxy*math.log2(pxy / (px * py))
            
            
            # **********  End  **********#

class WordSegment(object):
    """
    Main class for Chinese word segmentation
    Generate words from a long enough document
    """

    def __init__(self, doc, max_word_len=5, min_freq=0.00005, min_entropy=2.0, min_aggregation=50):
        super(WordSegment, self).__init__()
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_aggregation = min_aggregation
        self.word_infos = self.genWords(doc)

        # Filter out the results satisfy all the requirements
        filter_func = lambda v: len(v.text) > 1 \
                                and v.aggregation > self.min_aggregation \
                                and v.freq > self.min_freq \
                                and v.left > self.min_entropy \
                                and v.right > self.min_entropy \
                                and v.text[0] != '的' \
                                and v.text[-1] != '的'
        self.word_with_freq = [(w.text, w.freq) for w in list(filter(filter_func, self.word_infos))]
        # 按照特定的规则将结果排序，确保输出的结果看起来较好
        tmp3 = sorted(self.word_with_freq, key=lambda w: len(w[0]) ** 2 * w[1], reverse=True)
        self.words = [w[0] for w in tmp3]


    def genWords(self, doc):
        """
        Generate all candidate words with their frequency/entropy/aggregation informations
        @param doc the document used for words generation
        """
        pattern = re.compile(u"[\\u4e00-\\u9fa5]+", re.U)   # 只保留汉字
        doc = ' '.join(pattern.findall(doc))
        suffix_indexes = indexOfSortedSuffix(doc, self.max_word_len)
        word_cands = {}
        # compute frequency and neighbors
        for suf in suffix_indexes:
            word = doc[suf[0]:suf[1]]
            if ' ' in word and word != ' ':
                continue
            if word not in word_cands:
                word_cands[word] = WordInfo(word)
            word_cands[word].update(doc[suf[0] - 1:suf[0]], doc[suf[1]:suf[1] + 1])
        # word_cands是以文本片段为key，以该文本片段的WordInfo数据类型为value的字典
        # compute probability and entropy
        length = len(doc)
        for k in word_cands:
            word_cands[k].compute(length)
        # compute aggregation of words whose length > 1
        values = sorted(list(word_cands.values()), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1: continue
            v.computeAggregation(word_cands)
        # 对values中的词v从大到小排序，排序的依据是互信息v.aggregation、左信息熵v.left、右信息熵v.right三者的和。
        sorted_values = sorted(values, key=lambda v: v.aggregation + v.left + v.right, reverse=True)
        return sorted_values


#if __name__ == '__main__':
#    # with open('吸烟有害健康.txt', 'r', encoding='utf8') as f:
#    # with open('淄博烧烤.txt', 'r', encoding='utf8') as f:
#    with open('西红柿炒鸡蛋.txt', 'r', encoding='utf8') as f:
#    # with open('红楼梦_曹雪芹.txt', 'r', encoding='utf8') as f:
#    # with open('不自由毋宁死.txt', 'r', encoding='utf8') as f:
#    # with open('算法工程师.txt', 'r', encoding='utf8') as f:
#        doc = f.read()
#    ws = WordSegment(doc, max_word_len=5, min_aggregation=math.log10(1.02), min_entropy=0.4)
#    print(' '.join(ws.words[:8]))