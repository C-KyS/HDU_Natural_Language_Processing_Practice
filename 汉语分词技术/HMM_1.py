import json
from collections import defaultdict
import math


class HMMSegmenter:
    def __init__(self):
        # 状态集合：B(开头), M(中间), E(结尾), S(单独成词)
        self.states = ['B', 'M', 'E', 'S']

        # 初始概率
        self.start_p = {}

        # 转移概率
        self.trans_p = defaultdict(dict)

        # 发射概率
        self.emit_p = defaultdict(dict)

        # 状态频数
        self.state_count = defaultdict(int)

        # 最小概率，用于对数运算时的下界
        self.min_float = -3.14e100

    def train(self, train_file, model_file):
        """训练HMM模型

        Args:
            train_file: 训练文件路径
            model_file: 模型保存路径
        """
        # 初始化统计量
        for state in self.states:
            self.start_p[state] = 0.0
            for target in self.states:
                self.trans_p[state][target] = 0.0

        # 统计频数
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 将句子转换为观测序列和状态序列
                words = line.split()
                obs_seq = []
                state_seq = []
                for word in words:
                    if len(word) == 1:
                        state_seq.append('S')
                    else:
                        state_seq.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
                    obs_seq.extend(list(word))

                # 统计初始状态
                if state_seq:
                    self.start_p[state_seq[0]] += 1

                # 统计转移频数
                for i in range(len(state_seq) - 1):
                    current = state_seq[i]
                    next_state = state_seq[i + 1]
                    self.trans_p[current][next_state] += 1

                # 统计发射频数和状态频数
                for state, obs in zip(state_seq, obs_seq):
                    self.state_count[state] += 1
                    if obs in self.emit_p[state]:
                        self.emit_p[state][obs] += 1
                    else:
                        self.emit_p[state][obs] = 1

        # 计算初始概率（对数形式）
        total = sum(self.start_p.values())
        for state in self.start_p:
            if self.start_p[state] == 0:
                self.start_p[state] = self.min_float
            else:
                self.start_p[state] = math.log(self.start_p[state] / total)

        # 计算转移概率（对数形式）
        for state in self.trans_p:
            total = sum(self.trans_p[state].values())
            for target in self.trans_p[state]:
                if self.trans_p[state][target] == 0:
                    self.trans_p[state][target] = self.min_float
                else:
                    self.trans_p[state][target] = math.log(self.trans_p[state][target] / total)

        # 计算发射概率（对数形式）
        for state in self.emit_p:
            total = self.state_count[state]
            for obs in self.emit_p[state]:
                if self.emit_p[state][obs] == 0:
                    self.emit_p[state][obs] = self.min_float
                else:
                    self.emit_p[state][obs] = math.log(self.emit_p[state][obs] / total)

        # 保存模型
        model = {
            'start_p': self.start_p,
            'trans_p': {k: dict(v) for k, v in self.trans_p.items()},
            'emit_p': {k: dict(v) for k, v in self.emit_p.items()}
        }
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=4)

    def load_model(self, model_file):
        """加载训练好的模型"""
        with open(model_file, 'r', encoding='utf-8') as f:
            model = json.load(f)

        self.start_p = model['start_p']
        self.trans_p = defaultdict(dict)
        for k, v in model['trans_p'].items():
            self.trans_p[k] = v
        self.emit_p = defaultdict(dict)
        for k, v in model['emit_p'].items():
            self.emit_p[k] = v

    def viterbi(self, obs):
        """维特比算法

        Args:
            obs: 观测序列（即待分词的句子）

        Returns:
            最优状态序列
        """
        # 初始化
        V = [{}]  # 路径概率表
        path = {}  # 路径表

        # t=0时的初始概率
        for state in self.states:
            V[0][state] = self.start_p.get(state, self.min_float) + self.emit_p[state].get(obs[0], self.min_float)
            path[state] = [state]

        # t>0时的递推
        for t in range(1, len(obs)):
            V.append({})
            new_path = {}

            for curr_state in self.states:
                # 计算所有可能转移到当前状态的概率，取最大值
                max_prob = -math.inf
                best_prev_state = None

                for prev_state in self.states:
                    prob = V[t - 1][prev_state] + self.trans_p[prev_state].get(curr_state, self.min_float) + \
                           self.emit_p[curr_state].get(obs[t], self.min_float)

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                V[t][curr_state] = max_prob
                new_path[curr_state] = path[best_prev_state] + [curr_state]

            path = new_path

        # 找出最终最可能的状态
        last_state = max(V[-1].keys(), key=lambda state: V[-1][state])

        return path[last_state]

    def cut(self, text):
        """对文本进行分词

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        if not text:
            return []

        # 使用维特比算法得到最优状态序列
        state_seq = self.viterbi(text)

        # 根据状态序列进行分词
        result = []
        start = 0
        for i, state in enumerate(state_seq):
            if state == 'B':
                start = i
            elif state == 'E':
                result.append(text[start:i + 1])
            elif state == 'S':
                result.append(text[i])

        return result

    def segment_file(self, input_file, output_file):
        """对文件内容进行分词并保存结果

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue

                # 分词
                words = self.cut(line)
                f_out.write(' '.join(words) + '\n')


# 使用示例
if __name__ == '__main__':
    # 训练模型
    segmenter = HMMSegmenter()
    segmenter.train('./实训专题1/trainCorpus.txt', 'hmm_model.json')

    # 加载模型（如果已经训练过，可以直接加载）
    # segmenter = HMMSegmenter()
    # segmenter.load_model('hmm_model.json')

    # 对文件进行分词
    segmenter.segment_file('./实训专题1/flightnews.txt', 'flightnews_segmented.txt')